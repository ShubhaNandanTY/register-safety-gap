#!/usr/bin/env python3
"""
physframe_common.py -- shared inference engine for the PhysFrame pipeline.

Used by 03_generate_responses.py and 05_judge.py. Keep it in the same folder
as those scripts (they import it by name).

What it does
  * Runs one model over a list of chat prompts, one worker process per GPU
    (Kaggle "GPU T4 x2" => two workers, each with its own copy of the model
    and half of the prompts). Roughly 2x the throughput of device_map="auto".
  * Decodes greedily in rounds and drops finished rows from the batch between
    rounds, so one rambling response no longer holds 31 finished ones hostage
    until max_new_tokens. A self-check at start-up compares this against plain
    generate(); if they differ on this model/library version it falls back.
  * Picks a batch size from free GPU memory and halves it on CUDA OOM.
  * Appends every finished batch to disk, so a killed Kaggle session resumes
    where it stopped (rerun the same command).
  * Records SPEED, split into the parts that are usually conflated:

        download_s    fetching weights from the Hub (0 when cached)
        load_s        from_pretrained: weights -> GPU, incl. 4-bit quantisation
        warmup_s      first generate() call: CUDA init, kernel selection and,
                      with --compile, torch.compile. This is the "compile" cost.
        selfcheck_s   the chunked-vs-plain decode equivalence check
        gen_s         steady-state generation only (GPU-synchronised)
        new_tok_per_s generated tokens / gen_s   <- the inference speed
        items_per_s, s_per_item, total_tok_per_s (prompt+new), peak_mem_gb

    Files written to --out:
        speed_log.csv       one row per (model, GPU worker) + one "all" row
        speed_batches.csv   one row per batch (batch size, tokens, seconds)

Prompt construction differs from code/07_regenerate.py in one deliberate way:
the chat-templated text is tokenised with add_special_tokens=False. The
template already contains <bos> for Gemma, so the old default produced a
double <bos> on Gemma-2. Physics is a new corpus, so there is nothing to
stay byte-compatible with.

An OpenAI-compatible HTTP backend (run_api) is included for a closed model,
either as a judge or as the extra "strong" test model.
"""
import argparse, csv, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

LABELS = ["CORRECT", "RIGHT_ANSWER_WRONG_REASONING", "PARTIAL", "WRONG"]
CELLS = ["BENCH", "BENCH_PLACEBO", "FORMAL_MOTIVE", "CASUAL", "CASUAL_MOTIVE"]

SPEED_COLS = [
    "timestamp", "stage", "tag", "model", "backend", "worker", "gpu_name",
    "n_gpus", "quant", "compute_dtype", "attn", "decode_mode", "compile",
    "batch_size", "batch_size_final", "oom_backoffs", "max_new_tokens",
    "n_items", "prompt_tokens", "new_tokens", "download_s", "load_s",
    "warmup_s", "selfcheck_s", "gen_s", "wall_s", "new_tok_per_s",
    "total_tok_per_s", "items_per_s", "s_per_item", "peak_mem_gb"]
BATCH_COLS = ["stage", "tag", "model", "worker", "batch", "batch_size",
              "prompt_tokens_max", "new_tokens", "new_tokens_max", "seconds"]


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
SEARCH = []      # scripts append their --out folder; it is searched first

# The released dataset folder (physframe/data/) uses short names. Accept them,
# but only inside a folder that also holds problems.jsonl, so an unrelated
# prompts.csv elsewhere in /kaggle/input is never picked up.
ALIASES = {"physframe_prompts_1500.csv": "prompts.csv",
           "physframe_answer_key_300.csv": "answer_key.csv",
           "physframe_problems_300.jsonl": "problems.jsonl"}


def find(name, required=True):
    """Locate a file: an explicit path, then --out, /kaggle/working (fresh
    results) before /kaggle/input (attached datasets), then around the repo.
    Prints the path it used, so a stale copy cannot be picked up unnoticed."""
    here = Path(__file__).resolve().parent
    pats = [name] + [f"{d}/**/{name}" for d in SEARCH] + [
        f"/kaggle/working/**/{name}", f"/kaggle/input/**/{name}", f"**/{name}",
        str(here / "**" / name), str(here.parent / "**" / name)]
    alias = ALIASES.get(name)
    for p in pats:
        hits = [h for h in glob.glob(p, recursive=True) if Path(h).is_file()]
        if alias:
            hits += [h for h in glob.glob(p.replace(name, alias), recursive=True)
                     if Path(h).is_file()
                     and (Path(h).parent / "problems.jsonl").is_file()]
        if hits:
            hit = max(hits, key=os.path.getmtime)
            print(f"  using {hit}", flush=True)
            return hit
    if required:
        sys.exit(f"Could not find {name}. Attach it as a Kaggle input, or run "
                 f"from the repo root so it is reachable.")
    return None


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)


def model_slug(name):
    """Qwen/Qwen2.5-7B-Instruct -> qwen2.5-7b, google/gemma-2-9b-it -> gemma2-9b
    (the convention already used by response_id in the safety release)."""
    s = str(name).rstrip("/").split("/")[-1].lower()
    s = re.sub(r"-(instruct|it|chat)$", "", s)
    return s.replace("gemma-2-", "gemma2-").replace("gemma-3-", "gemma3-")


def response_id(model, problem_id, cell):
    return f"{model_slug(model)}__{problem_id}__{cell}"


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def gpu_name():
    """GPU model via nvidia-smi, so the parent process never opens a CUDA
    context (that would take memory from the worker on GPU 0)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return out.strip().split("\n")[0] or "cpu"
    except Exception:
        return "cpu"


def env_stamp():
    import torch, transformers
    out = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=gpu_name(),
               n_gpus=n_gpus(), transformers=transformers.__version__)
    for mod in ("accelerate", "bitsandbytes"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "n/a"
    return out


def append_rows(path, cols, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass            # a line cut off by a killed session; redo it
    return out


def _rate(a, b):
    return round(a / b, 3) if b else 0.0


def print_speed(rows):
    hdr = (f"  {'worker':8s}{'items':>7s}{'new tok':>9s}{'load s':>8s}"
           f"{'warmup s':>9s}{'gen s':>8s}{'tok/s':>8s}{'items/s':>9s}{'mem GB':>8s}")
    print(hdr)
    for r in rows:
        print(f"  {str(r['worker']):8s}{r['n_items']:7d}{r['new_tokens']:9d}"
              f"{r['load_s']:8.1f}{r['warmup_s']:9.1f}{r['gen_s']:8.1f}"
              f"{r['new_tok_per_s']:8.1f}{r['items_per_s']:9.2f}"
              f"{r['peak_mem_gb']:8.2f}", flush=True)


# --------------------------------------------------------------------------
# worker: one process, one GPU (or all GPUs with span), one model
# --------------------------------------------------------------------------
def build_prompt(tok, messages):
    """Apply the chat template. Templates that reject a system role (Gemma-2)
    get the system text folded into the first user turn, as in 07_regenerate."""
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False)
    except Exception:
        sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [dict(m) for m in messages if m["role"] != "system"]
        if sys_txt and rest:
            rest[0]["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
        return tok.apply_chat_template(rest, add_generation_prompt=True,
                                       tokenize=False)


def load_model(spec):
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name, token = spec["model"], hf_token()
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(name, token=token, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cdt = getattr(torch, spec.get("compute_dtype") or "bfloat16")
    kw = {}
    attn = spec.get("attn") or "auto"
    if attn == "auto" and "gemma-2" in name.lower():
        attn = "eager"          # as in 07_regenerate: soft-capping under eager
    if attn != "auto":
        kw["attn_implementation"] = attn
    quant = spec.get("quant") or "4bit"
    if not cuda:
        quant, cdt = "none", torch.float32
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=cdt,
                               bnb_4bit_use_double_quant=True)
            if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
    if cuda:
        kw["device_map"] = "auto" if spec.get("span") else {"": 0}
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw[dkey] = cdt
    model = AutoModelForCausalLM.from_pretrained(name, token=token, **kw).eval()
    return tok, model, attn, quant


def eos_ids(tok, model):
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(i for i in ids if i is not None)


def kv_bytes_per_token(model, cdt_bytes=2):
    c = model.config
    layers = getattr(c, "num_hidden_layers", 24)
    heads = getattr(c, "num_attention_heads", 16)
    kvh = getattr(c, "num_key_value_heads", None) or heads
    hd = getattr(c, "head_dim", None) or max(1, getattr(c, "hidden_size", 1024) // heads)
    return layers * 2 * kvh * hd * cdt_bytes


def mem_budget():
    """Bytes available for batches: 70% of what the weights left free on the
    fullest GPU. Call once, right after loading."""
    import torch
    if not torch.cuda.is_available():
        return None
    return 0.7 * min(torch.cuda.get_device_properties(i).total_memory
                     - torch.cuda.memory_allocated(i)
                     for i in range(torch.cuda.device_count()))


def auto_batch(model, seq_len, cap, budget):
    """Largest batch whose KV cache (x2 for activations) fits in the budget.
    Sized per batch from its longest prompt; OOM back-off covers the rest."""
    if budget is None:
        return min(cap, 8)
    per_seq = 2 * kv_bytes_per_token(model) * seq_len
    return int(max(1, min(cap, budget // per_seq)))


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cut(row, eos):
    """tokens up to the first EOS; finish_reason 'stop' if one was found."""
    for i, t in enumerate(row):
        if t in eos:
            return row[:i], "stop"
    return row, "length"


def gen_plain(model, tok, enc, max_new, eos, gkw):
    out = model.generate(**enc, max_new_tokens=max_new, **gkw)
    return [_cut(r, eos) for r in out[:, enc["input_ids"].shape[1]:].tolist()]


def gen_chunked(model, tok, enc, max_new, eos, gkw, chunk):
    """Greedy decode in rounds of `chunk` tokens. After each round, rows that
    reached EOS leave the batch and the KV cache; the rest continue from the
    cache. Same tokens as one generate() call, minus the wasted padding steps."""
    import torch
    ids, mask = enc["input_ids"], enc["attention_mask"]
    P, B = ids.shape[1], ids.shape[0]
    eos_t = torch.tensor(sorted(eos), device=ids.device)
    alive = list(range(B))
    done = [None] * B
    past, produced = None, 0
    while alive:
        step = min(chunk, max_new - produced)
        o = model.generate(input_ids=ids, attention_mask=mask,
                           past_key_values=past, max_new_tokens=step,
                           return_dict_in_generate=True, **gkw)
        seq = o.sequences
        got = seq.shape[1] - ids.shape[1]
        produced += got
        new = seq[:, P:]
        fin = torch.isin(new, eos_t).any(dim=1)
        if produced >= max_new or got < step:
            fin[:] = True
        fin_l = fin.tolist()
        rows = new.tolist()
        for j, a in enumerate(alive):
            if fin_l[j]:
                done[a] = _cut(rows[j], eos)
        if all(fin_l):
            break
        keep = (~fin).nonzero().squeeze(1)
        alive = [a for j, a in enumerate(alive) if not fin_l[j]]
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], got))], dim=1)
        past = o.past_key_values
        if len(alive) < seq.shape[0]:
            ids, mask = seq[keep], mask[keep]
            past.batch_select_indices(keep)
        else:
            ids = seq
    return done


def worker(spec):
    import torch
    stage, tag, name = spec["stage"], spec["tag"], spec["model"]
    wid = spec["worker"]
    pre = f"[{wid} {model_slug(name)}]"
    cuda = torch.cuda.is_available()

    t0 = time.perf_counter()
    tok, model, attn, quant = load_model(spec)
    _sync()
    load_s = time.perf_counter() - t0
    budget = mem_budget()
    eos = set(eos_ids(tok, model))
    gkw = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
               pad_token_id=tok.pad_token_id)
    print(f"{pre} loaded in {load_s:.1f}s  quant={quant} attn={attn}", flush=True)

    items = read_jsonl(spec["todo"])[spec["shard"]::spec["n_shards"]]
    texts = [build_prompt(tok, it["messages"]) for it in items]
    ntok = [len(x) for x in tok(texts, add_special_tokens=False).input_ids] if texts else []
    order = sorted(range(len(texts)), key=lambda i: -ntok[i])   # longest first
    max_new = spec["max_new_tokens"]

    def encode(idx):
        dev = model.device if cuda else "cpu"
        return tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                   add_special_tokens=False, return_token_type_ids=False).to(dev)

    compiled, orig_forward = "off", model.forward
    if spec.get("compile"):
        try:
            model.forward = torch.compile(model.forward, dynamic=True)
            compiled = "on"
        except Exception as e:
            compiled = f"failed:{type(e).__name__}"

    # ---- warm-up: the first call pays CUDA init / kernel selection / compile
    warmup_s = selfcheck_s = 0.0
    decode_mode = spec.get("decode") or "chunked"
    with torch.inference_mode():
        if order:
            w = order[:2]
            t0 = time.perf_counter()
            try:
                gen_plain(model, tok, encode(w), 8, eos, gkw)
            except Exception as e:
                if compiled != "on":
                    raise
                model.forward, compiled = orig_forward, f"failed:{type(e).__name__}"
                gen_plain(model, tok, encode(w), 8, eos, gkw)
            _sync()
            warmup_s = time.perf_counter() - t0
            print(f"{pre} warm-up {warmup_s:.1f}s  compile={compiled}", flush=True)

        # ---- self-check: chunked decode must reproduce plain generate()
        if order and decode_mode == "chunked":
            t0 = time.perf_counter()
            try:
                c = order[:4]
                n_chk = min(48, max_new)
                a = gen_plain(model, tok, encode(c), n_chk, eos, gkw)
                b = gen_chunked(model, tok, encode(c), n_chk, eos, gkw, 16)
                if a != b:
                    decode_mode = "plain(selfcheck-mismatch)"
            except Exception as e:
                decode_mode = f"plain(selfcheck-{type(e).__name__})"
            _sync()
            selfcheck_s = time.perf_counter() - t0
            print(f"{pre} decode mode: {decode_mode}  ({selfcheck_s:.1f}s)", flush=True)
    chunked = decode_mode == "chunked"

    fixed, cap = spec.get("batch_size") or 0, spec.get("batch_cap", 64)
    scale, backoffs, bs0, bs = 1.0, 0, None, 1
    if cuda:
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)

    # a session killed mid-write can leave a torn last line; start on a fresh one
    if os.path.exists(spec["part"]) and os.path.getsize(spec["part"]):
        with open(spec["part"], "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            torn = fh.read(1) != b"\n"
        if torn:
            with open(spec["part"], "a", encoding="utf-8") as fh:
                fh.write("\n")
    out_f = open(spec["part"], "a", encoding="utf-8")
    gen_s, n_new, n_prompt, n_done, b_idx, i = 0.0, 0, 0, 0, 0, 0
    batch_rows, last_dt, stopped = [], 0.0, False
    deadline = spec.get("deadline") or 0
    with torch.inference_mode():
        while i < len(order):
            # stop early enough that the next batch cannot run past the deadline
            if deadline and time.time() + 1.3 * last_dt + 60 > deadline:
                stopped = True
                print(f"{pre} time budget reached: stopping with {n_done}/{len(order)} done "
                      f"(saved; the next run resumes from here)", flush=True)
                break
            want = fixed or auto_batch(model, ntok[order[i]] + max_new, cap, budget)
            bs = max(1, int(want * scale))
            bs0 = bs0 or bs
            idx = order[i:i + bs]
            enc = None
            try:
                enc = encode(idx)
                _sync()
                t0 = time.perf_counter()
                res = (gen_chunked(model, tok, enc, max_new, eos, gkw, spec.get("chunk", 64))
                       if chunked else gen_plain(model, tok, enc, max_new, eos, gkw))
                _sync()
                dt = time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                del enc
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                scale, backoffs = scale / 2, backoffs + 1
                print(f"{pre} CUDA OOM at batch size {bs} -> halving", flush=True)
                continue
            new_here = 0
            for j, k in enumerate(idx):
                toks, why = res[j]
                new_here += len(toks)
                out_f.write(json.dumps(dict(
                    id=items[k]["id"],
                    text=tok.decode(toks, skip_special_tokens=True).strip(),
                    n_new_tokens=len(toks), finish_reason=why,
                    n_prompt_tokens=ntok[k], batch_size=len(idx)),
                    ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            last_dt = dt
            gen_s += dt
            n_new += new_here
            n_prompt += sum(ntok[k] for k in idx)
            n_done += len(idx)
            batch_rows.append(dict(
                stage=stage, tag=tag, model=name, worker=wid, batch=b_idx,
                batch_size=len(idx), prompt_tokens_max=ntok[idx[0]],
                new_tokens=new_here, new_tokens_max=max(len(r[0]) for r in res),
                seconds=round(dt, 3)))
            b_idx += 1
            i += len(idx)
            eta = gen_s / n_done * (len(order) - n_done)
            print(f"{pre} {n_done}/{len(order)}  bs={len(idx)}  "
                  f"{_rate(n_new, gen_s):.0f} tok/s  eta {eta:5.0f}s", flush=True)
    out_f.close()

    row = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, tag=tag,
        model=name, backend="local", worker=wid,
        gpu_name=torch.cuda.get_device_name(0) if cuda else "cpu",
        n_gpus=torch.cuda.device_count() if cuda else 0, quant=quant,
        compute_dtype=spec.get("compute_dtype") or "bfloat16", attn=attn,
        decode_mode=decode_mode, compile=compiled, batch_size=bs0 or 0,
        batch_size_final=bs, oom_backoffs=backoffs, max_new_tokens=max_new,
        n_items=n_done, prompt_tokens=n_prompt, new_tokens=n_new,
        download_s=0.0, load_s=round(load_s, 2), warmup_s=round(warmup_s, 2),
        selfcheck_s=round(selfcheck_s, 2), gen_s=round(gen_s, 2), wall_s=0.0,
        new_tok_per_s=_rate(n_new, gen_s),
        total_tok_per_s=_rate(n_new + n_prompt, gen_s),
        items_per_s=_rate(n_done, gen_s), s_per_item=_rate(gen_s, n_done),
        peak_mem_gb=round(max(torch.cuda.max_memory_allocated(d) for d in
                              range(torch.cuda.device_count())) / 2**30, 2) if cuda else 0.0)
    json.dump(dict(row=row, batches=batch_rows, stopped=stopped), open(spec["speed"], "w"))


# --------------------------------------------------------------------------
# parent: orchestration, resume, merge, speed log
# --------------------------------------------------------------------------
def _download(name):
    if os.path.isdir(name):
        return 0.0
    from huggingface_hub import snapshot_download
    t0 = time.perf_counter()
    try:
        snapshot_download(name, token=hf_token(),
                          allow_patterns=["*.json", "*.safetensors", "*.model",
                                          "*.txt", "*.jinja"])
    except Exception as e:
        sys.exit(f"Could not download {name}: {type(e).__name__}: {str(e)[:200]}\n"
                 f"Check that Internet is on. For gated models (Gemma) accept the "
                 f"licence on the Hub and add HF_TOKEN under Add-ons > Secrets.")
    return time.perf_counter() - t0


def free_cache_if_low(name, min_free_gb):
    """Kaggle's disk is small. Drop this model's Hub cache when space is short."""
    if os.path.isdir(name):
        return
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = Path(HF_HUB_CACHE) / ("models--" + name.replace("/", "--"))
        free = shutil.disk_usage(HF_HUB_CACHE).free / 2**30
        if d.exists() and free < min_free_gb:
            shutil.rmtree(d, ignore_errors=True)
            print(f"  freed Hub cache for {name} (only {free:.0f} GB was free)", flush=True)
    except Exception as e:
        print(f"  cache cleanup skipped: {type(e).__name__}", flush=True)


def _collect(out_dir, tag, slug, items):
    want = {it["id"] for it in items}
    res = {}
    for p in sorted(glob.glob(f"{out_dir}/parts/{tag}__{slug}__*.jsonl")):
        if p.endswith(".todo.jsonl"):
            continue
        for r in read_jsonl(p):
            if r.get("id") in want:
                res[r["id"]] = r
    return res


def run_local(stage, tag, model, items, out_dir, max_new_tokens, batch_size=0,
              batch_cap=64, quant="4bit", compute_dtype="bfloat16", attn="auto",
              decode="chunked", chunk=64, compile=False, span=False,
              min_free_gb=25, deadline=0):
    """Run `model` over items = [{"id": str, "messages": [...]}, ...].
    Returns {id: {text, n_new_tokens, finish_reason, n_prompt_tokens, batch_size}}.
    Resumable: finished ids are read back from <out_dir>/parts."""
    slug = model_slug(model)
    parts = Path(out_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    res = _collect(out_dir, tag, slug, items)
    todo = [it for it in items if it["id"] not in res]
    if not todo:
        print(f"skip {model} ({len(res)} items already done)", flush=True)
        return res
    print(f"\n=== {stage}: {model}  {len(todo)} to do, {len(res)} done ===", flush=True)
    wall0 = time.perf_counter()
    download_s = _download(model)
    print(f"  download {download_s:.1f}s", flush=True)

    g = n_gpus()
    n_workers = 1 if (span or g <= 1) else g
    # interleave by length so every shard sees the same length mix
    todo.sort(key=lambda it: -sum(len(m["content"]) for m in it["messages"]))
    todo_path = str(parts / f"{tag}__{slug}.todo.jsonl")
    with open(todo_path, "w", encoding="utf-8") as f:
        for it in todo:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    procs, speed_files = [], []
    stamp = time.strftime("%H%M%S")
    for k in range(n_workers):
        wid = "gpus" if span else (f"gpu{k}" if g else "cpu")
        spec = dict(stage=stage, tag=tag, model=model, worker=wid, shard=k,
                    n_shards=n_workers, todo=todo_path,
                    part=str(parts / f"{tag}__{slug}__s{k}of{n_workers}.jsonl"),
                    speed=str(parts / f"{tag}__{slug}__s{k}.{stamp}.speed.json"),
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                    batch_cap=batch_cap, quant=quant, compute_dtype=compute_dtype,
                    attn=attn, decode=decode, chunk=chunk, compile=compile,
                    span=span, deadline=deadline)
        sp = str(parts / f"{tag}__{slug}__s{k}.spec.json")
        json.dump(spec, open(sp, "w"))
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if hf_token():
            env["HF_TOKEN"] = hf_token()
        if g and not span:
            env["CUDA_VISIBLE_DEVICES"] = str(k)
        procs.append(subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", sp], env=env))
        speed_files.append(spec["speed"])
    codes = [p.wait() for p in procs]
    wall_s = time.perf_counter() - wall0

    rows, batches, stopped = [], [], False
    for sf in speed_files:
        if os.path.exists(sf):
            d = json.load(open(sf))
            stopped = stopped or d.get("stopped", False)
            d["row"].update(download_s=round(download_s, 2), wall_s=round(wall_s, 2))
            rows.append(d["row"])
            batches += d["batches"]
    if rows:
        if len(rows) > 1:
            gs = max(r["gen_s"] for r in rows)       # workers ran in parallel
            tot = dict(rows[0], worker="all",
                       n_items=sum(r["n_items"] for r in rows),
                       prompt_tokens=sum(r["prompt_tokens"] for r in rows),
                       new_tokens=sum(r["new_tokens"] for r in rows),
                       load_s=max(r["load_s"] for r in rows),
                       warmup_s=max(r["warmup_s"] for r in rows),
                       selfcheck_s=max(r["selfcheck_s"] for r in rows),
                       gen_s=gs, oom_backoffs=sum(r["oom_backoffs"] for r in rows),
                       peak_mem_gb=max(r["peak_mem_gb"] for r in rows))
            tot.update(new_tok_per_s=_rate(tot["new_tokens"], gs),
                       total_tok_per_s=_rate(tot["new_tokens"] + tot["prompt_tokens"], gs),
                       items_per_s=_rate(tot["n_items"], gs),
                       s_per_item=_rate(gs, tot["n_items"]))
            rows.append(tot)
        append_rows(f"{out_dir}/speed_log.csv", SPEED_COLS, rows)
        append_rows(f"{out_dir}/speed_batches.csv", BATCH_COLS, batches)
        print(f"  speed: {model}  (download {download_s:.1f}s, wall {wall_s:.1f}s)")
        print_speed(rows)

    res = _collect(out_dir, tag, slug, items)
    if any(codes):
        sys.exit(f"{model}: worker exit codes {codes}. {len(res)}/{len(items)} items "
                 f"are saved; rerun the same command to resume.")
    if len(res) < len(items):
        if stopped:
            print(f"  {model}: stopped at the time budget with {len(res)}/{len(items)} "
                  f"done. Everything finished is saved under parts/.", flush=True)
            return res
        sys.exit(f"{model}: {len(items) - len(res)} items are still missing although "
                 f"the workers exited cleanly; rerun the same command.")
    free_cache_if_low(model, min_free_gb)
    return res


# --------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (closed models)
# --------------------------------------------------------------------------
def run_api(stage, tag, model, items, out_dir, max_new_tokens, api_base,
            api_key, concurrency=8, retries=6):
    """POST {api_base}/chat/completions for each item, temperature 0.
    Works with any OpenAI-compatible endpoint. Resumable like run_local."""
    import urllib.request, urllib.error
    from concurrent.futures import ThreadPoolExecutor
    slug = model_slug(model)
    parts = Path(out_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    res = _collect(out_dir, tag, slug, items)
    todo = [it for it in items if it["id"] not in res]
    if not todo:
        print(f"skip {model} ({len(res)} items already done)", flush=True)
        return res
    print(f"\n=== {stage}: {model} via API  {len(todo)} to do ===", flush=True)
    url = api_base.rstrip("/") + "/chat/completions"
    tok_key = {"name": "max_tokens"}

    def call(it):
        t0 = time.perf_counter()
        err = None
        for a in range(retries):
            body = {"model": model, "messages": it["messages"],
                    "temperature": 0, tok_key["name"]: max_new_tokens}
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                ch, us = d["choices"][0], d.get("usage") or {}
                return dict(id=it["id"], text=(ch["message"].get("content") or "").strip(),
                            n_new_tokens=us.get("completion_tokens", 0),
                            finish_reason=ch.get("finish_reason") or "stop",
                            n_prompt_tokens=us.get("prompt_tokens", 0), batch_size=1,
                            seconds=round(time.perf_counter() - t0, 3))
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                err = f"HTTP {e.code}: {msg}"
                if e.code == 400 and "max_completion_tokens" in msg:
                    tok_key["name"] = "max_completion_tokens"
                    continue
                if e.code in (400, 401, 403, 404):
                    break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** a))
        return dict(id=it["id"], error=err)

    part = parts / f"{tag}__{slug}__api.jsonl"
    t0 = time.perf_counter()
    n_ok = n_new = n_prompt = n_err = 0
    last_err = None
    with open(part, "a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as ex:
        for n, r in enumerate(ex.map(call, todo), 1):
            if "error" in r:
                n_err, last_err = n_err + 1, r["error"]
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n_ok, n_new, n_prompt = n_ok + 1, n_new + r["n_new_tokens"], n_prompt + r["n_prompt_tokens"]
            if n % 50 == 0 or n == len(todo):
                print(f"  [api] {n}/{len(todo)}  errors={n_err}", flush=True)
    gs = time.perf_counter() - t0
    row = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, tag=tag,
               model=model, backend="api", worker=f"api x{concurrency}", gpu_name="",
               n_gpus=0, quant="", compute_dtype="", attn="", decode_mode="api",
               compile="off", batch_size=concurrency, batch_size_final=concurrency,
               oom_backoffs=0, max_new_tokens=max_new_tokens, n_items=n_ok,
               prompt_tokens=n_prompt, new_tokens=n_new, download_s=0.0, load_s=0.0,
               warmup_s=0.0, selfcheck_s=0.0, gen_s=round(gs, 2), wall_s=round(gs, 2),
               new_tok_per_s=_rate(n_new, gs), total_tok_per_s=_rate(n_new + n_prompt, gs),
               items_per_s=_rate(n_ok, gs), s_per_item=_rate(gs, n_ok), peak_mem_gb=0.0)
    append_rows(f"{out_dir}/speed_log.csv", SPEED_COLS, [row])
    print(f"  speed: {model} (API)")
    print_speed([row])
    res = _collect(out_dir, tag, slug, items)
    if n_err:
        sys.exit(f"{model}: {n_err} API calls failed (last: {last_err}). "
                 f"{len(res)}/{len(items)} saved; rerun to resume.")
    return res


def import_parts(src, out_dir):
    """Copy finished batches (parts/*.jsonl) from an earlier run into out_dir.
    src='auto' searches every attached input under /kaggle/input."""
    if not src or src == "none":
        return 0
    root = "/kaggle/input" if src == "auto" else src
    dst = Path(out_dir) / "parts"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in glob.glob(f"{root}/**/parts/*.jsonl", recursive=True):
        if f.endswith(".todo.jsonl"):
            continue
        name = Path(f).name
        target = dst / name
        k = 1
        while target.exists():               # never overwrite; _collect merges by id
            if target.read_bytes() == Path(f).read_bytes():
                break
            target = dst / f"{Path(name).stem}__prev{k}.jsonl"
            k += 1
        if not target.exists():
            shutil.copy(f, target)
            n += 1
    if n:
        print(f"resume: copied {n} part files from earlier runs under {root}", flush=True)
    return n


def start_backup(a, out_dir, default_name):
    """Restore from Google Drive and start the periodic upload. The final upload
    runs at exit, including after sys.exit."""
    import atexit
    name = a.drive_run or default_name
    # The optional backup helpers (email_backup.py, drive_sync.py) are not part of
    # this release; without them the run simply writes to out_dir.
    try:
        if not a.no_email:
            import email_backup
            atexit.register(email_backup.start(out_dir, name, a.email_every_h).close)
    except ImportError:
        pass
    try:
        if not a.no_drive:
            import drive_sync
            atexit.register(drive_sync.start(out_dir, name, a.drive_every_min).close)
    except ImportError:
        pass


def deadline_from(hours, t0):
    return t0 + hours * 3600 if hours and hours > 0 else 0


def add_engine_args(ap, batch_cap):
    """Speed/engine flags shared by 03 and 05."""
    ap.add_argument("--stop-after-h", type=float, default=11.0,
                    help="stop cleanly after this many hours so a Kaggle commit (12 h "
                         "limit) still saves its output; 0 = no limit")
    ap.add_argument("--resume-from", default="auto",
                    help="folder with parts/ from an earlier run; 'auto' searches "
                         "/kaggle/input; 'none' disables")
    ap.add_argument("--drive-run", default="",
                    help="Google Drive folder name for this run's backup (needs the "
                         "GDRIVE_* secrets); empty = a name built from the settings")
    ap.add_argument("--drive-every-min", type=float, default=10,
                    help="minutes between Drive uploads")
    ap.add_argument("--no-drive", action="store_true", help="turn the Drive backup off")
    ap.add_argument("--email-every-h", type=float, default=2.0,
                    help="hours between checkpoint emails (needs the EMAIL_* secrets)")
    ap.add_argument("--no-email", action="store_true", help="turn the email backup off")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = choose from free GPU memory (halves itself on OOM)")
    ap.add_argument("--batch-cap", type=int, default=batch_cap)
    ap.add_argument("--quant", choices=["4bit", "8bit", "none"], default="4bit")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float16"],
                    default="bfloat16",
                    help="float16 is often faster on a T4 but overflows on Gemma-2")
    ap.add_argument("--attn", choices=["auto", "eager", "sdpa"], default="auto",
                    help="auto = eager for Gemma-2 (as in 07_regenerate), default otherwise")
    ap.add_argument("--decode", choices=["chunked", "plain"], default="chunked")
    ap.add_argument("--chunk", type=int, default=64,
                    help="tokens per round before finished rows are dropped")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the forward pass; compile time lands in warmup_s")
    ap.add_argument("--min-free-gb", type=float, default=25,
                    help="delete a model's Hub cache after use if disk is below this")


def engine_kwargs(a):
    return dict(batch_size=a.batch_size, batch_cap=a.batch_cap, quant=a.quant,
                compute_dtype=a.compute_dtype, attn=a.attn, decode=a.decode,
                chunk=a.chunk, compile=a.compile, min_free_gb=a.min_free_gb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", help="internal: path to a worker spec")
    a = ap.parse_args()
    if not a.worker:
        sys.exit("This module is imported by 03_generate_responses.py and 05_judge.py.")
    worker(json.load(open(a.worker)))
