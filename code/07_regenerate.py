#!/usr/bin/env python3
"""
07_regenerate.py -- regenerate the corpus at a larger token cap.

Three modes:

  --mode validate   regenerate at the ORIGINAL cap (200) and diff against
                    generations/*.csv. This is the gate: if replies do not
                    match, the prompt construction differs and nothing
                    downstream is comparable. Run this FIRST.

  --mode generate   the real run, at --max-new-tokens (default 1024).

  --mode prefix     regenerate a subsample at 200 tokens and diff against the
                    1024-token outputs truncated to 200 tokens. Confirms the
                    prefix property holds on this hardware, which is what
                    lets you derive every smaller cap from one run.

Resumable: writes a checkpoint per model; rerunning skips finished models.

IMPORTANT -- PROMPT CONSTRUCTION
The original generation script (01/02) is not in the repo. Set SYSTEM_PROMPT
and build_prompt() below to EXACTLY what the original run used, recovered from
your Kaggle notebook history. --mode validate exists to prove you got it right.
"""
import argparse, json, os, sys, time, glob, hashlib
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

# ---------------------------------------------------------------------------
# EDIT THIS to match the original run exactly.
SYSTEM_PROMPT = None          # set from --mode autodetect, or fill in by hand

# Candidate constructions tried by --mode autodetect. The original generation
# script (01/02) is not in the repo, so rather than guess we search this grid
# for the one that reproduces the released generations exactly.
CANDIDATES = [
    ("no system prompt",        None),
    ("helpful assistant",       "You are a helpful assistant."),
    ("helpful honest harmless", "You are a helpful, honest and harmless assistant."),
    ("Qwen default",            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    ("concise assistant",       "You are a helpful assistant. Answer concisely."),
]
# ---------------------------------------------------------------------------


def build_prompt(tok, user_text):
    msgs = []
    # Gemma-2 chat template rejects a system role; fold it into the user turn.
    if SYSTEM_PROMPT:
        if "gemma" in tok.name_or_path.lower():
            user_text = f"{SYSTEM_PROMPT}\n\n{user_text}"
        else:
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def find(name):
    here = Path(__file__).resolve().parent
    pats = [f"/kaggle/input/**/{name}", f"**/{name}",
            str(here / "**" / name), str(here.parent / "**" / name)]
    for p in pats:
        c = glob.glob(p, recursive=True)
        if c:
            return c[0]
    raise FileNotFoundError(
        f"Could not find {name}. Attach it as a Kaggle input, or run from the "
        f"repo root so it is reachable.")


def load_prompts():
    p = pd.read_csv(find("register_pairs_300.csv"))
    rows = []
    for _, r in p.iterrows():
        for fr, col in [("eval_framing", "eval_framing"),
                        ("deploy_framing", "deploy_framing")]:
            rows.append(dict(pair_id=r.pair_id, category=r.category,
                             framing=fr, prompt=r[col]))
    return pd.DataFrame(rows)


def load_model(name, token):
    tok = AutoTokenizer.from_pretrained(name, token=token, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if "gemma-2" in name:
        # soft-capping is only correct under eager attention
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        name, token=token, device_map="auto", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        **kw)
    model.eval()
    return tok, model


@torch.inference_mode()
def run_model(name, prompts, max_new, batch_size, token):
    tok, model = load_model(name, token)
    texts = [build_prompt(tok, p) for p in prompts.prompt]
    # sort by length so batches are homogeneous -> far less padding waste
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt",
                  padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        for j, i in enumerate(idx):
            seq = new[j]
            keep = seq[seq != tok.pad_token_id]
            n_tok = int(keep.numel())
            out[i] = dict(
                reply=tok.decode(keep, skip_special_tokens=True).strip(),
                n_new_tokens=n_tok,
                finish_reason="length" if n_tok >= max_new else "stop")
        done = s + len(idx)
        el = time.time() - t0
        print(f"  {done}/{len(order)}  {el:6.0f}s  eta {el/done*(len(order)-done):6.0f}s",
              flush=True)
    del model
    torch.cuda.empty_cache()
    r = prompts.copy()
    for k in ["reply", "n_new_tokens", "finish_reason"]:
        r[k] = [o[k] for o in out]
    r["model"] = name
    r["max_new_tokens"] = max_new
    return r


def env_stamp():
    import transformers, accelerate
    try:
        import bitsandbytes; bnb = bitsandbytes.__version__
    except Exception:
        bnb = "n/a"
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                transformers=transformers.__version__,
                accelerate=accelerate.__version__, bitsandbytes=bnb)


@torch.inference_mode()
def autodetect(prompts, token, n=12, batch_size=4):
    """Find the prompt construction that reproduces the released generations."""
    global SYSTEM_PROMPT
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    old = pd.concat([pd.read_csv(find("qwen_generations.csv")),
                     pd.read_csv(find("gemma_generations.csv"))])
    old = old[old.model == name]
    sub = prompts.merge(old[["pair_id", "framing", "reply"]],
                        on=["pair_id", "framing"]).head(n)
    if not len(sub):
        sys.exit("autodetect: no overlap between prompts and released generations")
    print(f"autodetect: {len(sub)} prompts on {name}\n")
    best, best_rate = None, -1.0
    for label, cand in CANDIDATES:
        SYSTEM_PROMPT = cand
        r = run_model(name, sub[["pair_id", "category", "framing", "prompt"]],
                      200, batch_size, token)
        rate = float((r.reply.str.strip().values == sub.reply.str.strip().values).mean())
        print(f"  {label:26s} exact match {rate:.2f}")
        if rate > best_rate:
            best, best_rate = (label, cand), rate
    print(f"\nbest: {best[0]!r} at {best_rate:.2f}")
    if best_rate >= 0.9:
        print(f"Set SYSTEM_PROMPT = {best[1]!r} and run --mode validate.")
    else:
        print("No candidate reproduces the originals. The difference is probably")
        print("not the system prompt: try --batch-size 1, or check dtype and the")
        print("transformers/bitsandbytes versions recorded in env_*.json.")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["autodetect", "validate", "generate", "prefix"],
                    default="autodetect")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--sample", type=int, default=0,
                    help="limit prompts per model (prefix/smoke tests)")
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "GPU OFF -- set Accelerator to GPU T4 x2."
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN")

    cap = 200 if a.mode in ("validate", "prefix") else a.max_new_tokens
    stamp = env_stamp()
    print(json.dumps(stamp, indent=2), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(stamp, open(f"{a.out}/env_{a.mode}_{cap}.json", "w"), indent=2)

    prompts = load_prompts()

    if a.mode == "autodetect":
        autodetect(prompts, token, batch_size=a.batch_size)
        return

    if a.sample:
        prompts = prompts.sample(a.sample, random_state=0).reset_index(drop=True)

    parts = []
    for m in a.models:
        ck = f"{a.out}/gen_{m.split('/')[-1]}_{cap}.csv"
        if os.path.exists(ck):
            print(f"skip {m} (checkpoint exists)", flush=True)
            parts.append(pd.read_csv(ck)); continue
        print(f"\n=== {m} @ {cap} tokens ===", flush=True)
        r = run_model(m, prompts, cap, a.batch_size, token)
        r.to_csv(ck, index=False)
        parts.append(r)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(f"{a.out}/generations_{cap}.csv", index=False)
    print(f"\nwrote generations_{cap}.csv  ({len(df)} rows)")
    print("truncated (finish_reason=length): %.1f%%"
          % (100 * (df.finish_reason == "length").mean()))

    if a.mode == "validate":
        old = pd.concat([pd.read_csv(find("qwen_generations.csv")),
                         pd.read_csv(find("gemma_generations.csv"))])
        j = df.merge(old, on=["model", "pair_id", "framing"],
                     suffixes=("_new", "_old"))
        exact = (j.reply_new.str.strip() == j.reply_old.str.strip()).mean()
        print(f"\nVALIDATION: exact reply match {exact:.4f} on {len(j)} rows")
        if exact < 0.99:
            print("FAIL -- prompt construction differs from the original run.")
            print("Fix SYSTEM_PROMPT / build_prompt() before running --mode generate.")
            print("If close but not exact, try --batch-size 1 (padding affects logits).")
            sys.exit(1)
        print("PASS -- pipeline reproduces the released generations.")

    if a.mode == "prefix":
        big = pd.read_csv(find(f"generations_{a.max_new_tokens}.csv"))
        j = df.merge(big, on=["model", "pair_id", "framing"],
                     suffixes=("_200", "_big"))
        ok = [r.reply_big.startswith(r.reply_200[:200]) for _, r in j.iterrows()]
        print(f"\nPREFIX CHECK: {sum(ok)}/{len(ok)} agree")
        if sum(ok) < len(ok):
            print("Prefix property does NOT hold -- derive caps by regeneration, "
                  "not truncation.")


if __name__ == "__main__":
    main()
