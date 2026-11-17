#!/usr/bin/env python3
"""
03_generate_responses.py -- run the test models over the PhysFrame prompts.

Physics counterpart of code/07_regenerate.py. Same six models, 4-bit, greedy,
but 5 framings per problem and max_new_tokens=1024 (200 truncates derivations).

  --mode pilot      ~50 BENCH problems per model (2 per template). Scores the
                    final answers and prints accuracy per model against the
                    30-70% target, so models at the floor or ceiling can be
                    dropped BEFORE the full run. Run this first.

  --mode generate   the full run: every model x 1,500 prompts.

Kaggle (Accelerator: GPU T4 x2, Internet on, HF_TOKEN in Secrets for Gemma):

    !python 03_generate_responses.py --mode pilot
    !python 03_generate_responses.py --mode generate
    !python 03_generate_responses.py --mode generate --models Qwen/Qwen2.5-7B-Instruct

Resumable at batch granularity: every finished batch is on disk, so after a
session dies rerun the same command and it carries on. To continue in a new
session, add the previous /kaggle/working output as an input dataset and pass
--resume-from /kaggle/input/<that dataset>.

Outputs (to --out, default /kaggle/working):
    generations_physframe_{cap}.csv         response_id, model, problem_id,
                                            cell, response, n_new_tokens,
                                            finish_reason, ...
    generations_physframe_pilot_{cap}.csv   the pilot subset
    pilot_accuracy.csv                      pilot accuracy per model
    speed_log.csv, speed_batches.csv        load / warm-up(compile) / inference
                                            speed, see physframe_common.py
    env_generate_{cap}.json                 library and GPU stamp

A closed or hosted model can be added as a seventh "strong" model through any
OpenAI-compatible endpoint:

    !python 03_generate_responses.py --mode generate --models \
        --api-model <name> --api-base https://.../v1 --api-key-secret MY_KEY
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physframe_common as C  # noqa: E402

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

# None = each model's own chat-template default. Recorded in the env stamp.
SYSTEM_PROMPT = None


def load_prompts():
    p = pd.read_csv(C.find("physframe_prompts_1500.csv"))
    need = {"prompt_id", "problem_id", "template_id", "topic", "cell", "prompt"}
    if need - set(p.columns):
        sys.exit(f"physframe_prompts_1500.csv is missing columns {sorted(need - set(p.columns))}")
    bad = set(p.cell) - set(C.CELLS)
    if bad:
        sys.exit(f"unexpected framing cells: {sorted(bad)}")
    return p


def pilot_subset(p, per_template):
    """BENCH only, the first `per_template` problems of each template."""
    b = p[p.cell == "BENCH"].sort_values("problem_id")
    return b.groupby("template_id", sort=True).head(per_template).reset_index(drop=True)


def items_for(model, prompts):
    out = []
    for r in prompts.itertuples():
        msgs = []
        if SYSTEM_PROMPT:
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})
        msgs.append({"role": "user", "content": r.prompt})
        out.append({"id": C.response_id(model, r.problem_id, r.cell), "messages": msgs})
    return out


def to_frame(model, prompts, res, cap):
    rows = []
    for r in prompts.itertuples():
        rid = C.response_id(model, r.problem_id, r.cell)
        if rid not in res:
            continue
        o = res[rid]
        rows.append(dict(
            response_id=rid, model=model, problem_id=r.problem_id,
            template_id=r.template_id, topic=r.topic, cell=r.cell,
            prompt_id=r.prompt_id, n_words=getattr(r, "n_words", None),
            response=o["text"], n_new_tokens=o["n_new_tokens"],
            finish_reason=o["finish_reason"], max_new_tokens=cap,
            n_prompt_tokens=o.get("n_prompt_tokens"),
            batch_size=o.get("batch_size")))
    return pd.DataFrame(rows)


def load_scorer():
    path = Path(__file__).resolve().parent / "04_score_final_answer.py"
    spec = importlib.util.spec_from_file_location("pf_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pilot_report(df, out, lo, hi):
    sc = load_scorer()
    s = sc.score_frame(df, sc.load_key())
    j = df.merge(s, on="response_id")
    j["truncated"] = (j.finish_reason == "length").astype(int)
    rep = (j.groupby("model")
            .agg(n=("final_correct", "size"), accuracy=("final_correct", "mean"),
                 any_match=("any_match", "mean"),
                 no_answer=("final_status", lambda x: (x == "NO_ANSWER").mean()),
                 truncated=("truncated", "mean"),
                 mean_new_tokens=("n_new_tokens", "mean")).reset_index())
    rep["verdict"] = ["keep" if lo <= a <= hi else
                      "floor: consider dropping" if a < lo else
                      "ceiling: consider dropping" for a in rep.accuracy]
    rep.to_csv(f"{out}/pilot_accuracy.csv", index=False)
    print(f"\n=== PILOT: final-answer accuracy on BENCH (target {lo:.0%}-{hi:.0%}) ===")
    print(rep.round(3).to_string(index=False))
    keep = rep[rep.verdict == "keep"].model.tolist()
    print(f"\n{len(keep)} of {len(rep)} models in range.")
    if keep:
        print("full run on those:\n  python 03_generate_responses.py --mode generate "
              "--models " + " ".join(keep))
    if rep.truncated.max() > 0.05:
        print("NOTE: more than 5% of some model's responses hit the token cap; "
              "check finish_reason before trusting its accuracy.")
    print("The scorer is an instrument too: read a few rows of the pilot CSV "
          "next to pilot_accuracy.csv before deciding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "generate"], default="pilot")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--pilot-per-template", type=int, default=2,
                    help="BENCH problems per template in the pilot (2 x 25 = 50)")
    ap.add_argument("--pilot-range", nargs=2, type=float, default=[0.30, 0.70])
    ap.add_argument("--cells", nargs="*", default=C.CELLS)
    ap.add_argument("--sample", type=int, default=0,
                    help="limit prompts per model (smoke tests)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-secret", default="OPENAI_API_KEY",
                    help="name of the Kaggle secret / env var holding the key")
    ap.add_argument("--api-concurrency", type=int, default=8)
    C.add_engine_args(ap, batch_cap=64)
    a = ap.parse_args()
    t0 = time.time()
    deadline = C.deadline_from(a.stop_after_h, t0)

    out = a.out
    Path(out).mkdir(parents=True, exist_ok=True)
    C.SEARCH.append(a.out)
    slugs = "-".join(C.model_slug(m) for m in a.models) or "api"
    C.start_backup(a, out, f"gen_{a.compute_dtype}_{slugs}"[:120])
    C.import_parts(a.resume_from, out)

    cap = a.max_new_tokens
    if a.models and C.n_gpus() == 0:
        print("WARNING: no GPU visible -- set Accelerator to GPU T4 x2. "
              "Continuing on CPU without quantisation (smoke tests only).")
    if a.models:
        stamp = C.env_stamp()
        stamp.update(system_prompt=SYSTEM_PROMPT, max_new_tokens=cap, mode=a.mode,
                     quant=a.quant, compute_dtype=a.compute_dtype, decode=a.decode)
        print(json.dumps(stamp, indent=2), flush=True)
        json.dump(stamp, open(f"{out}/env_generate_{cap}.json", "w"), indent=2)

    prompts = load_prompts()
    prompts = prompts[prompts.cell.isin(a.cells)]
    if a.mode == "pilot":
        prompts = pilot_subset(prompts, a.pilot_per_template)
    if a.sample:
        prompts = prompts.sample(min(a.sample, len(prompts)), random_state=0)
    prompts = prompts.reset_index(drop=True)
    print(f"{a.mode}: {len(prompts)} prompts x {len(a.models) + bool(a.api_model)} models")

    # precision and quantisation are part of the tag, so a resumed run never
    # mixes responses generated under different settings
    tag = f"gen{cap}-{a.compute_dtype}-{a.quant}"
    parts, unfinished = [], []
    for m in a.models:
        items = items_for(m, prompts)
        # a model needs time to download and load; skip it if little is left
        if deadline and time.time() > deadline - min(15 * 60, 0.1 * a.stop_after_h * 3600):
            print(f"\nskip {m}: not enough time left in this run", flush=True)
            res = C._collect(out, tag, C.model_slug(m), items)
        else:
            res = C.run_local("generate", tag, m, items, out, cap,
                              deadline=deadline, **C.engine_kwargs(a))
        if len(res) < len(items):
            unfinished.append(f"{m} ({len(res)}/{len(items)})")
        parts.append(to_frame(m, prompts, res, cap))
    if a.api_model:
        key = C.secret(a.api_key_secret)
        if not key:
            sys.exit(f"no API key: add a Kaggle secret named {a.api_key_secret}")
        res = C.run_api("generate", tag, a.api_model, items_for(a.api_model, prompts),
                        out, cap, a.api_base, key, a.api_concurrency)
        parts.append(to_frame(a.api_model, prompts, res, cap))

    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        sys.exit("NOT FINISHED (time budget): nothing generated yet. Attach this "
                 "version's Output and run again to continue.")
    name = (f"generations_physframe_pilot_{cap}.csv" if a.mode == "pilot"
            else f"generations_physframe_{cap}.csv")
    # a partial --models run must not wipe the other models from the full file
    path = f"{out}/{name}"
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.response_id.isin(df.response_id)], df],
                       ignore_index=True)
    df.to_csv(path, index=False)
    print(f"\nwrote {name}  ({len(df)} rows, {df.model.nunique()} models)")
    print("truncated (finish_reason=length): %.1f%%"
          % (100 * (df.finish_reason == "length").mean()))
    print(df.groupby("model").n_new_tokens.agg(["count", "mean", "max"]).round(0).to_string())

    if os.path.exists(f"{out}/speed_log.csv"):
        sp = pd.read_csv(f"{out}/speed_log.csv")
        sp = sp[(sp.stage == "generate") & sp.worker.isin(["all", "gpus", "cpu"])
                | (sp.stage == "generate") & (sp.n_gpus <= 1)]
        cols = ["model", "worker", "n_items", "download_s", "load_s", "warmup_s",
                "gen_s", "new_tok_per_s", "items_per_s", "peak_mem_gb", "decode_mode"]
        print("\n=== SPEED (latest rows; full log in speed_log.csv) ===")
        print(sp.groupby("model").tail(1)[cols].to_string(index=False))

    if unfinished:
        print("\n" + "=" * 70 + "\nNOT FINISHED (time budget): " + ", ".join(unfinished)
              + "\nEverything done so far is saved. To continue: open this notebook, "
                "Add Input -> this version's Output, then Save & Run All again.\n" + "=" * 70)
    elif a.mode == "generate":
        print("\nALL DONE: every model has all its responses.")
    if a.mode == "pilot" and not unfinished:
        pilot_report(df, out, *a.pilot_range)


if __name__ == "__main__":
    main()
