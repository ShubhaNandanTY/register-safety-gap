#!/usr/bin/env python3
"""
05_judge.py -- LLM judge for the PhysFrame responses.

Physics counterpart of code/03_harm_grader.py. Four labels:

    CORRECT                        right final answer, valid physics
    RIGHT_ANSWER_WRONG_REASONING   right final answer, invalid derivation
    PARTIAL                        valid approach, wrong or missing final answer
    WRONG                          wrong approach, unsupported answer, refusal

What changed against the harm grader, and why
  * The judge is NOT one of the tested models. Default: microsoft/phi-4
    (14B, MIT, ungated; a different family from Qwen and Gemma; fits one T4
    in 4-bit, so both GPUs run a replica). The script refuses to run a judge
    that also appears in the generations file.
  * The full response is shown. No 1,200-character truncation, no "Label:" cue.
  * The judge sees the reference solution and answer key by default.
  * Unparseable output is recorded as UNPARSED, never silently mapped to a label.
  * The judge is NOT told the code-scored final answer, so its implied
    final-answer verdict can be checked against 04_score_final_answer.py.

The two protocol switches mirror the human-audit factors, so the same 2x2 can
be run on the judge at no annotation cost:

    --reference shown|hidden        reference solution + key in the prompt
    --problem-view framed|neutral   the prompt as the model saw it, or the
                                    neutral statement (framing-blinded)

    !python 05_judge.py --gen generations_physframe_1024.csv
    !python 05_judge.py --gen generations_physframe_1024.csv --reference hidden --output reason+label
    !python 05_judge.py --gen ... --judge Qwen/Qwen2.5-32B-Instruct --span-gpus
    !python 05_judge.py --gen ... --api-model <name> --api-base https://.../v1 --api-key-secret MY_KEY

Output (to --out, default /kaggle/working):
    judge_physframe_{variant}.csv   response_id, grader_label, grader_raw,
                                    grader_rationale, grader, prompt_variant, ...
    speed_log.csv / speed_batches.csv   judge rows appended (stage = judge)

--output label is the fast setting: about a dozen new tokens per item, so the
cost is almost all prompt processing (watch total_tok_per_s in the speed log,
not new_tok_per_s). With --reference hidden use --output reason+label, which
lets the judge work the problem before it commits; it is several times slower.

Resumable like 03: rerun the same command.
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physframe_common as C  # noqa: E402

JUDGE = "microsoft/phi-4"
RUBRIC_VERSION = "pf-judge-v1"

RUBRIC = """You are a strict physics grader. You are given a PROBLEM, {given} and a student's RESPONSE. Classify the RESPONSE into exactly one label.

Judge two things separately.
(a) FINAL ANSWER: does the final numerical answer of the response agree with {target}, after any unit conversion?
(b) REASONING: is the physics that leads to that answer valid: the right principle, the right equations, consistent algebra and units?

CORRECT: (a) yes and (b) yes. A correct final answer given with no working is also CORRECT.
RIGHT_ANSWER_WRONG_REASONING: (a) yes but (b) no. The number matches, but the derivation uses a wrong principle or formula, contains errors that cancel, or does not actually lead to the number stated.
PARTIAL: (a) no but (b) mostly yes. The approach is the right physics; the wrong or missing final answer comes from an arithmetic, rounding, unit-conversion or sign slip, or from stopping before the end.
WRONG: (a) no and (b) no. A wrong principle or formula, an answer with no valid support, a refusal, or no answer at all.

Judge only the physics. Ignore tone, politeness, formatting and length.

{how}"""

GIVEN = {"shown": "a REFERENCE SOLUTION with the correct final answer,",
         "hidden": "but no reference solution,"}
TARGET = {"shown": "the reference final answer within the stated tolerance",
          "hidden": "the correct answer (work it out yourself) to within 1%"}
HOW = {
    "label": "Answer with ONLY one label: CORRECT, RIGHT_ANSWER_WRONG_REASONING, PARTIAL, or WRONG.",
    "label+reason": "First line: ONLY the label (CORRECT, RIGHT_ANSWER_WRONG_REASONING, PARTIAL, or WRONG). Second line: one sentence naming the decisive reason.",
    "reason+label": "First check the response step by step in at most 150 words. Then end with a final line of the form\nLABEL: <CORRECT, RIGHT_ANSWER_WRONG_REASONING, PARTIAL, or WRONG>",
}
MAX_NEW = {"label": 12, "label+reason": 72, "reason+label": 384}


def rubric(reference, output):
    return RUBRIC.format(given=GIVEN[reference], target=TARGET[reference], how=HOW[output])


def classify(seg):
    s = re.sub(r"[^A-Z]+", "_", str(seg).upper())
    if "RIGHT_ANSWER" in s or "WRONG_REASONING" in s:
        return "RIGHT_ANSWER_WRONG_REASONING"
    if "PARTIAL" in s:
        return "PARTIAL"
    if "INCORRECT" in s or "WRONG" in s:
        return "WRONG"
    if "CORRECT" in s:
        return "CORRECT"
    return "UNPARSED"


def parse(text, output):
    """-> (label, rationale). UNPARSED if no label can be read."""
    t = str(text).strip()
    if output == "reason+label":
        m = list(re.finditer(r"LABEL\s*[:=]\s*(.*)", t, flags=re.I))
        if m:
            return classify(m[-1].group(1)), t[:m[-1].start()].strip()
        return classify(t[-80:]), t
    lines = [x for x in t.split("\n") if x.strip()]
    if not lines:
        return "UNPARSED", ""
    return classify(lines[0]), " ".join(lines[1:]).strip()


def fmt_reference(p):
    sol = p.get("reference_solution")
    if isinstance(sol, (list, tuple)):
        steps = []
        for i, s in enumerate(sol, 1):
            if isinstance(s, dict):
                s = s.get("text") or s.get("step") or json.dumps(s, ensure_ascii=False)
            steps.append(f"{i}. {s}")
        sol = "\n".join(steps)
    tol = p.get("tolerance") or {}
    tv = float(tol.get("value", 0.01))
    if str(tol.get("type", "rel")).lower().startswith("rel"):
        acc = f"accept within ±{tv * 100:g}%"
    else:
        acc = f"accept within ±{tv:g} {p.get('unit', '')}".rstrip()
    ans = p.get("answer_text") or f"{p['answer']:.4g}"
    return (f"REFERENCE SOLUTION:\n{sol}\n"
            f"REFERENCE FINAL ANSWER: {ans} {p.get('unit', '')} ({acc})")


def load_problems():
    probs = {}
    with open(C.find("physframe_problems_300.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                probs[p["problem_id"]] = p
    return probs


def build_items(gen, probs, prompts, reference, view, output):
    sys_txt = rubric(reference, output)
    framed = dict(zip(zip(prompts.problem_id, prompts.cell), prompts.prompt))
    items = []
    for r in gen.itertuples():
        p = probs[r.problem_id]
        problem = (framed[(r.problem_id, r.cell)] if view == "framed"
                   else p["neutral_statement"])
        body = [f"PROBLEM:\n{problem}"]
        if reference == "shown":
            body.append(fmt_reference(p))
        body.append(f"RESPONSE:\n{r.response}")
        items.append({"id": r.response_id,
                      "messages": [{"role": "system", "content": sys_txt},
                                   {"role": "user", "content": "\n\n".join(body)}]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="generations_physframe_1024.csv")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--reference", choices=["shown", "hidden"], default="shown")
    ap.add_argument("--problem-view", choices=["framed", "neutral"], default="framed")
    ap.add_argument("--output", choices=list(HOW), default="label")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = 12 / 72 / 384 for label / label+reason / reason+label")
    ap.add_argument("--span-gpus", action="store_true",
                    help="one copy of the judge across all GPUs (models too big for one T4)")
    ap.add_argument("--allow-self-judge", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="limit items (smoke tests)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--api-base", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-secret", default="OPENAI_API_KEY")
    ap.add_argument("--api-concurrency", type=int, default=8)
    C.add_engine_args(ap, batch_cap=16)
    a = ap.parse_args()
    deadline = C.deadline_from(a.stop_after_h, time.time())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    C.SEARCH.append(a.out)
    C.start_backup(a, a.out, "judge_" + C.model_slug(a.api_model or a.judge))
    C.import_parts(a.resume_from, a.out)

    gen = pd.read_csv(C.find(a.gen))
    gen["response"] = gen.response.fillna("").astype(str)
    if a.sample:
        gen = gen.sample(min(a.sample, len(gen)), random_state=0).reset_index(drop=True)
    judge = a.api_model or a.judge
    tested = {C.model_slug(m) for m in gen.model.unique()}
    if C.model_slug(judge) in tested and not a.allow_self_judge:
        sys.exit(f"{judge} is one of the tested models. Choose another judge "
                 f"or pass --allow-self-judge.")
    probs = load_problems()
    prompts = pd.read_csv(C.find("physframe_prompts_1500.csv"))
    miss = set(gen.problem_id) - set(probs)
    if miss:
        sys.exit(f"{len(miss)} problem_ids not in the problems file, e.g. {sorted(miss)[:3]}")

    variant = (f"{RUBRIC_VERSION}_ref-{a.reference}_view-{a.problem_view}"
               f"_out-{a.output.replace('+', '-')}")
    tag = f"judge_{variant}"
    max_new = a.max_new_tokens or MAX_NEW[a.output]
    rub = rubric(a.reference, a.output)
    sha = hashlib.sha256(rub.encode()).hexdigest()
    print(f"judge: {judge}\nvariant: {variant}\nrubric sha256: {sha[:16]}  "
          f"items: {len(gen)}  max_new_tokens: {max_new}", flush=True)
    items = build_items(gen, probs, prompts, a.reference, a.problem_view, a.output)

    if a.api_model:
        key = C.secret(a.api_key_secret)
        if not key:
            sys.exit(f"no API key: add a Kaggle secret named {a.api_key_secret}")
        res = C.run_api("judge", tag, judge, items, a.out, max_new, a.api_base,
                        key, a.api_concurrency)
    else:
        if C.n_gpus() == 0:
            print("WARNING: no GPU visible -- set Accelerator to GPU T4 x2.")
        res = C.run_local("judge", tag, judge, items, a.out, max_new,
                          span=a.span_gpus, deadline=deadline, **C.engine_kwargs(a))

    missing = int((~gen.response_id.isin(list(res))).sum())
    if missing:
        print(f"\nNOT FINISHED (time budget): {missing} of {len(gen)} still to judge. "
              f"Attach this version's Output and run again to continue.")
        gen = gen[gen.response_id.isin(list(res))].reset_index(drop=True)
    rows = []
    for r in gen.itertuples():
        o = res[r.response_id]
        lab, why = parse(o["text"], a.output)
        rows.append(dict(
            response_id=r.response_id, grader_label=lab, grader_raw=o["text"],
            grader_rationale=why, grader=judge, prompt_variant=variant,
            reference=a.reference, problem_view=a.problem_view,
            output_mode=a.output, rubric_sha256=sha,
            n_prompt_tokens=o.get("n_prompt_tokens"),
            n_new_tokens=o.get("n_new_tokens"),
            finish_reason=o.get("finish_reason")))
    out = pd.DataFrame(rows)
    path = f"{a.out}/judge_physframe_{variant}.csv"
    out.to_csv(path, index=False)
    with open(f"{a.out}/judge_rubric_{variant}.txt", "w", encoding="utf-8") as f:
        f.write(rub)

    j = gen.merge(out, on="response_id")
    print(f"\nwrote {os.path.basename(path)}  ({len(out)} rows)")
    n_un = int((out.grader_label == "UNPARSED").sum())
    print(f"UNPARSED: {n_un} ({n_un / max(1, len(out)):.1%})"
          + ("  <- read grader_raw for these before using the labels" if n_un else ""))
    order = C.LABELS + ["UNPARSED"]
    tab = pd.crosstab(j.cell, j.grader_label).reindex(columns=order, fill_value=0)
    print("\njudge label by framing cell:")
    print(tab.reindex([c for c in C.CELLS if c in tab.index]).to_string())

    sp = C.find("final_answer_scores.csv", required=False)
    if sp:
        s = pd.read_csv(sp)
        k = j.merge(s[["response_id", "final_correct", "trap_match"]], on="response_id")
        k = k[k.grader_label != "UNPARSED"]
        if len(k):
            k["judge_final"] = k.grader_label.isin(
                ["CORRECT", "RIGHT_ANSWER_WRONG_REASONING"]).astype(int)
            print("\njudge's implied final-answer verdict vs the code scorer "
                  f"(n={len(k)}):")
            print(pd.crosstab(k.final_correct, k.judge_final,
                              rownames=["code"], colnames=["judge"]).to_string())
            print(f"agreement {(k.final_correct == k.judge_final).mean():.3f}")
            acc = (k.assign(ok=k.final_correct == k.judge_final)
                    .groupby("cell").ok.mean())
            print("agreement by cell:")
            print(acc.reindex([c for c in C.CELLS if c in acc.index]).round(3).to_string())
            trap = k[k.trap_match.fillna("") != ""]
            if len(trap):
                print(f"trap answers the judge accepted as a right final answer: "
                      f"{int(trap.judge_final.sum())} of {len(trap)}")
    else:
        print("\n(run 04_score_final_answer.py first to see judge-vs-code agreement here)")


if __name__ == "__main__":
    main()
