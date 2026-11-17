#!/usr/bin/env python3
"""
07_judge_analysis.py -- every PhysFrame number for the paper, from the files.

    python 07_judge_analysis.py --gen generations_physframe_1024.csv \
        --scores final_answer_scores.csv --judge judge_phi4_final_7models.csv \
        [--neutral judge_..._view-neutral_...csv] [--speed speed_log_all_models.csv] \
        [--out results]

Writes CSV tables and RESULTS.md to --out. Confidence intervals are 95%
bootstrap intervals that resample PROBLEMS (all five framings of a problem
stay together), because responses to the same problem are not independent.

Tables
  T1  final-answer accuracy by model x framing (code vs answer key)
  T2  framing effects on accuracy: each framing minus BENCH, paired by problem;
      BENCH_PLACEBO - BENCH is the noise floor
  T3  judge vs truth per model: judge-implied accuracy, bias, agreement,
      passes-wrong / fails-right, RIGHT_ANSWER_WRONG_REASONING, UNPARSED
  T4  judge agreement with truth by framing, and casual-vs-formal difference
  T5  trap answers the judge accepted as right
  T6  framed vs framing-hidden judge view (if --neutral given)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ["BENCH", "BENCH_PLACEBO", "FORMAL_MOTIVE", "CASUAL", "CASUAL_MOTIVE"]
RIGHT = ["CORRECT", "RIGHT_ANSWER_WRONG_REASONING"]
SHORT = {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "google/gemma-2-2b-it": "Gemma-2-2B", "google/gemma-2-9b-it": "Gemma-2-9B",
         "Qwen3-Omni-30B-A3B-Instruct-4bit": "Qwen3-Omni-30B"}
ORDER = list(SHORT.values())
B = 2000


def boot(df, stat, seed=0):
    """Point estimate and 95% CI of stat(df), resampling problem_id clusters."""
    rng = np.random.default_rng(seed)
    groups = {p: d for p, d in df.groupby("problem_id")}
    keys = np.array(list(groups))
    est = stat(df)
    vals = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        vals.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def fmt(e, lo, hi, pct=True):
    f = (lambda x: f"{100 * x:+.1f}") if pct else (lambda x: f"{x:+.3f}")
    return f"{f(e)} [{f(lo)}, {f(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--neutral", default=None)
    ap.add_argument("--speed", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(a.gen)
    s = pd.read_csv(a.scores)
    J = pd.read_csv(a.judge)[["response_id", "grader_label"]]
    d = g.merge(s, on="response_id").merge(J, on="response_id", how="left")
    d["m"] = d.model.map(SHORT).fillna(d.model)
    d["judge_right"] = d.grader_label.isin(RIGHT).astype(float)
    d.loc[d.grader_label.isna() | (d.grader_label == "UNPARSED"), "judge_right"] = np.nan
    d["formal"] = d.cell.isin(["BENCH", "BENCH_PLACEBO", "FORMAL_MOTIVE"])
    md = []

    # ---- T1 accuracy
    t1 = d.pivot_table(index="m", columns="cell", values="final_correct").reindex(ORDER)[CELLS]
    t1["ALL"] = d.groupby("m").final_correct.mean()
    t1["within_5pct"] = d.groupby("m").rel_err.apply(lambda x: (x <= 0.05).mean())
    t1["truncated"] = d.groupby("m").finish_reason.apply(lambda x: (x == "length").mean())
    t1.round(4).to_csv(f"{a.out}/T1_accuracy.csv")
    md += ["## T1 Final-answer accuracy (code vs answer key, ±1% unless stated)", "",
           (100 * t1).round(1).to_markdown(), ""]

    # ---- T2 framing effects on accuracy (paired by problem)
    rows = []
    for m, dm in d.groupby("m"):
        w = dm.pivot_table(index="problem_id", columns="cell", values="final_correct").reset_index()
        for c in ["BENCH_PLACEBO", "FORMAL_MOTIVE", "CASUAL", "CASUAL_MOTIVE"]:
            e, lo, hi = boot(w, lambda x, c=c: (x[c] - x["BENCH"]).mean())
            rows.append(dict(model=m, contrast=f"{c} - BENCH", diff=e, lo=lo, hi=hi))
        e, lo, hi = boot(w, lambda x: ((x.CASUAL + x.CASUAL_MOTIVE) / 2 - (x.BENCH + x.FORMAL_MOTIVE) / 2).mean())
        rows.append(dict(model=m, contrast="casual - formal (avg)", diff=e, lo=lo, hi=hi))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{a.out}/T2_framing_effects_accuracy.csv", index=False)
    piv = t2.assign(v=[fmt(r.diff, r.lo, r.hi) for r in t2.itertuples()]).pivot(index="model", columns="contrast", values="v").reindex(ORDER)
    md += ["## T2 Framing effects on accuracy (points, paired by problem, 95% cluster-bootstrap CI)", "",
           "BENCH_PLACEBO − BENCH is the noise floor: a different formal wording with no motive.", "",
           piv.to_markdown(), ""]

    # ---- T3 judge vs truth
    rows = []
    for m, dm in d.groupby("m"):
        v = dm.dropna(subset=["judge_right"])
        e, lo, hi = boot(v, lambda x: x.judge_right.mean() - x.final_correct.mean())
        rows.append(dict(
            model=m, n=len(dm), unparsed=int(dm.judge_right.isna().sum()),
            true_acc=v.final_correct.mean(), judge_acc=v.judge_right.mean(),
            bias=e, bias_lo=lo, bias_hi=hi,
            agreement=(v.judge_right == v.final_correct).mean(),
            passes_wrong=int(((v.judge_right == 1) & (v.final_correct == 0)).sum()),
            fails_right=int(((v.judge_right == 0) & (v.final_correct == 1)).sum()),
            right_answer_wrong_reasoning=int((dm.grader_label == "RIGHT_ANSWER_WRONG_REASONING").sum()),
            passes_wrong_truncated=int(((v.judge_right == 1) & (v.final_correct == 0) & (v.finish_reason == "length")).sum())))
    t3 = pd.DataFrame(rows).set_index("model").reindex(ORDER)
    t3.round(4).to_csv(f"{a.out}/T3_judge_vs_truth.csv")
    show = t3.copy()
    for c in ["true_acc", "judge_acc", "agreement"]:
        show[c] = (100 * show[c]).round(1)
    show["bias (judge−true)"] = [fmt(r.bias, r.bias_lo, r.bias_hi) for r in t3.itertuples()]
    show = show.drop(columns=["bias", "bias_lo", "bias_hi"])
    md += ["## T3 Judge (phi-4) vs truth", "",
           "judge_acc = share the judge labels CORRECT or RIGHT_ANSWER_WRONG_REASONING (UNPARSED excluded).", "",
           show.to_markdown(), ""]

    # ---- T4 agreement by framing + casual vs formal
    v = d.dropna(subset=["judge_right"]).assign(agree=lambda x: (x.judge_right == x.final_correct).astype(float))
    t4 = v.pivot_table(index="m", columns="cell", values="agree").reindex(ORDER)[CELLS]
    rows = []
    for m, dm in v.groupby("m"):
        w = dm.pivot_table(index="problem_id", columns="cell", values="agree").reset_index()
        e, lo, hi = boot(w, lambda x: ((x.CASUAL + x.CASUAL_MOTIVE) / 2 - (x.BENCH + x.FORMAL_MOTIVE) / 2).mean())
        e2, lo2, hi2 = boot(w, lambda x: (x.BENCH_PLACEBO - x.BENCH).mean())
        rows.append(dict(model=m, casual_minus_formal=fmt(e, lo, hi), placebo_minus_bench=fmt(e2, lo2, hi2)))
    core = v[v.m.isin(["Qwen2.5-3B", "Qwen2.5-7B", "Qwen3-Omni-30B", "Gemma-2-9B"])]
    w = core.pivot_table(index=["problem_id", "m"], columns="cell", values="agree").reset_index()
    e, lo, hi = boot(w, lambda x: ((x.CASUAL + x.CASUAL_MOTIVE) / 2 - (x.BENCH + x.FORMAL_MOTIVE) / 2).mean())
    e2, lo2, hi2 = boot(w, lambda x: (x.BENCH_PLACEBO - x.BENCH).mean())
    rows.append(dict(model="4 core models pooled", casual_minus_formal=fmt(e, lo, hi), placebo_minus_bench=fmt(e2, lo2, hi2)))
    t4b = pd.DataFrame(rows).set_index("model")
    t4.round(4).to_csv(f"{a.out}/T4_agreement_by_framing.csv")
    t4b.to_csv(f"{a.out}/T4b_agreement_casual_vs_formal.csv")
    md += ["## T4 Judge agreement with truth by framing (%)", "", (100 * t4).round(1).to_markdown(), "",
           "Casual − formal agreement, and the placebo noise floor (points, 95% cluster-bootstrap CI):", "",
           t4b.to_markdown(), ""]

    # ---- T5 traps
    tr = v[v.trap_match.fillna("") != ""]
    t5 = tr.groupby("m").agg(trap_answers=("judge_right", "size"), judge_accepted=("judge_right", "sum")).reindex(ORDER).fillna(0)
    t5.loc["ALL"] = t5.sum()
    t5["accepted_pct"] = (100 * t5.judge_accepted / t5.trap_answers.replace(0, np.nan)).round(1)
    t5.to_csv(f"{a.out}/T5_trap_acceptance.csv")
    md += ["## T5 Trap answers (a named mistake) the judge accepted as right", "", t5.to_markdown(), ""]

    # ---- T6 neutral view
    if a.neutral and os.path.exists(a.neutral):
        N = pd.read_csv(a.neutral)[["response_id", "grader_label"]].rename(columns={"grader_label": "neutral_label"})
        x = d.merge(N, on="response_id")
        x["neu_right"] = x.neutral_label.isin(RIGHT).astype(float)
        x.loc[x.neutral_label == "UNPARSED", "neu_right"] = np.nan
        rows = []
        for m, dm in x.groupby("m"):
            vv = dm.dropna(subset=["judge_right", "neu_right"])
            rows.append(dict(model=m, n=len(dm), label_changed=int((dm.grader_label != dm.neutral_label).sum()),
                             label_changed_pct=round(100 * (dm.grader_label != dm.neutral_label).mean(), 1),
                             true_acc=round(100 * vv.final_correct.mean(), 1),
                             judge_acc_framed=round(100 * vv.judge_right.mean(), 1),
                             judge_acc_hidden=round(100 * vv.neu_right.mean(), 1),
                             agree_framed=round(100 * (vv.judge_right == vv.final_correct).mean(), 1),
                             agree_hidden=round(100 * (vv.neu_right == vv.final_correct).mean(), 1),
                             right_to_wrong=int(((vv.judge_right == 1) & (vv.neu_right == 0)).sum()),
                             wrong_to_right=int(((vv.judge_right == 0) & (vv.neu_right == 1)).sum())))
        t6 = pd.DataFrame(rows).set_index("model")
        t6.to_csv(f"{a.out}/T6_framed_vs_hidden_view.csv")
        md += ["## T6 Judge shown the framed prompt vs a neutral restatement (framing hidden)", "", t6.to_markdown(), ""]

    if a.speed and os.path.exists(a.speed):
        sp = pd.read_csv(a.speed)
        sp = sp[sp.worker.isin(["all"]) | (sp.backend == "api")].drop_duplicates("model", keep="last")
        sp["m"] = sp.model.map(SHORT).fillna(sp.model)
        cols = ["m", "backend", "load_s", "warmup_s", "gen_s", "new_tok_per_s", "batch_size", "peak_mem_gb"]
        md += ["## Generation speed", "", sp[[c for c in cols if c in sp]].round(2).to_markdown(index=False), ""]

    Path(f"{a.out}/RESULTS.md").write_text("# PhysFrame results (generated by 07_judge_analysis.py)\n\n" + "\n".join(md))
    print("\n".join(md))
    print(f"\nwrote tables and RESULTS.md to {a.out}/")


if __name__ == "__main__":
    main()
