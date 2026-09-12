#!/usr/bin/env python3
"""
analyze_all_audits.py -- the tables for the revised paper.

Produces:
  Table A  three audits of the same grader, side by side
  Table B  annotator-level variance within the 442-item audit
  Table C  two graders vs blinded human gold
  Table D  decomposition of the 0.258 -> 0.034 shift

Inputs (searched recursively from --root):
  sheet_A1.csv sheet_A2_labeled.csv sheet_A3_labeled.csv   Audit 3 (blinded)
  annotatorA_filled_A.csv annotatorB_labeled.csv           Audit 2 (442 items)
  hedge_audit_200.csv                                      Audit 1
  judge_v2_rescored.csv                                    primary grader
  second_grader_labels.csv                                 Llama Guard
  register_pairs_300.csv
"""
import argparse, glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["REFUSAL", "HEDGE", "HARMFUL"]
NORM = {"HEDGE": "HEDGE", "Hedge": "HEDGE", "REFUSE": "REFUSAL",
        "Refusal": "REFUSAL", "REFUSAL": "REFUSAL", "COMPLY": "HARMFUL",
        "Comply": "HARMFUL", "HARMFUL": "HARMFUL"}
RNG = np.random.default_rng(20260819)


def find(name, root):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    hits = [h for h in hits if Path(h).is_file()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def prefix_grade(df, judge, text_col, n=300):
    """Resolve grader label by unique reply-prefix match within framing."""
    out = []
    for _, r in df.iterrows():
        c = judge[(judge.framing == r.framing)
                  & (judge.reply.str.startswith(str(r[text_col])[:n], na=False))]
        out.append(c.grade.iloc[0] if c.grade.nunique() == 1 else None)
    return out


def corrected(sample, label_col, comp):
    """Prevalence-corrected P(harm) per framing from an audited confusion matrix."""
    res = {}
    for f, counts in comp.items():
        tot = sum(counts.values())
        p = 0.0
        for gr, cnt in counts.items():
            s = sample[(sample.framing == f) & (sample.grade == gr)]
            if len(s):
                p += (cnt / tot) * (s[label_col] == "HARMFUL").mean()
        res[f] = p
    return res


def dirichlet_gap(sample, label_col, comp, draws=100000):
    post = {}
    for f, counts in comp.items():
        tot = sum(counts.values())
        acc = np.zeros(draws)
        for gr, cnt in counts.items():
            s = sample[(sample.framing == f) & (sample.grade == gr)]
            alpha = np.array([(s[label_col] == l).sum() for l in LABELS], float) + 1.0
            acc += (cnt / tot) * RNG.dirichlet(alpha, draws)[:, LABELS.index("HARMFUL")]
        post[f] = acc
    return post["deploy_framing"] - post["eval_framing"]


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(np.mean([x == c for x in a]) * np.mean([y == c for y in b]) for c in cats)
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = a.root

    judge = pd.read_csv(find("judge_v2_rescored.csv", R))
    pairs = pd.read_csv(find("register_pairs_300.csv", R))
    comp = {f: {g: int(((judge.framing == f) & (judge.grade == g)).sum())
                for g in LABELS} for f in ["eval_framing", "deploy_framing"]}
    print("corpus composition (grader):")
    for f, c in comp.items():
        print(f"  {f:15s} {c}")

    # ---------------- Audit 1 -------------------------------------------
    a1 = pd.read_csv(find("hedge_audit_200.csv", R))
    a1["lab"] = a1.human_grade.map(NORM)
    a1_rates = {f: (a1[a1.framing == f].lab == "HARMFUL").mean()
                for f in ["eval_framing", "deploy_framing"]}
    # Audit 1's own estimator: hedge cell only, refusals 0% harmful, harmful 100%
    a1_corr = {}
    for f in comp:
        tot = sum(comp[f].values())
        a1_corr[f] = (comp[f]["HEDGE"] / tot) * a1_rates[f] + comp[f]["HARMFUL"] / tot

    # ---------------- Audit 2 (442, two annotators) ----------------------
    A = pd.read_csv(find("annotatorA_filled_A.csv", R))
    B = pd.read_csv(find("annotatorB_labeled.csv", R))
    assert (A.item_id.values == B.item_id.values).all()
    A["A"] = A.label.map(NORM)
    A["B"] = B.label.map(NORM)
    A["grade"] = prefix_grade(A, judge, "response")
    a442 = A.dropna(subset=["grade"])

    # ---------------- Audit 3 (239, blinded) -----------------------------
    sheets = pd.concat([pd.read_csv(p) for p in
                        sorted(glob.glob(f"{R}/**/sheet_A*.csv", recursive=True))])
    wide = sheets.pivot(index="item_id", columns="annotator", values="human_label")
    gold = wide.apply(lambda r: Counter(r.dropna()).most_common(1)[0][0], axis=1)
    a3 = sheets.drop_duplicates("item_id").set_index("item_id")
    a3["gold"] = gold
    a3["grade"] = prefix_grade(a3.assign(framing=None).reset_index(), judge,
                               "model_response") if False else None
    # framing/grade for audit 3 via reply+intent (exact text match)
    jj = judge.merge(pairs[["pair_id", "intent"]], on="pair_id", how="left")
    fr, gr = [], []
    for _, r in a3.reset_index().iterrows():
        c = jj[(jj.reply == r.model_response) & (jj.intent == r.request_description)]
        fr.append(c.framing.iloc[0] if c.framing.nunique() == 1 else None)
        gr.append(c.grade.iloc[0] if c.grade.nunique() == 1 else None)
    a3 = a3.reset_index()
    a3["framing"], a3["grade"] = fr, gr
    a3v = a3.dropna(subset=["framing", "grade"])

    # ---------------- Table A -------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE A -- three audits of the same grader, same corpus")
    print("=" * 74)
    print(f"{'audit':38s} {'window':>9s} {'blind':>6s} {'ann':>4s} {'gap':>8s}")
    rows = [
        ("1  hedge-only, 1 annotator", "500 ch", "no", 1,
         a1_corr["deploy_framing"] - a1_corr["eval_framing"]),
    ]
    for col in ["A", "B"]:
        c = corrected(a442, col, comp)
        rows.append((f"2  stratified, annotator {col}", "700 ch", "no", 1,
                     c["deploy_framing"] - c["eval_framing"]))
    c3 = corrected(a3v, "gold", comp)
    rows.append(("3  stratified, 3 annotators, blinded", "full", "yes", 3,
                 c3["deploy_framing"] - c3["eval_framing"]))
    for n, w, b, k, g in rows:
        print(f"{n:38s} {w:>9s} {b:>6s} {k:>4d} {g:>8.3f}")
    gap3 = dirichlet_gap(a3v, "gold", comp)
    print(f"\n  audit 3 95% CrI [{np.percentile(gap3,2.5):.3f}, "
          f"{np.percentile(gap3,97.5):.3f}]   P(gap>0)={np.mean(gap3>0):.2f}")
    print(f"  unvalidated grader gap: "
          f"{(judge[judge.framing=='deploy_framing'].grade=='HARMFUL').mean() - (judge[judge.framing=='eval_framing'].grade=='HARMFUL').mean():.3f}")

    # ---------------- Table B -------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE B -- annotator variance WITHIN audit 2 (same items, same window)")
    print("=" * 74)
    print(pd.crosstab(A.A, A.B, rownames=["annotator A"], colnames=["annotator B"]))
    print(f"\nraw agreement {np.mean(A.A==A.B):.3f}   Cohen kappa {kappa(list(A.A),list(A.B)):.3f}")
    for col in ["A", "B"]:
        c = corrected(a442, col, comp)
        print(f"  annotator {col}: formal {c['eval_framing']:.3f}  "
              f"casual {c['deploy_framing']:.3f}  gap "
              f"{c['deploy_framing']-c['eval_framing']:+.3f}")

    # ---------------- Table C -------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE C -- two graders vs blinded human gold")
    print("=" * 74)
    lg = pd.read_csv(find("second_grader_labels.csv", R))
    r = {f: (lg[lg.framing == f].guard_label == "UNSAFE").mean()
         for f in ["eval_framing", "deploy_framing"]}
    print(f"Llama Guard UNSAFE rate: eval {r['eval_framing']:.3f}  "
          f"deploy {r['deploy_framing']:.3f}  gap "
          f"{r['deploy_framing']-r['eval_framing']:+.3f}")
    lgi = lg.merge(judge[["model", "pair_id", "framing", "reply", "grade"]],
                   on=["model", "pair_id", "framing"]).merge(
                       pairs[["pair_id", "intent"]], on="pair_id")
    g = []
    for _, r2 in a3.iterrows():
        c = lgi[(lgi.reply == r2.model_response) & (lgi.intent == r2.request_description)]
        g.append(c.guard_label.iloc[0] if c.guard_label.nunique() == 1 else None)
    a3["guard"] = g
    d = a3.dropna(subset=["guard"])
    print("\nLlama Guard vs blinded gold:")
    print(pd.crosstab(d.gold, d.guard))
    tp = ((d.guard == "UNSAFE") & (d.gold == "HARMFUL")).sum()
    fp = ((d.guard == "UNSAFE") & (d.gold != "HARMFUL")).sum()
    fn = ((d.guard == "SAFE") & (d.gold == "HARMFUL")).sum()
    both = ((d.guard == "UNSAFE") & (d.gold.isin(["HARMFUL", "HEDGE"]))).sum()
    print(f"\nUNSAFE == HARMFUL          : precision {tp}/{tp+fp} = {tp/(tp+fp):.3f}"
          f"   recall {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
    print(f"UNSAFE == HARMFUL or HEDGE : precision {both}/{tp+fp} = {both/(tp+fp):.3f}")
    print("  -> Llama Guard's UNSAFE class tracks non-refusal, not actionability.")

    # ---------------- Table D -------------------------------------------
    print("\n" + "=" * 74)
    print("TABLE D -- decomposition of the audit-1 -> audit-3 shift")
    print("=" * 74)
    h3 = {f: (a3v[(a3v.framing == f) & (a3v.grade == "HEDGE")].gold == "HARMFUL").mean()
          for f in comp}
    def est(hedge, full):
        o = {}
        for f in comp:
            tot = sum(comp[f].values())
            if full:
                s_h = a3v[(a3v.framing == f) & (a3v.grade == "HARMFUL")]
                s_r = a3v[(a3v.framing == f) & (a3v.grade == "REFUSAL")]
                hr = (s_h.gold == "HARMFUL").mean() if len(s_h) else 0
                rr = (s_r.gold == "HARMFUL").mean() if len(s_r) else 0
            else:
                hr, rr = 1.0, 0.0
            o[f] = (comp[f]["REFUSAL"]/tot)*rr + (comp[f]["HEDGE"]/tot)*hedge[f] \
                 + (comp[f]["HARMFUL"]/tot)*hr
        return o["deploy_framing"] - o["eval_framing"]
    print(f"  audit 1 as published                      {est(a1_rates, False):.3f}")
    print(f"  audit 1 hedge rates + audit 3 estimator   {est(a1_rates, True):.3f}")
    print(f"  audit 3 hedge rates + audit 1 estimator   {est(h3, False):.3f}")
    print(f"  audit 3 as published                      {est(h3, True):.3f}")


if __name__ == "__main__":
    main()
