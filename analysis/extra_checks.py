"""Extra checks for the PhysFrame judge study (numpy + pandas only).

Input: one CSV with one row per response. Rename your columns in COLS below.
  model            evaluated model name
  template         template id (25 of them)
  problem          problem id (300 of them; a problem belongs to one template)
  framing          one of FRAMINGS below
  code_correct     1 if the strict code scorer marks the final answer right
  lenient_correct  1 if the correct value appears anywhere in the response
  truncated        1 if the response hit the token limit
  rel_err          |extracted answer - key| / |key|   (NaN if nothing was extracted)
  judge_label      CORRECT / RIGHT_ANSWER_WRONG_REASONING / PARTIAL / WRONG

Run:  python3 extra_checks.py responses.csv          (real data)
      python3 extra_checks.py --synthetic            (fake data, only to test the code)
"""
import sys
import numpy as np
import pandas as pd

COLS = dict(model="model", template="template", problem="problem", framing="framing",
            code_correct="code_correct", lenient_correct="lenient_correct",
            truncated="truncated", rel_err="rel_err", judge_label="judge_label")
FRAMINGS = ["bench", "placebo", "formal_motive", "casual", "casual_motive"]
FORMAL, CASUAL = ["bench", "formal_motive"], ["casual", "casual_motive"]
PASS_LABELS = {"CORRECT", "RIGHT_ANSWER_WRONG_REASONING"}   # judge says "final answer right"
B = 2000
rng = np.random.default_rng(0)


def load(path):
    df = pd.read_csv(path).rename(columns={v: k for k, v in COLS.items()})
    df["judge_pass"] = df["judge_label"].isin(PASS_LABELS).astype(int)
    return df


# ---------- basic rates ----------
def rates(truth, judge):
    """All the per-class numbers for one set of responses (truth and judge are 0/1 arrays)."""
    truth, judge = np.asarray(truth), np.asarray(judge)
    n = len(truth)
    tp = int(((truth == 1) & (judge == 1)).sum()); fn = int(((truth == 1) & (judge == 0)).sum())
    fp = int(((truth == 0) & (judge == 1)).sum()); tn = int(((truth == 0) & (judge == 0)).sum())
    p, q, po = truth.mean(), judge.mean(), (tp + tn) / n
    pe = p * q + (1 - p) * (1 - q)
    return dict(n=n, true=100 * p, judge=100 * q, bias=100 * (q - p), agree=100 * po,
                sens=tp / (tp + fn) if tp + fn else np.nan,     # share of right answers the judge passes
                spec=tn / (tn + fp) if tn + fp else np.nan,     # share of wrong answers the judge fails
                kappa=(po - pe) / (1 - pe) if pe < 1 else np.nan, FP=fp, FN=fn)


def per_model(df, truth="code_correct"):
    return pd.DataFrame({m: rates(g[truth], g["judge_pass"]) for m, g in df.groupby("model")}).T


# ---------- cluster bootstrap ----------
def boot_ci(df, stat, unit, B=B):
    """95% interval for stat(df), resampling whole clusters (unit = 'problem' or 'template')."""
    groups = [g for _, g in df.groupby(unit)]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def bias(g):
    return 100 * (g["judge_pass"].mean() - g["code_correct"].mean())


def agree(g):
    return 100 * (g["judge_pass"] == g["code_correct"]).mean()


def casual_minus_formal(g, f=agree):
    return f(g[g["framing"].isin(CASUAL)]) - f(g[g["framing"].isin(FORMAL)])


# ---------- the checks ----------
def check_cluster_level(df, models):
    print("\n[1] Same statistic, two bootstrap units (problem = 300 clusters, template = 25 clusters)")
    for m in models:
        g = df[df["model"] == m]
        print(f"  bias {m:16s} {bias(g):+6.2f}  problem CI {boot_ci(g, bias, 'problem')}  template CI {boot_ci(g, bias, 'template')}")
    pooled = df[df["model"].isin(models)]
    print(f"  casual-formal agreement, pooled {casual_minus_formal(pooled):+.2f}"
          f"  problem CI {boot_ci(pooled, casual_minus_formal, 'problem')}"
          f"  template CI {boot_ci(pooled, casual_minus_formal, 'template')}")


def check_truth_variants(df):
    print("\n[2] Judge bias (points) under different definitions of the truth")
    out = pd.DataFrame({
        "strict": per_model(df)["bias"],
        "lenient": per_model(df, "lenient_correct")["bias"],
        "finished only": per_model(df[df["truncated"] == 0])["bias"],
    })
    for tol in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05):      # relative-tolerance sweep
        d = df.assign(tc=(df["rel_err"] <= tol).astype(int))
        out[f"tol {100 * tol:g}%"] = per_model(d, "tc")["bias"]
    print(out.round(1).to_string())
    fp = df[(df["judge_pass"] == 1) & (df["code_correct"] == 0)]
    print("  judge-pass / code-fail responses:", len(fp), "| of these, correct value appears somewhere (lenient):",
          int(fp["lenient_correct"].sum()), "| truncated:", int(fp["truncated"].sum()))


def check_acceptance_curve(df):
    print("\n[3] Judge pass rate by distance from the key (shows the tolerance the judge really applies)")
    bins = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, np.inf]
    d = df.dropna(subset=["rel_err"]).assign(bin=lambda x: pd.cut(x["rel_err"], bins, right=False))
    print(d.groupby("bin", observed=True)["judge_pass"].agg(["size", "mean"]).round(3).to_string())


def check_per_class(df):
    print("\n[4] Per-model error rates (sens = right answers passed, spec = wrong answers failed)")
    print(per_model(df).round(3).to_string())
    print("\n    judge label x code verdict")
    print(pd.crosstab(df["judge_label"], df["code_correct"]).to_string())


def check_framing(df, models):
    print("\n[5] Is the framing effect in the judge's error rates, or only in the mix of right/wrong answers?")
    d = df[df["model"].isin(models)]
    rows = {f: rates(g["code_correct"], g["judge_pass"]) for f, g in d.groupby("framing")}
    print(pd.DataFrame(rows).T.loc[FRAMINGS, ["true", "agree", "sens", "spec"]].round(3).to_string())
    print("    truncated share by framing:", d.groupby("framing")["truncated"].mean().round(3).to_dict())
    cell = lambda f: agree(d[d["framing"] == f])
    print(f"    register (casual - formal) {(cell('casual') + cell('casual_motive') - cell('bench') - cell('formal_motive')) / 2:+.2f}"
          f" | motive {(cell('formal_motive') + cell('casual_motive') - cell('bench') - cell('casual')) / 2:+.2f}"
          f" | placebo - bench {cell('placebo') - cell('bench'):+.2f}")


def check_transfer(df, models):
    print("\n[6] Calibrate the judge on model A (rows), then correct model B's judge score (columns).")
    print("    corrected = (judge rate + spec - 1) / (sens + spec - 1). Last rows: truth and raw judge.")
    pm = per_model(df)
    tab = pd.DataFrame({b: {a: 100 * (pm.loc[b, "judge"] / 100 + pm.loc[a, "spec"] - 1) /
                               (pm.loc[a, "sens"] + pm.loc[a, "spec"] - 1) for a in models} for b in models})
    tab.loc["truth"] = pm.loc[models, "true"]; tab.loc["raw judge"] = pm.loc[models, "judge"]
    print(tab.astype(float).round(1).to_string())


def check_validation_designs(df, n=100, draws=5000, bar=85.0):
    print(f"\n[7] What a validation study of {n} items would report, by where the items come from")
    strata = {"all responses": df, **{f"only {m}": g for m, g in df.groupby("model")},
              **{f"only {f}": g for f, g in df.groupby("framing")}}
    for name, g in strata.items():
        t, j = g["code_correct"].to_numpy(), g["judge_pass"].to_numpy()
        idx = rng.integers(0, len(g), (draws, n))
        ag = 100 * (t[idx] == j[idx]).mean(axis=1)
        print(f"  {name:22s} agreement median {np.median(ag):5.1f}  95% range [{np.percentile(ag, 2.5):5.1f}, {np.percentile(ag, 97.5):5.1f}]"
              f"  passes a {bar:.0f}% bar in {100 * (ag >= bar).mean():5.1f}% of studies  kappa on all = {rates(t, j)['kappa']:.2f}")


def check_rank_stability(df, unit="template"):
    print(f"\n[8] Does the judge keep the model order? (bootstrap over {unit}s)")
    groups = [g for _, g in df.groupby(unit)]
    same = 0
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        t = s.groupby("model")["code_correct"].mean(); j = s.groupby("model")["judge_pass"].mean()
        same += list(t.sort_values().index) == list(j.sort_values().index)
    print(f"  full order identical in {100 * same / B:.1f}% of resamples")


# ---------- fake data, only to test that the code runs ----------
def synthetic():
    spec = {  # model: (accuracy, cut share, judge sens, judge spec), loosely shaped like Tables 2 and 3
        "Qwen3-Omni-30B": (0.774, 0.435, 0.96, 0.36), "Qwen2.5-7B": (0.724, 0.077, 0.80, 0.71),
        "Qwen2.5-3B": (0.383, 0.124, 0.67, 0.90), "Gemma-2-9B": (0.215, 0.0, 0.46, 0.98),
        "Qwen2.5-1.5B": (0.112, 0.151, 0.52, 0.96), "Gemma-2-2B": (0.019, 0.019, 0.35, 0.98),
        "Qwen2.5-0.5B": (0.007, 0.15, 0.05, 1.0)}
    rows = []
    for t in range(25):
        hard = rng.normal(0, 1.5)                      # template difficulty (shared by its 12 problems)
        for k in range(12):
            for m, (acc, cut, se, sp) in spec.items():
                p = 1 / (1 + np.exp(-(np.log(acc / (1 - acc)) + hard)))
                for f in FRAMINGS:
                    c = rng.random() < p
                    err = abs(rng.normal(0, 0.004)) if c else abs(rng.normal(0.3, 0.2)) + 0.011
                    passed = rng.random() < (se if c else 1 - sp)
                    rows.append(dict(model=m, template=t, problem=t * 12 + k, framing=f, code_correct=int(c),
                                     lenient_correct=int(c or rng.random() < 0.05), truncated=int(rng.random() < cut),
                                     rel_err=err, judge_label="CORRECT" if passed else ("PARTIAL" if c else "WRONG")))
    df = pd.DataFrame(rows)
    df["judge_pass"] = df["judge_label"].isin(PASS_LABELS).astype(int)
    return df


if __name__ == "__main__":
    fake = len(sys.argv) < 2 or sys.argv[1] == "--synthetic"
    df = synthetic() if fake else load(sys.argv[1])
    if fake:
        print("*** SYNTHETIC DATA: the numbers below only show that the code runs ***")
    strongest = list(per_model(df).sort_values("true", ascending=False).index[:4])
    check_cluster_level(df, strongest)
    check_truth_variants(df)
    check_acceptance_curve(df)
    check_per_class(df)
    check_framing(df, strongest)
    check_transfer(df, strongest)
    check_validation_designs(df)
    check_rank_stability(df)
