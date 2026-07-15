
import pandas as pd
import numpy as np
from scipy import stats

DATA = "data/register_pairs_300.csv"
QWEN = "generations/qwen_generations.csv"
GEMMA = "generations/gemma_generations.csv"
HEDGE_AUDIT = "human_labels/hedge_audit_200.csv"

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
pairs = pd.read_csv(DATA)
gens = pd.concat([pd.read_csv(QWEN), pd.read_csv(GEMMA)], ignore_index=True)
pid2intent = dict(zip(pairs.pair_id, pairs.intent))
gens["intent"] = gens.pair_id.map(pid2intent)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it", "google/gemma-2-9b-it",
]

# ---------------------------------------------------------------------------
# 2. Refusal-based RVG with clustered (by-intent) bootstrap
# ---------------------------------------------------------------------------
def clustered_rvg(sub, n_boot=5000, seed=0):
    piv = sub.pivot_table(index=["intent", "pair_id"], columns="framing",
                          values="refused", aggfunc="first").dropna().reset_index()
    piv["diff"] = piv["eval_framing"] - piv["deploy_framing"]
    rvg = piv["diff"].mean()
    intents = piv["intent"].unique()
    grp = {it: piv[piv.intent == it]["diff"].values for it in intents}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(intents, len(intents), replace=True)
        boots.append(np.concatenate([grp[it] for it in chosen]).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    e = piv["eval_framing"].values.astype(float)
    d = piv["deploy_framing"].values.astype(float)
    # McNemar exact
    b = int(((e == 1) & (d == 0)).sum()); c = int(((e == 0) & (d == 1)).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    return rvg, lo, hi, p

print("=" * 70)
print("Refusal-based Register Validity Gap (clustered bootstrap over 25 intents)")
print("=" * 70)
print(f"{'model':26s} {'RVG':>6s} {'95% CI':>18s} {'McNemar p':>11s}")
for m in MODELS:
    rvg, lo, hi, p = clustered_rvg(gens[gens.model == m])
    print(f"{m.split('/')[-1]:26s} {rvg:6.3f}  [{lo:6.3f},{hi:6.3f}]  {p:11.2e}")

# corpus refusal gap
e_all = gens[gens.framing == "eval_framing"]["refused"].mean()
d_all = gens[gens.framing == "deploy_framing"]["refused"].mean()
print(f"\nCorpus refusal gap: {e_all - d_all:.3f}")

# ---------------------------------------------------------------------------
# 3. Human-corrected harm gap (from the 200-item hedge audit)
# ---------------------------------------------------------------------------
def norm(x):
    x = str(x).strip().upper().replace("?", "")
    if x.startswith("R"): return "REFUSAL"
    if x.startswith("COMPL") or x.startswith("HARM"): return "HARMFUL"
    if x.startswith("H"): return "HEDGE"
    return x

audit = pd.read_csv(HEDGE_AUDIT)
lbl_col = "human_grade" if "human_grade" in audit.columns else audit.columns[-1]
audit["hg"] = audit[lbl_col].map(norm)

# naive grader corpus counts (from harm-grader run; see paper Sec. 5)
N = 1800
harm_f, hedge_f = 16, 684     # formal framing
harm_c, hedge_c = 76, 1105    # casual framing

kf = int(((audit.framing == "eval_framing") & (audit.hg == "HARMFUL")).sum())
nf = int((audit.framing == "eval_framing").sum())
kc = int(((audit.framing == "deploy_framing") & (audit.hg == "HARMFUL")).sum())
nc = int((audit.framing == "deploy_framing").sum())
pf, pc = kf / nf, kc / nc

print("\n" + "=" * 70)
print("Harm-gap correction from 200-item hedge audit")
print("=" * 70)
print(f"grader-hedges actually harmful:  formal {kf}/{nf}={pf:.2f}   casual {kc}/{nc}={pc:.2f}")

naive_gap = (harm_c - harm_f) / N
G = (harm_c + pc * hedge_c) / N - (harm_f + pf * hedge_f) / N
rng = np.random.default_rng(42)
S = 200_000
pf_s = rng.beta(kf + 1, nf - kf + 1, S)
pc_s = rng.beta(kc + 1, nc - kc + 1, S)
g_s = (harm_c + pc_s * hedge_c) / N - (harm_f + pf_s * hedge_f) / N
lo, hi = np.percentile(g_s, [2.5, 97.5])
print(f"\nNaive grader harm gap:      {naive_gap:.3f}")
print(f"Corrected harm gap:         {G:.3f}   95% CrI [{lo:.3f}, {hi:.3f}]")
print(f"corrected harm rate: formal {(harm_f + pf*hedge_f)/N:.3f}  casual {(harm_c + pc*hedge_c)/N:.3f}")
print(f"P(gap>0)={ (g_s>0).mean():.3f}   P(gap>naive)={ (g_s>naive_gap).mean():.3f}")
print("\nDone.")
