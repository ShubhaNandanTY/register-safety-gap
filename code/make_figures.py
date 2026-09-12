#!/usr/bin/env python3
"""Compute every number in the paper and emit the figures."""
import glob, json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/claude/work/ARCHIVE/03_code/analysis/run"
OUT = Path("/home/claude/work/paper/fig"); OUT.mkdir(parents=True, exist_ok=True)
import os; os.chdir(R)
RNG = np.random.default_rng(20260819)
LABELS = ["REFUSAL", "HEDGE", "HARMFUL"]
NORM = {"HEDGE": "HEDGE", "Hedge": "HEDGE", "REFUSE": "REFUSAL", "Refusal": "REFUSAL",
        "REFUSAL": "REFUSAL", "COMPLY": "HARMFUL", "Comply": "HARMFUL", "HARMFUL": "HARMFUL"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": .6, "ytick.major.width": .6, "figure.dpi": 300,
})
C_EVAL, C_DEP, C_GREY = "#3E5C76", "#C1666B", "#8A8A8A"

judge = pd.read_csv(f"{R}/judge_v2_rescored.csv")
pairs = pd.read_csv(f"{R}/register_pairs_300.csv")
comp = {f: {g: int(((judge.framing == f) & (judge.grade == g)).sum()) for g in LABELS}
        for f in ["eval_framing", "deploy_framing"]}

# ---------------------------------------------------------------- audits
a1 = pd.read_csv(f"{R}/hedge_audit_200.csv"); a1["lab"] = a1.human_grade.map(NORM)
A = pd.read_csv(f"{R}/annotatorA_filled_A.csv"); Bn = pd.read_csv(f"{R}/annotatorB_labeled.csv")
A["A"] = A.label.map(NORM); A["B"] = Bn.label.map(NORM)
g = []
for _, r in A.iterrows():
    c = judge[(judge.framing == r.framing) & judge.reply.str.startswith(str(r.response)[:300], na=False)]
    g.append(c.grade.iloc[0] if c.grade.nunique() == 1 else None)
A["grade"] = g
a442 = A.dropna(subset=["grade"])

sheets = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{R}/sheet_A*.csv"))])
wide = sheets.pivot(index="item_id", columns="annotator", values="human_label")
gold = wide.apply(lambda r: Counter(r.dropna()).most_common(1)[0][0], axis=1)
a3 = sheets.drop_duplicates("item_id").set_index("item_id"); a3["gold"] = gold
jj = judge.merge(pairs[["pair_id", "intent"]], on="pair_id", how="left")
fr, gr = [], []
for _, r in a3.reset_index().iterrows():
    c = jj[(jj.reply == r.model_response) & (jj.intent == r.request_description)]
    fr.append(c.framing.iloc[0] if c.framing.nunique() == 1 else None)
    gr.append(c.grade.iloc[0] if c.grade.nunique() == 1 else None)
a3 = a3.reset_index(); a3["framing"], a3["grade"] = fr, gr
a3v = a3.dropna(subset=["framing", "grade"])

def post_full(sample, col, draws=200000):
    out = {}
    for f, counts in comp.items():
        tot = sum(counts.values()); acc = np.zeros(draws)
        for gl, cnt in counts.items():
            s = sample[(sample.framing == f) & (sample.grade == gl)]
            al = np.array([(s[col] == l).sum() for l in LABELS], float) + 1.0
            acc += (cnt / tot) * RNG.dirichlet(al, draws)[:, 2]
        out[f] = acc
    return out["deploy_framing"] - out["eval_framing"]

def post_hedge_only(rates_kn, draws=200000):
    """audit-1 estimator: hedge cell only; refusals 0%, harmful 100%."""
    out = {}
    for f, counts in comp.items():
        tot = sum(counts.values()); k, n = rates_kn[f]
        out[f] = (counts["HEDGE"]/tot)*RNG.beta(k+1, n-k+1, draws) + counts["HARMFUL"]/tot
    return out["deploy_framing"] - out["eval_framing"]

a1kn = {f: (int((a1[a1.framing == f].lab == "HARMFUL").sum()), int((a1.framing == f).sum()))
        for f in comp}
P = {"Audit 1\nunblinded, 1 annotator": post_hedge_only(a1kn),
     "Audit 2\nannotator A": post_full(a442, "A"),
     "Audit 2\nannotator B": post_full(a442, "B"),
     "Audit 3\nblinded, 3 annotators": post_full(a3v, "gold")}
grader_gap = ((judge[judge.framing == "deploy_framing"].grade == "HARMFUL").mean()
              - (judge[judge.framing == "eval_framing"].grade == "HARMFUL").mean())

def plugin_full(sample, col):
    o = {}
    for f, counts in comp.items():
        tot = sum(counts.values()); p = 0.0
        for gl, cnt in counts.items():
            s = sample[(sample.framing == f) & (sample.grade == gl)]
            if len(s):
                p += (cnt/tot) * (s[col] == "HARMFUL").mean()
        o[f] = p
    return o["deploy_framing"] - o["eval_framing"]

def plugin_hedge_only(kn):
    o = {}
    for f, counts in comp.items():
        tot = sum(counts.values()); k, n = kn[f]
        o[f] = (counts["HEDGE"]/tot)*(k/n) + counts["HARMFUL"]/tot
    return o["deploy_framing"] - o["eval_framing"]

# plug-in point estimates (headline; match the verified tables)
PT = {"Audit 1\nunblinded, 1 annotator": plugin_hedge_only(a1kn),
      "Audit 2\nannotator A": plugin_full(a442, "A"),
      "Audit 2\nannotator B": plugin_full(a442, "B"),
      "Audit 3\nblinded, 3 annotators": plugin_full(a3v, "gold")}

S = {"grader_gap": float(grader_gap)}
for k, v in P.items():
    S[k.replace("\n", " ")] = [float(PT[k]), float(np.percentile(v, 2.5)),
                               float(np.percentile(v, 97.5)), float(v.mean())]
S["P_audit3_gt0"] = float((P["Audit 3\nblinded, 3 annotators"] > 0).mean())

# ============================================== FIG 1  forest plot
fig, ax = plt.subplots(figsize=(5.5, 1.20))
names = list(P.keys())[::-1]
for i, n in enumerate(names):
    v = P[n]; m = PT[n]; lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
    col = C_DEP if "Audit 2" in n else C_EVAL
    ax.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(m, i, "o", color=col, ms=5.5, zorder=3, mec="white", mew=.7)
    ax.text(hi + .012, i, f"{m:.3f}", va="center", fontsize=7.5, color=col)
ax.axvline(grader_gap, color=C_GREY, ls="--", lw=.9, zorder=1)
ax.text(grader_gap, len(names) - .28, " unvalidated grader (0.033)",
        fontsize=6.5, color=C_GREY, va="bottom", ha="left")
ax.axvline(0, color="#CCCCCC", lw=.7, zorder=0)
SHORT_A = {"Audit 1\nunblinded, 1 annotator": "Audit 1, 1 annot.",
           "Audit 2\nannotator A": "Audit 2, annot. A",
           "Audit 2\nannotator B": "Audit 2, annot. B",
           "Audit 3\nblinded, 3 annot.": "Audit 3, blinded",
           "Audit 3\nblinded, 3 annotators": "Audit 3, blinded"}
ax.set_yticks(range(len(names)))
ax.set_yticklabels([SHORT_A.get(n, n.replace("\n", ", ")) for n in names], fontsize=7)
ax.tick_params(axis="y", pad=2)
ax.set_xlabel("corrected harm gap  (casual $-$ formal)")
ax.set_xlim(-.18, .48); ax.set_ylim(-.65, len(names) + .05)
fig.subplots_adjust(left=.185, right=.985, top=.91, bottom=.34)
fig.savefig(OUT/"fig1_audits.pdf"); plt.close(fig)

# ============================================== FIG 2  RVG + annotator scatter
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.22),
                         gridspec_kw={"width_ratios": [1.30, 1], "wspace": .50})
ax = axes[0]
rv = (judge.assign(ref=judge.refused).groupby(["model", "framing"]).ref.mean().unstack())
rv["gap"] = rv.eval_framing - rv.deploy_framing
short = {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen 0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 1.5B",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen 3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
         "google/gemma-2-2b-it": "Gemma 2B", "google/gemma-2-9b-it": "Gemma 9B"}
rv = rv.loc[list(short)]
y = np.arange(len(rv))
ax.barh(y, rv.gap, color=[C_EVAL]*4 + [C_DEP]*2, height=.52)
for i, v in enumerate(rv.gap):
    ax.text(v + .022, i, f"{v:.2f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([short[m] for m in rv.index], fontsize=7.5)
ax.invert_yaxis(); ax.set_xlim(0, .74)
ax.set_ylim(len(rv) - .35, -.65)
ax.tick_params(axis="y", pad=2)
ax.set_xlabel("refusal gap (RVG)", fontsize=7.5, labelpad=2); ax.set_title("(a) refusal level", fontsize=8, loc="left")

ax = axes[1]
cm = pd.crosstab(A.A, A.B).reindex(index=LABELS, columns=LABELS).fillna(0).astype(int)
im = ax.imshow(cm.values, cmap="BuPu", vmin=0, vmax=cm.values.max())
for i in range(3):
    for j in range(3):
        v = cm.values[i, j]
        ax.text(j, i, v, ha="center", va="center", fontsize=8,
                color="white" if v > cm.values.max()*.6 else "black")
ax.set_xticks(range(3))
ax.set_xticklabels(["ref.", "hedge", "harm"], fontsize=6.4)
ax.set_yticks(range(3))
ax.set_yticklabels(["ref.", "hedge", "harm"], fontsize=6.4)
ax.set_xlabel("annotator B", fontsize=7.5, labelpad=2)
ax.set_ylabel("annotator A", fontsize=7.5, labelpad=2)
ax.set_title(r"(b) audit 2, $\kappa=0.09$", fontsize=8, loc="left")
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(.4)
fig.subplots_adjust(left=.125, right=.985, top=.86, bottom=.26, wspace=.46)
fig.savefig(OUT/"fig2_rvg_annot.pdf"); plt.close(fig)

# ============================================== FIG 3  calibration + boundary
lg = pd.read_csv(f"{R}/second_grader_labels.csv")
lgi = lg.merge(judge[["model", "pair_id", "framing", "reply"]],
               on=["model", "pair_id", "framing"]).merge(pairs[["pair_id", "intent"]], on="pair_id")
gv = []
for _, r in a3.iterrows():
    c = lgi[(lgi.reply == r.model_response) & (lgi.intent == r.request_description)]
    gv.append(c.guard_label.iloc[0] if c.guard_label.nunique() == 1 else None)
a3["guard"] = gv
d = a3.dropna(subset=["guard"])

def gapfor(pred):
    o = {}
    for f, c in comp.items():
        tot = sum(c.values()); p = 0.0
        for gl, cnt in c.items():
            s = a3v[(a3v.framing == f) & (a3v.grade == gl)]
            if len(s): p += (cnt/tot) * pred.loc[s.index].mean()
        o[f] = p
    return o["deploy_framing"] - o["eval_framing"]

g_nonref = gapfor(a3v.gold != "REFUSAL")
g_harm = gapfor(a3v.gold == "HARMFUL")
g_kw = float(judge[judge.framing=="eval_framing"].refused.mean()
             - judge[judge.framing=="deploy_framing"].refused.mean())
g_lg = float((lg[lg.framing=="deploy_framing"].guard_label=="UNSAFE").mean()
             - (lg[lg.framing=="eval_framing"].guard_label=="UNSAFE").mean())
S["boundary"] = {"keyword_refusal": g_kw, "human_nonrefusal": float(g_nonref),
                 "llamaguard": g_lg, "primary_grader": float(grader_gap),
                 "human_actionable": float(g_harm)}

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.20),
                         gridspec_kw={"width_ratios": [1.30, .9, 1.15], "wspace": .55})
ax = axes[0]
prec, cols, xs_a = [], [], []
for k, (f, c) in enumerate([("eval_framing", C_EVAL), ("deploy_framing", C_DEP)]):
    for j, gl in enumerate(["refusal", "hedge", "harmful"]):
        sset = a3v[(a3v.framing == f) & (a3v.grade == gl.upper())]
        if len(sset):
            prec.append((sset.gold == gl.upper()).mean())
            cols.append(c); xs_a.append(j + (k - .5) * .46)
ax.bar(xs_a, prec, color=cols, width=.38)
for x, v in zip(xs_a, prec):
    ax.text(x, v + .05, f"{v:.2f}", ha="center", fontsize=5.8)
ax.set_xticks(range(3))
ax.set_xticklabels(["refusal", "hedge", "harmful"], fontsize=6.8)
ax.set_xlim(-.65, 2.65)
ax.set_ylim(0, 1.30); ax.set_ylabel("precision", fontsize=7)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_EVAL, label="formal"), Patch(color=C_DEP, label="casual")],
          frameon=False, fontsize=6, loc="upper right", handlelength=1.0,
          borderpad=.1, labelspacing=.25, handletextpad=.4)
ax.set_title("(a) primary grader", fontsize=7.5, loc="left")

ax = axes[1]
ct = pd.crosstab(d.gold, d.guard).reindex(index=LABELS, columns=["SAFE","UNSAFE"]).fillna(0).astype(int)
ax.bar(range(3), ct.SAFE, color="#BFC9D1", width=.62, label="safe")
ax.bar(range(3), ct.UNSAFE, bottom=ct.SAFE, color=C_DEP, width=.62, label="unsafe")
ax.set_ylim(0, (ct.SAFE + ct.UNSAFE).max() * 1.32)
ymax = float((ct.SAFE + ct.UNSAFE).max())
for i in range(3):
    sv, uv = int(ct.SAFE.iloc[i]), int(ct.UNSAFE.iloc[i])
    # a count only goes inside its segment when the segment is tall enough to hold it
    if sv and sv / ymax > .10:
        ax.text(i, sv/2, str(sv), ha="center", va="center", fontsize=6.2)
    elif sv:
        ax.text(i - .40, sv/2, str(sv), ha="right", va="center", fontsize=6.2)
    if uv and uv / ymax > .10:
        ax.text(i, sv + uv/2, str(uv), ha="center", va="center", fontsize=6.2, color="white")
    elif uv:
        ax.text(i + .40, sv + uv/2, str(uv), ha="left", va="center", fontsize=6.2)
ax.set_xticks(range(3)); ax.set_xticklabels(["ref.","hedge","harm."], fontsize=6.2)
ax.set_xlabel("human gold", fontsize=7, labelpad=2)
ax.set_ylabel("items", fontsize=7, labelpad=2)
ax.legend(frameon=False, fontsize=6, loc="upper right", borderpad=.1,
          labelspacing=.25, handlelength=1.0, handletextpad=.4)
ax.set_title("(b) Llama Guard", fontsize=7.5, loc="left")

ax = axes[2]
pts = [("kw.", g_kw, C_GREY), ("non-ref.", g_nonref, C_EVAL),
       ("Guard", g_lg, C_GREY), ("harm", g_harm, C_EVAL)]
xs = np.arange(len(pts))
ax.plot(xs, [p[1] for p in pts], "-", color="#AAAAAA", lw=.9, zorder=1)
for i, (n, v, c) in enumerate(pts):
    ax.plot(i, v, "o", color=c, ms=5, zorder=2, mec="white", mew=.6)
    ha = "left" if i == 0 else "center"
    dx = .12 if i == 0 else 0
    ax.text(i + dx, v + .022, f"{v:.3f}", ha=ha, fontsize=6)
ax.set_xticks(xs)
ax.set_xticklabels([p[0] for p in pts], fontsize=6.0)
ax.set_xlabel("decision boundary", fontsize=7, labelpad=1)
ax.set_ylim(-.04, .36); ax.set_xlim(-.45, 3.45)
ax.set_ylabel("measured gap", fontsize=7)
ax.set_title("(c) gap vs. boundary", fontsize=7.5, loc="left")
fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.26, wspace=.55)
fig.savefig(OUT/"fig3_calibration.pdf"); plt.close(fig)

# ============================================== FIG 4  truncation + bounds
gen = judge.copy()
gen["w"] = gen.reply.str.split().str.len()
gen["trunc"] = ~gen.reply.str.strip().str[-1].isin(list('.!?"'))
a3v2 = a3v.copy()
a3v2["trunc"] = ~a3v2.model_response.str.strip().str[-1].isin(list('.!?"'))
a3v2["unc"] = sheets.groupby("item_id").uncertain.max().reindex(a3v2.item_id).values

def gap_of(sample, cc=comp):
    o = {}
    for f, c in cc.items():
        tot = sum(c.values()); p = 0.0
        for gl, cnt in c.items():
            s = sample[(sample.framing == f) & (sample.grade == gl)]
            if len(s): p += (cnt/tot) * (s.gold == "HARMFUL").mean()
        o[f] = p
    return o["deploy_framing"] - o["eval_framing"]

comp_c = {f: {g: int(((judge.framing == f) & (judge.grade == g) &
              (judge.reply.str.strip().str[-1].isin(list('.!?"')))).sum())
              for g in LABELS} for f in comp}
lo = a3v2.copy(); lo.loc[lo.trunc, "gold"] = "HEDGE"
hi = a3v2.copy(); hi.loc[hi.trunc & (hi.gold != "REFUSAL"), "gold"] = "HARMFUL"
SENS = [
    ("as published",                     gap_of(a3v2)),
    ("complete responses only",          gap_of(a3v2[~a3v2.trunc], comp_c)),
    ("drop uncertain-flagged",           gap_of(a3v2[a3v2.unc == 0])),
    ("truncated $\\rightarrow$ non-harmful", gap_of(lo)),
    ("truncated $\\rightarrow$ harmful",     gap_of(hi)),
]
S["sensitivity"] = {k.replace("$\\rightarrow$", "->"): float(v) for k, v in SENS}

fig, ax = plt.subplots(figsize=(3.5, 1.9))
for f, tag, c in [("eval_framing", "formal", C_EVAL), ("deploy_framing", "casual", C_DEP)]:
    ax.hist(gen[gen.framing == f].w, bins=np.arange(0, 185, 6), alpha=.62, color=c, label=tag)
ax.axvline(151, color="black", ls="--", lw=.8)
ax.text(148, ax.get_ylim()[1]*.93, "200-token cap ", ha="right", fontsize=6.5)
ax.set_xlabel("response length (words)"); ax.set_ylabel("responses")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT/"fig4_truncation.pdf", bbox_inches="tight"); plt.close(fig)



# ============================================== FIG 5  budget curve
import numpy as _np
G = pd.read_csv("budget_gaps_merged.csv")
gen1024 = pd.read_csv("generations_1024_FULL.csv")
tr = {int(B): float((gen1024.n_new_tokens > B).mean()) for B in G.budget}

fig, ax = plt.subplots(figsize=(2.42, 1.34))
ax.plot(G.budget, G.refusal_gap, "o-", color=C_EVAL, ms=4.5, lw=1.4,
        label=r"$\Delta_{\mathrm{refusal}}$")
ax.plot(G.budget, G.grader_harm_gap, "s-", color=C_DEP, ms=4.5, lw=1.4,
        label=r"$\hat{\Delta}_{\mathrm{harm}}$")
ax.plot(G.budget, [tr[int(b)] for b in G.budget], "^--", color=C_GREY, ms=4,
        lw=1.1, label="truncation rate")
ax.axhline(0, color="#CCCCCC", lw=.7, zorder=0)
ax.set_ylim(-.03, .95)
ax.set_xlabel("budget (tokens)", fontsize=7.5, labelpad=2)
ax.set_xticks([200, 500, 800])
ax.tick_params(labelsize=7)
ax.set_ylabel("rate", fontsize=7.5, labelpad=2)
ax.legend(frameon=False, fontsize=6, loc="upper right", borderpad=.1,
          labelspacing=.28, handlelength=1.4, handletextpad=.45)
fig.subplots_adjust(left=.20, right=.975, top=.95, bottom=.24)
fig.savefig(OUT/"fig5_budget.pdf"); plt.close(fig)
S["budget_curve"] = {int(r.budget): float(r.grader_harm_gap) for _, r in G.iterrows()}

# ---------------------------------------------------------------- stats
S["rvg_corpus"] = float(judge[judge.framing == "eval_framing"].refused.mean()
                        - judge[judge.framing == "deploy_framing"].refused.mean())
S["kappa_AB"] = float((lambda a, b: (np.mean([x == y for x, y in zip(a, b)]) - sum(
    np.mean([x == c for x in a])*np.mean([y == c for y in b]) for c in set(a) | set(b))) / (
    1 - sum(np.mean([x == c for x in a])*np.mean([y == c for y in b]) for c in set(a) | set(b))))(
    list(A.A), list(A.B)))
nr = gen[gen.refused == 0]
S["trunc_nonrefusal"] = {"eval": float(nr[nr.framing=="eval_framing"].trunc.mean()),
                          "deploy": float(nr[nr.framing=="deploy_framing"].trunc.mean())}
from sklearn.metrics import f1_score, balanced_accuracy_score
_p = a3v.grade.values; _g = a3v.gold.values
S["grader_macroF1"] = float(f1_score(_g,_p,average="macro",labels=LABELS))
S["grader_balacc"] = float(balanced_accuracy_score(_g,_p))
_d = A[A.A!=A.B]
S["audit2_disagree_pairs"] = {"/".join(sorted(t)): int(n) for t,n in
    Counter(tuple(sorted([a,b])) for a,b in zip(_d.A,_d.B)).items()}
S["guard_gap"] = float((lg[lg.framing == "deploy_framing"].guard_label == "UNSAFE").mean()
                       - (lg[lg.framing == "eval_framing"].guard_label == "UNSAFE").mean())
tp = int(((d.guard == "UNSAFE") & (d.gold == "HARMFUL")).sum())
fp = int(((d.guard == "UNSAFE") & (d.gold != "HARMFUL")).sum())
S["guard_prec_harmful"] = tp/(tp+fp)
S["guard_prec_union"] = float(((d.guard == "UNSAFE") & (d.gold.isin(["HARMFUL", "HEDGE"]))).sum()/(tp+fp))
json.dump(S, open("/home/claude/work/paper/stats.json", "w"), indent=2)
print(json.dumps(S, indent=2))
