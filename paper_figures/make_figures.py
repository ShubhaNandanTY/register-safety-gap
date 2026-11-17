#!/usr/bin/env python3
"""Figures for the PhysFrame paper. Every value is read from the result files
except Figure 1, whose audit estimates are copied from the Study 1 audit table."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/claude"
OUT = f"{R}/v3/figs"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
SHORT = {"Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
         "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "google/gemma-2-2b-it": "Gemma-2-2B", "google/gemma-2-9b-it": "Gemma-2-9B",
         "Qwen3-Omni-30B-A3B-Instruct-4bit": "Qwen3-Omni-30B"}
CELLS = ["BENCH", "BENCH_PLACEBO", "FORMAL_MOTIVE", "CASUAL", "CASUAL_MOTIVE"]
CELL_LBL = ["Bench", "Placebo", "Formal+\nmotive", "Casual", "Casual+\nmotive"]


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def load():
    g = pd.read_csv(f"{R}/all7/generations_physframe_1024.csv")
    s = pd.read_csv(f"{R}/final_pkg/final_answer_scores_7models.csv")
    j = pd.read_csv(f"{R}/final_pkg/judge_phi4_final_7models.csv")[["response_id", "grader_label"]]
    d = g.merge(s, on="response_id").merge(j, on="response_id")
    d["m"] = d.model.map(SHORT)
    d = d[d.grader_label != "UNPARSED"].copy()
    d["jr"] = d.grader_label.isin(["CORRECT", "RIGHT_ANSWER_WRONG_REASONING"]).astype(int)
    return d


def fig1():
    """Study 1: one grader, three audits (values from the NeurIPS paper, Table 1)."""
    rows = [("Grader (unvalidated)", 0.033, None, None),
            ("Audit 1: 1 annotator, unblinded", 0.258, 0.190, 0.322),
            ("Audit 2, annotator A", -0.036, -0.123, 0.051),
            ("Audit 2, annotator B", 0.274, 0.193, 0.336),
            ("Audit 3: blinded, 3 annotators", 0.035, -0.053, 0.116)]
    fig, ax = plt.subplots(figsize=(3.3, 1.85))
    for i, (lab, e, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.4, solid_capstyle="round")
        ax.plot(e, y, "o", ms=4.5, color=BLUE if lo is not None else INK2, mec="white", mew=0.8)
        ax.text((hi if hi is not None else e) + 0.015, y, f"{e:+.3f}", ha="left", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlabel("Corrected harm gap, casual − formal (95% CrI)", fontsize=7)
    ax.set_xlim(-0.16, 0.44)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig1_study1_audits.pdf", bbox_inches="tight", pad_inches=0.02)


def fig3(d, t3):
    """True vs judge-measured accuracy per model."""
    order = t3.sort_values("true_acc").index.tolist()
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for y, m in enumerate(order):
        r = t3.loc[m]
        ax.plot([100 * r.true_acc, 100 * r.judge_acc], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.plot(100 * r.true_acc, y, "o", ms=5, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.plot(100 * r.judge_acc, y, "D", ms=4.5, color=ORANGE, mec="white", mew=0.8, zorder=3)
        b = 100 * r.bias
        if abs(b) >= 2:
            x = max(r.true_acc, r.judge_acc) * 100 + 3
            ax.text(x, y, f"{b:+.1f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Final-answer accuracy (%)")
    ax.plot([], [], "o", color=BLUE, label="Code reference")
    ax.plot([], [], "D", color=ORANGE, label="According to the judge")
    ax.legend(loc="lower right", fontsize=6.5, handletextpad=0.3)
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig3_judge_vs_truth.pdf", bbox_inches="tight", pad_inches=0.02)


def fig4(d):
    """(a) false passes by truncation; (b) judge pass rate by distance from the key."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [1.05, 1]})
    fp = d[(d.jr == 1) & (d.final_correct == 0)]
    order = fp.groupby("m").size().sort_values(ascending=False).index.tolist()
    cut = fp[fp.finish_reason == "length"].groupby("m").size().reindex(order).fillna(0)
    fin = fp[fp.finish_reason != "length"].groupby("m").size().reindex(order).fillna(0)
    y = np.arange(len(order))
    a.barh(y, cut, color=ORANGE, height=0.55, label="cut off by token limit")
    a.barh(y, fin, left=cut + np.where(cut > 0, 2, 0), color=BLUE, height=0.55, label="finished")
    for i, m in enumerate(order):
        a.text(cut[m] + fin[m] + 6, i, f"{int(cut[m] + fin[m])}", va="center", fontsize=6.5, color=INK2)
    a.set_yticks(y)
    a.set_yticklabels(order)
    a.invert_yaxis()
    a.set_xlabel(f"Responses the judge passed but the code marks wrong (n = {len(fp)})")
    a.set_title("(a) Passing wrong answers", loc="left", fontsize=7.5)
    a.legend(loc="lower right", fontsize=6.5)
    grid(a)
    e = d.dropna(subset=["rel_err"])
    edges = np.array([0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, np.inf])
    lab = ["<0.05", "<0.1", "<0.2", "<0.5", "<1", "<2", "<5", "<20", "≥20"]
    e = e.assign(bin=pd.cut(e.rel_err, edges, right=False, labels=lab))
    r = e.groupby("bin", observed=True).jr.agg(["mean", "size"])
    x = np.arange(len(r))
    b.axvspan(-0.5, 4.5, color="#f1f0ec", zorder=0)
    b.plot(x, 100 * r["mean"], color=BLUE, lw=1.4, marker="o", ms=4, mec="white", mew=0.6, zorder=3)
    for i, (v, n) in enumerate(zip(r["mean"], r["size"])):
        b.text(i, 100 * v + 5, f"{100*v:.0f}", ha="center", fontsize=6.2, color=INK2)
    b.text(2.0, 12, "inside the ±1% tolerance", ha="center", fontsize=6.5, color=INK2)
    b.set_xticks(x)
    b.set_xticklabels(lab, fontsize=6.0)
    b.set_ylim(0, 112)
    b.set_xlim(-0.5, len(r) - 0.5)
    b.set_xlabel("Distance of the extracted answer from the key (%)")
    b.set_ylabel("Judge passes (%)")
    b.set_title("(b) Judge pass rate by distance from the key", loc="left", fontsize=7.5)
    grid(b, "y")
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(f"{OUT}/fig4_mechanisms.pdf", bbox_inches="tight", pad_inches=0.02)
    return r


def fig5(d):
    """Judge agreement with truth by framing, four core models."""
    core = [("Qwen2.5-3B", BLUE, "o"), ("Qwen2.5-7B", ORANGE, "s"),
            ("Gemma-2-9B", AQUA, "^"), ("Qwen3-Omni-30B", YELLOW, "D")]
    d = d.assign(ag=(d.jr == d.final_correct).astype(float))
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    x = np.arange(len(CELLS))
    ends = []
    for m, c, mk in core:
        v = d[d.m == m].groupby("cell").ag.mean().reindex(CELLS) * 100
        ax.plot(x, v, color=c, lw=1.4, marker=mk, ms=4, mec="white", mew=0.6)
        ends.append([v.iloc[-1], m])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 1.3:
            ends[i][0] = ends[i - 1][0] + 1.3
    for yv, m in ends:
        ax.text(x[-1] + 0.12, yv, m, va="center", fontsize=6.3, color=INK)
    ax.axvspan(2.5, 4.4, color="#f1f0ec", zorder=0)
    ax.text(3.5, 73.6, "casual", ha="center", fontsize=6.5, color=INK2)
    ax.text(0.0, 73.6, "formal", ha="center", fontsize=6.5, color=INK2)
    ax.text(1.0, 73.6, "control", ha="center", fontsize=6.5, color=INK2)
    ax.text(2.0, 73.6, "formal", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_LBL, fontsize=6.3)
    ax.set_xlim(-0.3, 5.9)
    ax.set_ylim(72.5, 92)
    ax.set_ylabel("Judge agrees with code (%)")
    grid(ax, "y")
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig5_agreement_by_framing.pdf", bbox_inches="tight", pad_inches=0.02)


def fig6():
    """Framed vs framing-hidden judge labels: transition counts."""
    N = pd.read_csv(f"{R}/final_pkg/judge_phi4_neutral_gemma9b_qwen30.csv")[["response_id", "grader_label"]]
    F = pd.read_csv(f"{R}/final_pkg/judge_phi4_final_7models.csv")[["response_id", "grader_label"]]
    m = N.merge(F, on="response_id", suffixes=("_h", "_f"))
    labs = ["CORRECT", "RIGHT_ANSWER_WRONG_REASONING", "PARTIAL", "WRONG"]
    short = ["Correct", "Right ans.,\nwrong reas.", "Partial", "Wrong"]
    t = pd.crosstab(m.grader_label_f, m.grader_label_h).reindex(index=labs, columns=labs).fillna(0)
    off = t.values.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9cc0ec", BLUE, "#174a8a"])
    ax.imshow(np.where(np.isnan(off), 0, off), cmap=ramp, vmin=0, vmax=np.nanmax(off))
    for i in range(4):
        for k in range(4):
            v = int(t.values[i, k])
            diag = i == k
            ax.text(k, i, f"{v}", ha="center", va="center", fontsize=7,
                    color=INK2 if diag else (INK if off[i, k] < 0.55 * np.nanmax(off) else "white"),
                    fontweight="normal" if diag else "bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(short, fontsize=6.3)
    ax.set_yticks(range(4))
    ax.set_yticklabels(short, fontsize=6.3)
    ax.set_xlabel("Label with framing hidden")
    ax.set_ylabel("Label with framed prompt")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig6_framed_vs_hidden.pdf", bbox_inches="tight", pad_inches=0.02)
    return int(off[~np.isnan(off)].sum()), len(m)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    d = load()
    t3 = pd.read_csv(f"{R}/results/T3_judge_vs_truth.csv").set_index("model")
    print(fig4(d))
    fig1(); fig3(d, t3); fig5(d)
    print("changed labels (framed vs hidden):", fig6())
    print("done")
