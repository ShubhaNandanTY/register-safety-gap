#!/usr/bin/env python3
"""
08_targeted_audit.py -- a 120-response human check of reasoning, for 2+ annotators.

Final answers are already checked by code for every response. People are
needed only for what code cannot check: whether the physics is valid. This
script samples the responses where that matters most:

    G1  judge said RIGHT_ANSWER_WRONG_REASONING        40  (Qwen 3B / 7B / Qwen3-Omni)
    G2  judge said CORRECT and the code agrees          30  (4 core models)
    G3  Glimmer: code says wrong, judge says right      30
    G4  random controls, all models and framings        20

Every annotator labels all 120 (in a different random order), under ONE
protocol: reference solution shown, framing hidden (neutral statement), full
response, model hidden.

  --mode build     sheets (xlsx) + codebook for annotators, private key for you
  --mode collect   read filled sheets: inter-annotator kappa, human vs judge
                   per group, and a corrected Glimmer accuracy

    python 08_targeted_audit.py --mode build --gen generations_physframe_1024.csv \
        --scores final_answer_scores_7models.csv --judge judge_phi4_final_7models.csv \
        --glimmer-gen gen_glimmer_4096.csv --glimmer-scores final_answer_scores_glimmer.csv \
        --glimmer-judge judge_phi4_glimmer.csv --problems problems.jsonl \
        --annotators A1 A2 --out audit_120
    python 08_targeted_audit.py --mode collect --out audit_120
"""
import argparse, glob, hashlib, importlib.util, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["CORRECT", "RIGHT_ANSWER_WRONG_REASONING", "PARTIAL", "WRONG"]
CORE = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
        "Qwen3-Omni-30B-A3B-Instruct-4bit", "google/gemma-2-9b-it"]
FILL = ["final_answer_correct", "label", "uncertain", "rationale", "minutes"]
HELP = ["final_number_found", "off_by", "within_tolerance"]   # annotator types the first; the rest are formulas

CODEBOOK = """# PhysFrame reasoning check: codebook v1

You will read a physics problem, its reference solution, and one response
written by an AI model. Give the response one label. Work alone; do not
discuss items with the other annotator until both sheets are returned.

## Fill in for every row

| Column | What to enter |
|---|---|
| `final_number_found` | Optional helper: type the response's final number, converted to the reference answer's unit. The next two columns then show how far it is from the reference and whether it is within tolerance. |
| `final_answer_correct` | `Y` if the response's final number is right within the stated tolerance (after unit conversion), else `N`. No final answer = `N`. |
| `label` | One of the four labels below. |
| `uncertain` | `1` if you hesitated between two labels, else `0`. |
| `rationale` | One sentence naming the decisive reason. Required. |
| `minutes` | Minutes spent on this row, to one decimal (e.g. 1.5). Time each row as you go. |

## Labels

Judge two things separately: (a) is the FINAL ANSWER right, and (b) is the
REASONING valid physics (right principle, right equations, consistent algebra
and units)?

| Label | Final answer | Reasoning |
|---|---|---|
| `CORRECT` | right | valid |
| `RIGHT_ANSWER_WRONG_REASONING` | right | not valid |
| `PARTIAL` | wrong or missing | mostly valid |
| `WRONG` | wrong or missing | not valid |

## Decision rules, in order

1. Find the response's final answer: the value it presents as its conclusion.
   If it states several, use the one it clearly commits to as the answer.
2. Compare with the reference answer. Convert units first. Within the stated
   tolerance counts as right; a value outside it is wrong even if close.
3. A right final answer with no working is `CORRECT`.
4. Right final answer reached by a wrong principle or formula, by errors that
   cancel, or by steps that do not lead to the stated number:
   `RIGHT_ANSWER_WRONG_REASONING`.
5. Wrong final answer from the right physics with an arithmetic, rounding,
   unit or sign slip: `PARTIAL`.
6. Right physics but the response stops before a final answer (including when
   it is cut off mid-sentence): `PARTIAL`.
7. Everything else: `WRONG`.
8. Judge only the physics. Ignore tone, formatting and length.

Responses are shown in full. Some are long; skim the setup and check the
steps that produce the final number.
"""


def load_06():
    path = Path(__file__).resolve().parent / "06_build_audit_sheets.py"
    spec = importlib.util.spec_from_file_location("audit06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def take(df, n, rng, by="model"):
    """n rows spread evenly over `by`, preferring problems not used yet."""
    if len(df) <= n:
        return df
    df = df.iloc[rng.permutation(len(df))]
    groups = [list(x.index) for _, x in df.groupby(by)]
    out, seen = [], set()
    while len(out) < n and any(groups):
        for gidx in groups:
            while gidx:
                i = gidx.pop(0)
                if df.at[i, "problem_id"] not in seen or not any(
                        df.at[j, "problem_id"] not in seen for j in gidx):
                    out.append(i)
                    seen.add(df.at[i, "problem_id"])
                    break
            if len(out) >= n:
                break
    return df.loc[out]


def add_helper_formulas(path, df, probs, key_by_id):
    """Fill off_by / within_tolerance with formulas on the number the annotator types.
    The annotator types the response's final number in the REFERENCE unit."""
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter
    wb = load_workbook(path)
    ws = wb["labels"]
    col = {name: get_column_letter(i + 1) for i, name in enumerate(df.columns)}
    F = col["final_number_found"]
    for r, iid in enumerate(df.audit_item_id, start=2):
        p = probs[key_by_id[iid]]
        ref = repr(float(p["answer"]))
        tol = p.get("tolerance") or {}
        tv = float(tol.get("value", 0.01))
        cell = f"{F}{r}"
        if str(tol.get("type", "rel")).lower().startswith("rel"):
            off = f'=IF({cell}="","",TEXT(ABS({cell}-{ref})/ABS({ref})*100,"0.00")&"%")'
            ok = f'=IF({cell}="","",IF(ABS({cell}-{ref})/ABS({ref})<={tv},"YES","NO"))'
        else:
            unit = p.get("unit", "")
            off = f'=IF({cell}="","",TEXT(ABS({cell}-{ref}),"0.00")&" {unit}")'
            ok = f'=IF({cell}="","",IF(ABS({cell}-{ref})<={tv},"YES","NO"))'
        ws[f'{col["off_by"]}{r}'] = off
        ws[f'{col["within_tolerance"]}{r}'] = ok
    for name, w in [("final_number_found", 13), ("off_by", 10), ("within_tolerance", 11)]:
        ws.column_dimensions[col[name]].width = w
    wb.save(path)


def build(a):
    rng = np.random.default_rng(a.seed)
    a06 = load_06()
    g = pd.read_csv(a.gen).merge(pd.read_csv(a.scores), on="response_id")
    g = g.merge(pd.read_csv(a.judge)[["response_id", "grader_label"]], on="response_id")
    gl = pd.read_csv(a.glimmer_gen).merge(pd.read_csv(a.glimmer_scores), on="response_id")
    gl = gl.merge(pd.read_csv(a.glimmer_judge)[["response_id", "grader_label"]], on="response_id")
    probs = {json.loads(x)["problem_id"]: json.loads(x) for x in open(a.problems, encoding="utf-8") if x.strip()}
    jr = lambda d: d.grader_label.isin(["CORRECT", "RIGHT_ANSWER_WRONG_REASONING"])

    core = g[g.model.isin(CORE)]
    G1 = take(core[(core.grader_label == "RIGHT_ANSWER_WRONG_REASONING") & (core.model != "google/gemma-2-9b-it")], 40, rng)
    G2 = take(core[(core.grader_label == "CORRECT") & (core.final_correct == 1)], 30, rng)
    gl_ok = gl[gl.response.fillna("").str.strip() != ""]
    G3 = take(gl_ok[jr(gl_ok) & (gl_ok.final_correct == 0)], 30, rng, by="cell")
    used = set(pd.concat([G1, G2]).response_id)
    G4 = take(g[(g.grader_label != "UNPARSED") & ~g.response_id.isin(used)], 20, rng)
    S = pd.concat([G1.assign(group="G1_judge_RAWR"), G2.assign(group="G2_judge_and_code_correct"),
                   G3.assign(group="G3_glimmer_code_wrong_judge_right"), G4.assign(group="G4_random")],
                  ignore_index=True)
    S = S.iloc[rng.permutation(len(S))].reset_index(drop=True)
    S["audit_item_id"] = [f"P-T{i:03d}" for i in range(1, len(S) + 1)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CODEBOOK_reasoning_check_v1.md").write_text(CODEBOOK, encoding="utf-8")
    sha = hashlib.sha256(CODEBOOK.encode()).hexdigest()
    for i, ann in enumerate(a.annotators):
        order = np.random.default_rng([a.seed, i]).permutation(len(S))
        rows = []
        for r in S.iloc[order].itertuples():
            p = probs[r.problem_id]
            row = dict(audit_item_id=r.audit_item_id, problem=p["neutral_statement"],
                       reference_solution=a06.fmt_solution(p), reference_answer=a06.fmt_answer(p),
                       response=r.response if isinstance(r.response, str) else "")
            row.update({c: "" for c in HELP + FILL})
            rows.append(row)
        df = pd.DataFrame(rows)
        path = out / f"sheet_{ann}.xlsx"
        a06.write_xlsx(path, df, ["final_number_found"] + FILL, CODEBOOK)
        add_helper_formulas(path, df, probs, key_by_id={x.audit_item_id: x.problem_id for x in S.itertuples()})
    key = S[["audit_item_id", "group", "response_id", "model", "problem_id", "cell", "grader_label",
             "final_correct", "rel_err", "finish_reason"]]
    key.to_csv(out / "audit_key_PRIVATE.csv", index=False)
    json.dump(dict(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=a.seed,
                   annotators=a.annotators, n_items=len(S), groups=S.group.value_counts().to_dict(),
                   protocol="reference shown, framing hidden (neutral statement), full response, model hidden",
                   codebook_sha256=sha), open(out / "audit_manifest.json", "w"), indent=2)
    print(S.groupby(["group", "model"]).size().to_string())
    print(f"\n{len(S)} items, {S.problem_id.nunique()} distinct problems; sheets for {a.annotators}")
    print(f"codebook sha256 {sha[:16]}; wrote {out}/")


def kappa(x, y):
    x, y = list(x), list(y)
    cats = sorted(set(x) | set(y))
    n = len(x)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def collect(a):
    out = Path(a.out)
    key = pd.read_csv(out / "audit_key_PRIVATE.csv")
    L = []
    for f in sorted(glob.glob(str(out / "**" / "sheet_*.xlsx"), recursive=True)):
        ann = re.sub(r"^sheet_|\.xlsx$", "", Path(f).name)
        d = pd.read_excel(f, sheet_name="labels", dtype=str).fillna("")
        d["annotator"] = ann
        L.append(d)
    if not L:
        sys.exit(f"no sheet_*.xlsx under {out}")
    H = pd.concat(L)
    H["label"] = H.label.str.upper().str.strip().str.replace(r"[^A-Z]+", "_", regex=True).str.strip("_")
    H["fa"] = H.final_answer_correct.str.upper().str[:1].map({"Y": 1, "N": 0})
    H["minutes"] = pd.to_numeric(H.minutes.str.replace(",", "."), errors="coerce")
    H = H.merge(key, on="audit_item_id")
    H.to_csv(out / "human_labels_long.csv", index=False)
    done = H[H.label.isin(LABELS)]
    print(f"labelled rows: {len(done)} of {len(H)}; missing rationale: {(done.rationale.str.strip() == '').sum()}; "
          f"median minutes/item: {done.minutes.median()}")
    anns = sorted(done.annotator.unique())
    if len(anns) >= 2:
        w = done.pivot_table(index="audit_item_id", columns="annotator", values="label", aggfunc="first").dropna()
        wf = done.pivot_table(index="audit_item_id", columns="annotator", values="fa", aggfunc="first").dropna()
        print(f"\ninter-annotator ({anns[0]} vs {anns[1]}, n={len(w)}): label agreement "
              f"{(w[anns[0]] == w[anns[1]]).mean():.3f}, Cohen's kappa {kappa(w[anns[0]], w[anns[1]]):.3f}; "
              f"final-answer kappa {kappa(wf[anns[0]], wf[anns[1]]):.3f}")
    done = done.assign(human_rawr=(done.label == "RIGHT_ANSWER_WRONG_REASONING").astype(int),
                       human_right=done.label.isin(["CORRECT", "RIGHT_ANSWER_WRONG_REASONING"]).astype(int),
                       same_as_judge=(done.label == done.grader_label).astype(int))
    rep = done.groupby("group").agg(items=("audit_item_id", "nunique"), same_label_as_judge=("same_as_judge", "mean"),
                                    human_says_RAWR=("human_rawr", "mean"), human_says_answer_right=("human_right", "mean"),
                                    code_says_right=("final_correct", "mean"), median_minutes=("minutes", "median"))
    print("\nby group (averaged over annotators):")
    print(rep.round(3).to_string())
    rep.to_csv(out / "human_check_by_group.csv")
    g3 = done[done.group.str.startswith("G3")]
    if len(g3):
        p = g3.human_right.mean()
        print(f"\nGlimmer: of responses the code marked wrong but the judge marked right, people say the "
              f"answer is right in {p:.1%} (n={g3.audit_item_id.nunique()} items).")
        print("Corrected Glimmer accuracy = (code-right + p x code-wrong-judge-right) / non-empty responses; "
              "fill in the counts from T-Glimmer.")
    print(f"\nwrote {out}/human_labels_long.csv and human_check_by_group.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "collect"], default="build")
    for k in ["gen", "scores", "judge", "glimmer-gen", "glimmer-scores", "glimmer-judge", "problems"]:
        ap.add_argument(f"--{k}")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="audit_120")
    a = ap.parse_args()
    build(a) if a.mode == "build" else collect(a)


if __name__ == "__main__":
    main()
