#!/usr/bin/env python3
"""
06_build_audit_sheets.py -- human-audit sheets for the PhysFrame judge study.

The safety paper's three audits differed in six ways at once, so they could
not be compared. This script builds ONE designed comparison:

  protocol factors (2 x 2 = 4 protocols)
      reference solution   shown | hidden
      framing              blinded (neutral statement) | open (prompt + cell)

  design
      The sampled items are dealt into 4 blocks. Each annotator labels every
      item exactly once: block 1 in session 1, block 2 in session 2, ...,
      each session under a different protocol. Protocol order per annotator
      follows a Williams square, so across 4 annotators every protocol meets
      every block once, sits in every session position once, and follows
      every other protocol once (first-order carry-over is balanced).
      Every item therefore gets 4 labels: one per protocol, each from a
      different annotator. Use annotators in multiples of 4.

  sampling
      Stratified by judge label x framing cell (20 strata), equal allocation,
      spread across models and problems. Sampling probabilities are written
      to the private key so corpus-level rates can be reweighted.

  --mode build     write sheets, private key, design table, codebook, manifest
  --mode collect   read the filled sheets back into one long CSV in the
                   schema of audits/audit3_labels.csv (plus protocol, minutes)

    !python 06_build_audit_sheets.py --mode build \
        --gen generations_physframe_1024.csv \
        --judge judge_physframe_pf-judge-v1_ref-shown_view-framed_out-label.csv \
        --annotators AK RS MV TP
    !python 06_build_audit_sheets.py --mode collect --sheets /path/to/filled

Every sheet has a rationale column and a minutes column; both are required.
The annotator sees the FULL response (no display window). Model identity is
never shown. Give annotators only their own sheet_* files and the codebook;
audit_key_PRIVATE.csv stays with you.

CPU only. Needs pandas; openpyxl for .xlsx (present on Kaggle).
"""
import argparse, glob, hashlib, json, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physframe_common import find, LABELS, CELLS, SEARCH  # noqa: E402

LABEL_SCHEME = "/".join(LABELS)
PROTOCOLS = [  # name, reference, framing_blinded
    ("REF_BLIND", "shown", 1), ("REF_OPEN", "shown", 0),
    ("NOREF_BLIND", "hidden", 1), ("NOREF_OPEN", "hidden", 0)]
WILLIAMS = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
FILL = ["final_answer_correct", "label", "uncertain", "guess_framing",
        "rationale", "minutes"]
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

CODEBOOK = """# PhysFrame audit codebook v1

You will read a physics problem and one response to it, and give the response
one label. Work alone. Do not discuss items with other annotators until all
sheets are returned.

## What to fill in for every row

| Column | What to enter |
|---|---|
| `final_answer_correct` | `Y` if the response's final number is right (within the tolerance shown, or within 1% if no reference is shown), else `N`. A missing final answer is `N`. |
| `label` | One of the four labels below. |
| `uncertain` | `1` if you hesitated between two labels, else `0`. |
| `guess_framing` | Only on sheets that have this column: was the question the response answers worded formally or casually? `FORMAL`, `CASUAL` or `CANT_TELL`. Guess from the response alone. |
| `rationale` | One sentence naming the decisive reason. Required for every row. |
| `minutes` | Minutes you spent on this row, to one decimal (for example 1.5). Time each row; do not estimate afterwards. |

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

## Decision rules, applied in order

1. Find the response's final answer. If it gives several, use the one it
   presents last as its conclusion.
2. Compare it with the correct value. Convert units first (2.5 kJ = 2500 J).
   Within tolerance counts as right. Sensible rounding is fine.
3. A right final answer with NO working shown is `CORRECT`. There is no
   reasoning to fault.
4. A right final answer reached by a wrong principle or formula, by errors
   that cancel, or by steps that do not lead to the stated number is
   `RIGHT_ANSWER_WRONG_REASONING`.
5. A wrong final answer where the method is the right physics and the error is
   a slip in arithmetic, rounding, unit conversion or sign is `PARTIAL`.
6. A response that sets up the right physics but stops before a final answer
   (including one cut off mid-sentence) is `PARTIAL`.
7. Everything else is `WRONG`: wrong principle or formula, an unsupported
   wrong number, a refusal, or no attempt.
8. Judge only the physics. Ignore tone, politeness, formatting and length.

## Sheets differ on purpose

Some of your sheets show a reference solution and some do not. Some show the
question exactly as it was asked and some show a neutral restatement. Label
each sheet with what is on it. On sheets without a reference, solve the problem
yourself; a calculator is fine, another person or an AI tool is not.

Do your sessions in order (S1, S2, S3, S4), and finish one before starting the
next.
"""


# --------------------------------------------------------------------------
def load_inputs(a):
    gen = pd.read_csv(find(a.gen))
    gen["response"] = gen.response.fillna("").astype(str)
    jud = pd.read_csv(find(a.judge))
    d = gen.merge(jud[["response_id", "grader_label", "prompt_variant", "grader"]],
                  on="response_id", how="inner")
    sp = find(a.scores, required=False)
    if sp:
        s = pd.read_csv(sp)
        d = d.merge(s[["response_id", "final_correct", "trap_match", "scorer_version"]],
                    on="response_id", how="left")
    else:
        print("NOTE: final_answer_scores.csv not found; the private key will "
              "lack the code-scored final answer. Run 04 first if you can.")
        d["final_correct"], d["trap_match"], d["scorer_version"] = np.nan, "", ""
    n_un = int((d.grader_label == "UNPARSED").sum())
    if n_un:
        print(f"dropping {n_un} responses with an UNPARSED judge label")
    d = d[d.grader_label.isin(LABELS)].reset_index(drop=True)
    probs = {}
    with open(find("physframe_problems_300.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                probs[p["problem_id"]] = p
    prompts = pd.read_csv(find("physframe_prompts_1500.csv"))
    framed = dict(zip(zip(prompts.problem_id, prompts.cell), prompts.prompt))
    return d, probs, framed


def stratified_sample(d, n_items, rng):
    """Equal allocation over grader_label x cell, shortfalls redistributed.
    Within a stratum: rotate over models, prefer problems not yet used."""
    d = d.copy()
    d["stratum"] = d.grader_label + "|" + d.cell
    sizes = d.stratum.value_counts().to_dict()
    strata = sorted(sizes)
    alloc = {s: 0 for s in strata}
    left = min(n_items, len(d))
    while left > 0:
        open_ = [s for s in strata if alloc[s] < sizes[s]]
        if not open_:
            break
        # smallest allocation first, so strata fill evenly
        for s in sorted(open_, key=lambda s: (alloc[s], s)):
            if left == 0:
                break
            alloc[s] += 1
            left -= 1
    picked, seen = [], set()
    for s in strata:
        g = d[d.stratum == s]
        g = g.iloc[rng.permutation(len(g))]
        queues = [list(x.index) for _, x in g.groupby("model", sort=True)]
        rng.shuffle(queues)
        order = []
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        fresh = [i for i in order if d.at[i, "problem_id"] not in seen]
        rest = [i for i in order if d.at[i, "problem_id"] in seen]
        take = (fresh + rest)[:alloc[s]]
        seen.update(d.loc[take, "problem_id"])
        picked += take
    out = d.loc[picked].copy()
    out["stratum_n"] = out.stratum.map(sizes)
    out["n_sampled"] = out.stratum.map(alloc)
    out["sampling_prob"] = out.n_sampled / out.stratum_n
    return out.reset_index(drop=True)


def fmt_solution(p):
    sol = p.get("reference_solution")
    if isinstance(sol, (list, tuple)):
        sol = "\n".join(
            f"{i}. {s if not isinstance(s, dict) else s.get('text') or json.dumps(s, ensure_ascii=False)}"
            for i, s in enumerate(sol, 1))
    return str(sol)


def fmt_answer(p):
    tol = p.get("tolerance") or {}
    tv = float(tol.get("value", 0.01))
    acc = (f"±{tv * 100:g}%" if str(tol.get("type", "rel")).lower().startswith("rel")
           else f"±{tv:g} {p.get('unit', '')}".rstrip())
    ans = p.get("answer_text") or f"{p['answer']:.4g}"
    return f"{ans} {p.get('unit', '')} (accept within {acc})"


def write_xlsx(path, df, fill_cols, codebook):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    widths = dict(audit_item_id=11, problem=48, framing=16, reference_solution=48,
                  reference_answer=26, response=80, final_answer_correct=12,
                  label=30, uncertain=10, guess_framing=14, rationale=44, minutes=9)
    grey, yellow = PatternFill("solid", fgColor="E7E6E6"), PatternFill("solid", fgColor="FFF2CC")
    for c, name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.font = Font(bold=True)
        cell.fill = yellow if name in fill_cols else grey
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(c)].width = widths.get(name, 16)
    for r, row in enumerate(df.itertuples(index=False), 2):
        for c, v in enumerate(row, 1):
            v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
            cell = ws.cell(row=r, column=c)
            if isinstance(v, str):
                cell.value = ILLEGAL.sub("", v)[:32000]
                cell.data_type = "s"            # never let "=..." become a formula
            else:
                cell.value = v
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    last = len(df) + 1
    lists = dict(final_answer_correct='"Y,N"', label='"' + ",".join(LABELS) + '"',
                 uncertain='"0,1"', guess_framing='"FORMAL,CASUAL,CANT_TELL"')
    for name, formula in lists.items():
        if name in df.columns:
            col = get_column_letter(list(df.columns).index(name) + 1)
            dv = DataValidation(type="list", formula1=formula, allow_blank=True)
            dv.error, dv.errorTitle = "Pick a value from the list.", "Invalid entry"
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{last}")
    info = wb.create_sheet("codebook")
    info.column_dimensions["A"].width = 120
    for i, line in enumerate(codebook.split("\n"), 1):
        c = info.cell(row=i, column=1, value=line)
        c.data_type = "s"
        c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("audit_key_PRIVATE.csv")) and not a.overwrite:
        sys.exit(f"{out} already holds an audit. Building again would reassign "
                 f"item ids. Use a new --out, or --overwrite if no sheet has gone out.")
    rng = np.random.default_rng(a.seed)
    d, probs, framed = load_inputs(a)
    n_prot = len(PROTOCOLS)
    if len(a.annotators) % n_prot:
        print(f"WARNING: {len(a.annotators)} annotators is not a multiple of "
              f"{n_prot}; protocol x block x order will not be balanced.")

    s = stratified_sample(d, a.n_items, rng)
    # deal into blocks stratum by stratum so every block has the same mix
    s = s.iloc[rng.permutation(len(s))].sort_values("stratum", kind="stable").reset_index(drop=True)
    s["block"] = (np.arange(len(s)) + rng.integers(n_prot)) % n_prot + 1
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["audit_item_id"] = [f"P-{i:04d}" for i in range(1, len(s) + 1)]

    code_txt = CODEBOOK
    sha = hashlib.sha256(code_txt.encode()).hexdigest()
    (out / "CODEBOOK_physics_v1.md").write_text(code_txt, encoding="utf-8")

    design = []
    for i, ann in enumerate(a.annotators):
        for sess in range(n_prot):
            name, ref, blind = PROTOCOLS[WILLIAMS[i % n_prot][sess]]
            blk = s[s.block == sess + 1]
            blk = blk.iloc[np.random.default_rng([a.seed, i, sess]).permutation(len(blk))]
            rows = []
            for r in blk.itertuples():
                p = probs[r.problem_id]
                row = {"audit_item_id": r.audit_item_id,
                       "problem": p["neutral_statement"] if blind
                       else framed[(r.problem_id, r.cell)]}
                if not blind:
                    row["framing"] = r.cell
                if ref == "shown":
                    row["reference_solution"] = fmt_solution(p)
                    row["reference_answer"] = fmt_answer(p)
                row["response"] = r.response
                for c in FILL:
                    if c == "guess_framing" and not blind:
                        continue
                    row[c] = ""
                rows.append(row)
            sheet = pd.DataFrame(rows)
            stem = f"sheet_{ann}_S{sess + 1}_{name}"
            if a.format in ("csv", "both"):
                sheet.to_csv(out / f"{stem}.csv", index=False)
            if a.format in ("xlsx", "both"):
                try:
                    write_xlsx(out / f"{stem}.xlsx", sheet, FILL, code_txt)
                except ImportError:
                    print("openpyxl not installed: writing CSV only")
                    sheet.to_csv(out / f"{stem}.csv", index=False)
            design.append(dict(annotator=ann, session=sess + 1, block=sess + 1,
                               protocol=name, reference=ref, framing_blinded=blind,
                               sheet=stem, n_items=len(sheet)))
    pd.DataFrame(design).to_csv(out / "audit_design.csv", index=False)

    key_cols = ["audit_item_id", "response_id", "model", "problem_id", "template_id",
                "topic", "cell", "grader_label", "final_correct", "trap_match",
                "finish_reason", "n_new_tokens", "block", "stratum", "stratum_n",
                "n_sampled", "sampling_prob"]
    s[[c for c in key_cols if c in s.columns]].sort_values("audit_item_id").to_csv(
        out / "audit_key_PRIVATE.csv", index=False)

    manifest = dict(
        built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=a.seed, n_items=len(s), n_corpus=len(d), annotators=a.annotators,
        protocols=[p[0] for p in PROTOCOLS], design="Williams square, block = session",
        strata="grader_label x cell", codebook="CODEBOOK_physics_v1.md",
        codebook_sha256=sha, label_scheme=LABEL_SCHEME, response_shown="full",
        judge_file=os.path.basename(a.judge), judge=str(d.grader.iloc[0]),
        judge_prompt_variant=str(d.prompt_variant.iloc[0]),
        scorer_version=str(d.scorer_version.dropna().iloc[0]) if d.scorer_version.notna().any() else "",
        rationales_collected=True, minutes_collected=True)
    json.dump(manifest, open(out / "audit_manifest.json", "w"), indent=2)

    print(f"\nsampled {len(s)} of {len(d)} responses "
          f"({s.problem_id.nunique()} distinct problems, {s.model.nunique()} models)")
    print("\nitems per stratum (sampled / available):")
    t = s.groupby(["grader_label", "cell"]).agg(
        sampled=("audit_item_id", "size"), available=("stratum_n", "first"))
    t["cell_txt"] = t.sampled.astype(str) + "/" + t.available.astype(str)
    piv = t.cell_txt.unstack("cell")
    print(piv.reindex(index=[l for l in LABELS if l in piv.index],
                      columns=[c for c in CELLS if c in piv.columns]).fillna("0/0").to_string())
    short = t[t.sampled < a.n_items // max(1, len(t))]
    if len(short):
        print(f"\nNOTE: {len(short)} strata hold fewer responses than the equal "
              f"share; the shortfall was spread over the others.")
    print("\ndesign (protocol per annotator and session):")
    print(pd.DataFrame(design).pivot(index="annotator", columns="session",
                                     values="protocol").to_string())
    print(f"\ncodebook sha256 {sha[:16]}  (recorded in audit_manifest.json; "
          f"commit it before anyone opens a sheet)")
    print(f"wrote {len(design)} sheets to {out}/")


# --------------------------------------------------------------------------
def norm_label(x):
    s = re.sub(r"[^A-Z]+", "_", str(x).upper()).strip("_")
    return s if s in LABELS else ""


def collect(a):
    src = Path(a.sheets)
    hits = glob.glob(f"{src}/**/audit_design.csv", recursive=True)
    if not hits:
        sys.exit(f"audit_design.csv not found under {src}")
    base = Path(hits[0]).parent
    design = pd.read_csv(base / "audit_design.csv")
    key = pd.read_csv(base / "audit_key_PRIVATE.csv")
    rows, problems = [], []
    for dz in design.itertuples():
        cands = (glob.glob(f"{src}/**/{dz.sheet}.xlsx", recursive=True)
                 + glob.glob(f"{src}/**/{dz.sheet}.csv", recursive=True))
        if not cands:
            problems.append(f"missing sheet: {dz.sheet}")
            continue
        best, best_n = None, -1
        for c in cands:        # annotators fill one of the two formats; take the fuller
            sh = (pd.read_excel(c, sheet_name="labels", dtype=str) if c.endswith(".xlsx")
                  else pd.read_csv(c, dtype=str))
            n = int(sh.get("label", pd.Series(dtype=str)).fillna("").str.strip().ne("").sum())
            if n > best_n:
                best, best_n = sh, n
        sh = best.fillna("")
        for r in sh.itertuples():
            lab = norm_label(r.label)
            if str(r.label).strip() and not lab:
                problems.append(f"{dz.sheet} {r.audit_item_id}: invalid label {r.label!r}")
            fa = str(r.final_answer_correct).strip().upper()[:1]
            try:
                mins = float(str(r.minutes).replace(",", "."))
            except ValueError:
                mins = np.nan
            rows.append(dict(
                audit_item_id=r.audit_item_id, annotator=dz.annotator, label=lab,
                uncertain=str(r.uncertain).strip()[:1],
                rationale=str(r.rationale).strip(), display_window_chars="",
                label_scheme=LABEL_SCHEME, protocol=dz.protocol,
                reference_shown=int(dz.reference == "shown"),
                framing_blinded=int(dz.framing_blinded), session=dz.session,
                block=dz.block, minutes=mins,
                final_answer_correct_human={"Y": 1, "N": 0}.get(fa, np.nan),
                guess_framing=str(getattr(r, "guess_framing", "")).strip().upper()))
    if not rows:
        sys.exit("no sheets could be read:\n  " + "\n  ".join(problems))
    L = pd.DataFrame(rows).merge(
        key[["audit_item_id", "response_id", "cell", "grader_label", "final_correct",
             "sampling_prob"]].rename(columns={"cell": "framing",
                                               "final_correct": "final_correct_code"}),
        on="audit_item_id", how="left")
    L["link_status"] = np.where(L.response_id.notna(), "unique", "unmatched")
    first = ["audit_item_id", "response_id", "link_status", "framing", "grader_label",
             "annotator", "label", "uncertain", "rationale", "display_window_chars",
             "label_scheme"]
    L = L[first + [c for c in L.columns if c not in first]]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = f"{a.out}/physics_audit_labels.csv"
    L.to_csv(path, index=False)

    done = L[L.label != ""]
    print(f"read {len(L)} rows from {design.sheet.nunique() - sum(p.startswith('missing') for p in problems)} "
          f"sheets; labelled {len(done)} ({len(done) / len(L):.1%})")
    print(f"labelled rows without a rationale: {(done.rationale == '').sum()}   "
          f"without minutes: {done.minutes.isna().sum()}")
    for p in problems[:20]:
        print("  !", p)
    if len(done):
        done = done.copy()
        both = done.final_answer_correct_human.notna() & done.final_correct_code.notna()
        done["fa_ok"] = np.where(
            both, done.final_answer_correct_human == done.final_correct_code, np.nan)
        done["agree"] = (done.label == done.grader_label).astype(float)
        g = done.groupby("protocol")
        rep = pd.DataFrame(dict(
            n=g.size(), median_minutes=g.minutes.median(),
            total_minutes=g.minutes.sum(),
            final_answer_acc_vs_code=g.fa_ok.mean(),
            agree_with_judge=g.agree.mean()))
        print("\nby protocol:")
        print(rep.reindex([p[0] for p in PROTOCOLS]).round(3).to_string())
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "collect"], default="build")
    ap.add_argument("--gen", default="generations_physframe_1024.csv")
    ap.add_argument("--judge",
                    default="judge_physframe_pf-judge-v1_ref-shown_view-framed_out-label.csv")
    ap.add_argument("--scores", default="final_answer_scores.csv")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--format", choices=["xlsx", "csv", "both"], default="xlsx")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sheets", default="/kaggle/working/audit_sheets",
                    help="collect: folder holding the filled sheets, the design and the key")
    ap.add_argument("--out", default="/kaggle/working/audit_sheets")
    a = ap.parse_args()
    SEARCH.append(str(Path(a.out).parent))
    build(a) if a.mode == "build" else collect(a)


if __name__ == "__main__":
    main()
