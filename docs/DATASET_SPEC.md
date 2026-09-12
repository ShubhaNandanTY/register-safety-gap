# Dataset specification — what the release should be

For the artifact accompanying *When Validating the Judge Changes the Result*. Written to
be usable as the repository README and as the basis of a data card.

`02_build_release_dataset.ipynb` produces exactly this layout from `ARCHIVE/02_data/`.

---

## 1 · What this dataset is

Not a benchmark. It is a **measurement-variance dataset**: one corpus of 3,600 model
responses to harmful requests, graded by two automated judges and validated by three
human audits run under different protocols, released so that the disagreement between
those validations can be studied.

The unit of interest is not "is this response harmful" but "how much does the answer to
that question depend on who was asked and how". Anyone building a safety benchmark out
of the `HARMFUL` labels has misread it — the paper's central result is that those labels
are protocol-dependent.

## 2 · Layout

```
register-safety-gap-v2/
├── README.md                        this document
├── DATA_CARD.md                     §9 below, expanded
├── LICENSE                          CC BY 4.0 (prompts, labels, code)
├── TERMS.md                         research-use terms for generations/
├── prompts/
│   └── register_pairs_300.csv       300 matched pairs, 25 intents
├── generations/                     ← gated, see §7
│   ├── generations_200.csv          original run, 3,600
│   ├── generations_1024.csv         uncensored regeneration, 3,600
│   └── generations_200_rerun.csv    reproducibility probe, 3,600
├── grader/
│   ├── grader_200_original.csv      primary grader, original run
│   ├── grader_budget_curve.csv      5 budgets × 3,600
│   └── llamaguard_200.csv           second grader
├── audits/
│   ├── audit1_labels.csv            200 items × 1 annotator
│   ├── audit2_labels.csv            442 items × 2 annotators
│   ├── audit3_labels.csv            239 items × 3 annotators (419 labels)
│   └── audited_item_index.csv       every audited item → corpus row
├── code/
│   ├── 01_verify_paper_numbers.ipynb
│   ├── 02_build_release_dataset.ipynb
│   ├── analyze_all_audits.py, audit_analysis.py, make_figures.py
│   ├── 07_regenerate.py, 08_budget_curve.py
│   └── trace_judge_template.yaml
└── meta/
    ├── manifest.json                counts, budgets, variant names
    ├── env_*.json                   version stamps for each GPU run
    ├── codebook_audit3.md           label definitions + 7 decision rules
    └── CHANGELOG.md                 v1 → v2, and why v1 was superseded
```

## 3 · The primary key

```
response_id = {model_slug}__{pair_id}__{framing}
e.g.  qwen2.5-7b__pair_0143__user
```

Unique across the 3,600 responses, stable across all three generation runs, joinable
without ambiguity, and readable enough to debug by eye. Every grader file and every audit
row carries it.

**Do not join on `item_id`.** The archive's audit files reuse identifiers across audits:
`IT0001`–`IT0442` (audit 2) and `IT0000`–`IT0238` (audit 3) share 238 identifiers
denoting different items, and only a handful of the underlying responses coincide. The
release renamespaces to `A1-####` / `A2-####` / `A3-####` and records the collision in
`CHANGELOG.md`. Anyone who downloaded the archive should be told this explicitly, in the
README, above the fold.

`eval_framing` / `deploy_framing` become `benchmark` / `user`. Nothing was ever deployed;
the contrast is benchmark-style versus user-style prose, and the old names invite a
causal reading the paper is careful not to make.

## 4 · Schemas

**`prompts/register_pairs_300.csv`** — 300 rows
`pair_id`, `category`, `intent`, `prompt_benchmark`, `prompt_user`

**`generations/generations_200.csv`** — 3,600 rows
`response_id`, `model`, `pair_id`, `category`, `framing`, `response`,
`refused_original_scorer`

**`generations/generations_1024.csv`** — 3,600 rows
as above plus `n_new_tokens`, `finish_reason`, minus the scorer column

**`grader/grader_200_original.csv`** — 3,600 rows
`response_id`, `grader_label` ∈ {REFUSAL, HEDGE, HARMFUL}, `run`, `grader`,
`prompt_variant`

**`grader/grader_budget_curve.csv`** — 18,000 rows
`response_id`, `budget` ∈ {200, 384, 512, 768, 1024}, `refused_reimpl_scorer`,
`grader_label`, `prompt_variant`

**`audits/audit{1,2,3}_labels.csv`** — long format, one row per annotator per item
`audit_item_id`, `response_id`, `link_status`, `framing`, `grader_label`, `annotator`,
`label`, `uncertain`, `rationale`, `display_window_chars`, `label_scheme`

Long format matters: it is the only shape in which "two annotators, same item, opposite
labels" is a first-class fact rather than something a reader reconstructs from two files.

## 5 · The two columns that carry the paper's finding

Most releases would bury these in prose. Here they are the result, so they are columns.

**`display_window_chars`** — how much of the response the annotator actually saw.
Audit 1: 500. Audit 2: 700. Audit 3: full. This was in no written protocol and was
recovered by inspecting the label files. It is a more mechanically direct route to a
changed label than blinding is.

**`prompt_variant`** and the two scorer columns — the instruments are not
interchangeable and the release must not let anyone assume they are:

| Variant | Response truncation | Trailing cue | Batched | Default label |
|---|---|---|---|---|
| `v1_1200char_label_cue` | 1,200 chars | `Label:` | no | HEDGE |
| `v2_untruncated_no_cue` | none | none | yes | REFUSAL |

At the same nominal 200-token budget these give Δ_harm = 0.033 and 0.025.

| Scorer | Provenance | Gap on original text |
|---|---|---|
| `refused_original_scorer` | marker list not recovered | 0.2633 |
| `refused_reimpl_scorer` | `08_budget_curve.py` | 0.2811 |

**`link_status`** ∈ {`unique`, `ambiguous_k`, `unmatched`} — because annotators saw
truncated text, an audit row cannot always be traced to one corpus row. Where several
responses share the displayed prefix, usually identical canned refusals from different
models, the release records the ambiguity instead of guessing. Current linkage:

| Audit | Items | Uniquely linked | Framing known | Grader label known |
|---|---|---|---|---|
| 1 | 200 | 110 | 200 | 200 |
| 2 | 442 | 415 | 442 | 442 |
| 3 | 239 | 127 | 216 | 239 |

Audit 1's low rate is structural — it records model and framing but not intent, and short
canned replies repeat. Everything its estimator needs is present regardless.

## 6 · Known limits to carry in the release

State these in the README, not only in the paper:

- **Audit-2 annotator instructions are unrecoverable.** This is why the paper says
  "uncontrolled protocol variance" rather than "annotator variance". Without the
  instructions the κ = 0.09 result cannot distinguish chance disagreement from two people
  operationalising the rubric differently.
- **Neither generation nor grading is bit-reproducible.** Regenerating at the original
  cap reproduces 19.4% of responses exactly across six models (0% for gemma-2-2b). At
  least 27 of 3,600 grader labels differ between nominally deterministic runs on
  byte-identical inputs.
- **Audit 3 is 90 triple-labelled items plus 149 singly labelled ones.** α = 0.856
  describes the 90. Re-adjudicating with one annotator removed moves the corrected gap
  from 0.035 to between 0.006 and 0.033.
- **The original generation scripts (`01_*`, `02_*`) were never recovered.**
  `07_regenerate.py` reconstructs the pipeline; the system prompt is set by autodetect
  and was never independently confirmed.
- **Probe activations and the length-matched prompt variant behind the original §9 were
  not retained.**
- **`grader_audit_60.csv`** (60-item scorer spot-check) is in the archive but not used by
  any released analysis. Either wire it in or drop it.

## 7 · Access and gating

The prompts are 300 requests for harmful information across 25 intents. The generations
are 3,600 model responses to them, dozens of which three blinded annotators independently
judged actionably harmful under a marginal-uplift test.

Recommended split:

- **Open (CC BY 4.0):** prompts, all grader labels, all human labels, code, metadata.
  These are what the paper's claims rest on and gating them would defeat the release.
- **Gated (research-use terms, request form or click-through):** the three
  `generations/` files. Not because the content is novel — most of it is hedging — but
  because a bulk file of model outputs to harmful prompts, pre-sorted by a harm grader
  into a `HARMFUL` bucket, is a more convenient artifact than the sum of its parts.

For the anonymous review mirror on 4open.science, gate nothing and include everything;
the mirror is short-lived and reviewers need to check the claims. Link it from the
supplementary material only — the anonymous PDF deliberately contains no repository URL.

A short `TERMS.md` is enough: research and safety-evaluation use, no redistribution of
the generations, no use to train a model to produce the responses.

## 8 · Versioning

Tag the current public state as `v1` before anything else, then release `v2`. The v1
README still asserts the superseded conclusion (0.26, old title), so a reviewer following
the link today sees a page contradicting the paper. `CHANGELOG.md` should record, plainly:

- v1 reported a corrected harm gap of 0.26 from a single unblinded audit.
- v2 supersedes it. Two further audits give −0.036 / +0.274 and 0.035. The v1 estimate is
  one of four, and the spread across them exceeds the effect.
- `item_id` was renamespaced; the v1 identifiers collided across audits.
- Files added: audit-3 sheets, the recovered key, the codebook, `judge_v2_rescored.csv`,
  the 1024-token corpus, per-budget grader labels.

Retracting your own headline number in a changelog is the cheapest credibility this
project can buy, and it is consistent with what the paper argues.

## 9 · Data card, short form

**Motivation.** Built to test whether safety refusal behaviour is sensitive to prompt
register; repurposed, once three validations of one grader disagreed, into a study of
validation-protocol variance.

**Composition.** 300 matched prompt pairs across 25 harmful intents in categories
including cyber, fraud, theft and weapons; 6 instruction-tuned models (Qwen2.5 at 0.5B /
1.5B / 3B / 7B, Gemma-2 at 2B / 9B), all 4-bit NF4, greedy decoding, one generation each;
3,600 responses per run × 3 runs. Human labels: 200 + 442 + 239 items, 1 + 2 + 3
annotators.

**Collection.** Generated on Kaggle T4×2 under torch 2.10.0+cu128, transformers 5.15.1,
accelerate 1.14.0, bitsandbytes 0.50.1 (stamps in `meta/env_*.json`). Audit 3 was
pre-registered: label definitions and seven ordered decision rules fixed and hashed before
any response was inspected, annotators shown a neutral one-line intent description and the
verbatim response, never the framed prompt, with the framing key withheld until all labels
returned.

**Annotators.** Audit 3: three annotators, none an author, none told the hypothesis, all
blinded to framing. Audit 1: one annotator, unblinded — identity not verified at time of
writing. Audit 2: two annotators, unblinded, instructions unrecoverable.

**Uses.** Studying evaluator and protocol variance in LLM-judged safety evaluation;
calibrating harm graders against human labels; testing aggregation and adjudication
schemes. **Not** suitable as a harm benchmark or as training data for a harm classifier —
the labels disagree by design and the disagreement is the content.

**Distribution.** Prompts, labels and code CC BY 4.0; generations under research-use
terms. Non-archival workshop release; no personal data; all content synthetic.
