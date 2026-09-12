# Register Safety Gap — validation-protocol variance in LLM safety grading

Data, code and audits for a study of what happens when you validate one LLM harm
grader against human labels three separate times.

**Headline result.** The three validations disagree with each other by more than any of
them disagrees with the grader nobody checked. On one corpus, under one rubric, the
implied harm gaps run from **−0.036 to +0.258**, a span of 0.310, which is wider than
the 0.035 effect the audits existed to measure.

## Read this before using the numbers

An earlier version of this work reported a corrected harm gap of **0.26** and concluded
that the grader was concealing seven-eighths of a real safety gap. **That conclusion is
withdrawn.** It came from a single unblinded annotator working from half-length
excerpts under a rubric with no marginal-uplift test. A later pre-registered,
framing-blinded audit with three annotators returned **0.035**, close to the grader it
was meant to discredit. If you have cited or forked the earlier version, use this one.

## What is here

| Path | Contents |
|---|---|
| `prompts/` | 300 matched prompt pairs over 25 harmful intents, 10 categories |
| `generations/` | 3,600 responses per run. Original 200-token, a 200-token rerun, and an uncensored 1024-token regeneration |
| `grader/` | Primary grader labels, Llama Guard labels, and the 5-budget curve |
| `audits/` | All three human audits, long format, one row per annotator per item |
| `code/` | Verification and dataset-build notebooks, analysis and generation scripts |
| `meta/` | Environment stamps, manifest, TRACE template, verification report |
| `docs/` | Dataset specification and the pre-submission review |
| `paper/` | The submitted PDF |

## Start here

`code/01_verify_paper_numbers.ipynb` recomputes 36 statistics from the released CSVs
and prints a PASS/MISMATCH table against the values in the paper. It runs on CPU in
about two minutes and needs only pandas, numpy, scikit-learn and krippendorff. For a
paper about measurement reliability, being able to re-derive the numbers yourself is the
point.

`code/02_build_release_dataset.ipynb` regenerates this release from the raw archive.

## Joining the files

The primary key is `response_id`, formed as `{model_slug}__{pair_id}__{framing}`, for
example `qwen2.5-7b__pair_0143__user`. It is unique across the 3,600 responses and
stable across all three generation runs.

**Do not join the audits on `item_id`.** The original files reused identifiers across
audits, so `IT0001`–`IT0442` from audit 2 and `IT0000`–`IT0238` from audit 3 share 238
identifiers that denote different items. Joining on them produces plausible-looking
garbage. The released files are renamespaced to `A1-`, `A2-` and `A3-`.

## Two columns that carry the finding

`display_window_chars` records how much of each response the annotator actually saw:
500 characters in audit 1, 700 in audit 2, the full text in audit 3. This appeared in no
written protocol and was recovered by inspecting the label files.

`prompt_variant` distinguishes two grader invocations that are not interchangeable. The
original truncated responses at 1,200 characters, appended a `Label:` cue and ran
unbatched. The budget-curve grader did none of these. At the same nominal budget they
report 0.033 and 0.025.

`link_status` is `unique`, `ambiguous_k` or `unmatched`. Because annotators saw
truncated text, some audit rows cannot be traced to exactly one corpus row. Those are
marked rather than guessed at.

## Known limits

- Audit 2's annotator instructions could not be recovered, so chance disagreement cannot
  be separated from divergent operationalisation.
- Neither generation nor grading is bit-reproducible. Regenerating at the original cap
  reproduces 19.4% of responses exactly across six models, and 0% for gemma-2-2b.
- Audit 3 has 90 triple-labelled items and 149 singly labelled ones, so α = 0.856
  describes 38% of that sample.
- The original keyword refusal scorer was never recovered. The budget curve runs on a
  reimplementation reporting +0.018 higher on byte-identical text.
- The framing contrast bundles formality, perspective, motive, length and lexical
  diversity. It does not isolate register.

## Licence

Prompts, labels, code and metadata are CC BY 4.0. The three files in `generations/` are
model outputs to harmful prompts and are covered by `TERMS.md`.

## Citation

Submitted to the TAE (Trust-AI-Eval) workshop at NeurIPS 2026. Non-archival.
