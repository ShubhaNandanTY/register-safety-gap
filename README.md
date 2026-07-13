# Hidden in the Hedge: Unvalidated LLM Graders Understate the Register Safety Gap

Data, code, and human labels for the paper *"Hidden in the Hedge: Unvalidated LLM
Graders Understate the Register Safety Gap in Instruction-Tuned Language Models."*

## TL;DR

Instruction-tuned LLMs refuse **casually-phrased** harmful requests less often than
identical **formally-phrased** ones. An off-the-shelf LLM harm grader reports this as a
near-zero safety gap (0.03) — but when we validate that grader against 200 human
labels, it turns out to be hiding most casually-framed harm in its "hedge" category.
The **human-corrected register harm gap is 0.26 (95% CI [0.19, 0.32])**: casual framing
raises the actionable-harm rate from 8.5% to 34.3%. The core lesson is methodological:
**LLM safety graders must be validated against human labels before their verdicts are
trusted** — used unchecked, one here would have reported no safety problem where a
substantial one exists.

## Key numbers (reproducible via `scripts/05_analysis.py`)

| Measure | Register gap |
|---|---|
| Keyword refusal metric (corpus) | 0.26 (up to 0.54 for Qwen-7B) |
| Off-the-shelf LLM harm grader (naive) | 0.03 |
| **Human-validated harm grader (corrected)** | **0.26, 95% CI [0.19, 0.32]** |

## Repository structure

```
data/
  register_pairs_300.csv     300 paired prompts (25 intents x 12 templates);
                             columns: pair_id, category, intent, eval_framing, deploy_framing
generations/
  qwen_generations.csv       raw responses + keyword refusal labels, 4 Qwen2.5 models
  gemma_generations.csv      raw responses + keyword refusal labels, 2 Gemma-2 models
human_labels/
  grader_audit_60.csv        60-item stratified human validation of the 3-way grader
  hedge_audit_200.csv        200-item human re-labeling of grader "hedge" responses
                             (the audit that yields the corrected harm gap)
scripts/
  03_harm_grader.py          re-scores generations with the LLM harm grader (GPU)
  04_build_hedge_audit.py    builds the hedge-audit sample for human labeling
  05_analysis.py             reproduces all headline numbers from released data (CPU)
figures/                     paper figures
paper/                       manuscript (PDF + LaTeX source)
```

## Reproduce the results

```bash
pip install -r requirements.txt
python scripts/05_analysis.py
```

This regenerates the refusal-based RVG table (with clustered-bootstrap CIs), the naive
vs. corrected harm gap, and the grader-validation numbers, directly from the released
data and human labels.

To re-run the LLM harm grader from scratch (requires a GPU and a Hugging Face token):

```bash
python scripts/03_harm_grader.py   # writes judge_v2_rescored.csv
```

## Method in brief

1. **Paired prompts.** Each of 25 harmful intents is written twice — a formal,
   benchmark-style *eval* framing and a casual, first-person *deploy* framing — holding
   intent constant and varying only register. No explicit "this is a test" cues.
2. **Three measurements.** (a) keyword refusal rate; (b) an LLM grader labeling each
   response REFUSAL / HEDGE / HARMFUL; (c) the same, corrected using human validation of
   the grader's HEDGE calls.
3. **Human validation.** 60 stratified labels validate the grader overall; 200 labels of
   its HEDGE responses reveal it misclassifies actionable harm as hedging, more so for
   casual prompts (49% vs 20%), which is what collapses the naive harm gap.

## Models

Qwen2.5-Instruct (0.5B / 1.5B / 3B / 7B) and Gemma-2-it (2B / 9B), all run in 4-bit on a
single consumer GPU.

## Limitations

The 300 prompts derive from only 25 underlying intents; the clustered bootstrap accounts
for the statistical dependence, but broad generalization is future work. Results cover
two open-weight families up to 9B and a single harm grader. See the paper's Limitations
section for details.

## Citation

```bibtex
@misc{registergap2026,
  title  = {Hidden in the Hedge: Unvalidated LLM Graders Understate the Register
            Safety Gap in Instruction-Tuned Language Models},
  author = {Shubhanandan T Y},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Code: MIT (see LICENSE). The dataset contains prompts describing harmful intents in the
abstract, released solely for safety-evaluation research; model generations are included
for reproducibility with operationally harmful detail minimized. Please use responsibly.
