# PhysFrame: data and code

Anonymous release accompanying the submission "When Validating the Judge Changes the Verdict".

| Folder | Contents |
|---|---|
| `data/` | 300 problems (`problems.jsonl`), answer key with tolerances and trap answers, 1,500 prompts in five framings |
| `generations/` | 10,500 responses from seven models at 1,024 tokens; 300 Muse Glimmer 30B responses at 4,096 tokens |
| `scores/` | code-reference scores (strict and lenient, extracted value, relative error, trap match) |
| `judge/` | phi-4 labels: main run, framing-hidden run (Gemma-2-9B, Qwen3-Omni-30B), Muse Glimmer |
| `human_check/` | item key for the 120 audited responses, annotator A1 labels (120), annotator A2 (37), codebook as used |
| `code/` | generation (03), scorer (04), judge (05), audit sheets (06, 08), analysis (07) |
| `analysis/` | `extra_checks.py` and `framing.py` with their outputs; `responses.csv` is the per-response table they read |
| `paper_figures/` | `make_figures.py` and the figure PDFs |

Rerun the robustness checks:

    cd analysis
    python3 extra_checks.py responses.csv
    python3 framing.py        # template-level intervals for the framing effects

Six judge outputs could not be parsed (label `UNPARSED`) and are excluded from judge analyses.
In the human check, annotator identities are withheld.
