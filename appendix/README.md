# Appendix — Supplementary Analyses

Supplementary experiments that position and stress-test the main four-step
pipeline. Both reuse the shared data in [`../data/`](../data/README.md).

1. **Single-prompt baseline** (paper Table 5) — performs all four steps in a
   single LLM call / single function call, as a baseline against the sequential
   four-step pipeline (Prompt Chaining).
2. **American-persona ablation** (paper Appendix A, Table A.1) — the same
   four-step pipeline applied to the same Japanese conversations, but with the
   system-prompt persona changed from *"Japan Conversation Analyst"* to
   *"American Conversation Analyst"* (a prompt-framing ablation; CoT baseline in
   all steps).

## Structure

```
appendix/
├── code/
│   ├── single_prompt_experiment.py        # Single-prompt baseline driver (CLI)
│   ├── single_prompt_reevaluation.ipynb   # Re-score single-prompt vs 4-step
│   ├── eval_unified_single_vs_4step.ipynb # Unified scorer -> Table 5 CSVs
│   ├── american_pipeline_worker.py        # Standalone 4-step worker (American persona)
│   ├── american_01_prompts.ipynb          # American-persona prompt library
│   ├── american_02_run_pipeline.ipynb     # Run the American-persona pipeline
│   └── american_03_evaluation.ipynb       # American vs Japanese evaluation -> Table A.1 CSVs
└── results/
    ├── comparison/        # unified_eval_*.csv (backs paper Table 5)
    ├── american/          # american_eval_*.csv (backs paper Table A.1)
    └── confusion_matrix/  # Confusion matrices for the American-persona runs
```

The per-technique evaluation (paper Tables 6–7) and the Japanese-prompt
confusion matrices (paper Figures 7–9) live in the main
[`../evaluation/`](../evaluation/) folder.

## How the pieces fit together

- **`single_prompt_experiment.py`** runs the same four analysis steps in a
  single prompt + single function call
  (`python appendix/code/single_prompt_experiment.py --model gpt5_medium`).
- **`eval_unified_single_vs_4step.ipynb`** scores the four-step and
  single-prompt outputs with one shared set of metric functions and writes the
  comparison tables (Table 5) to `appendix/results/comparison/`.
- **`american_pipeline_worker.py`** is a standalone version of the four-step
  pipeline; `american_02_run_pipeline.ipynb` launches it as parallel
  subprocesses, one per conversation.

### Raw outputs (not committed)

Raw per-iteration model outputs are git-ignored. The notebooks expect them at:

| Source | Path |
|--------|------|
| Four-step pipeline runs  | `results/<model>/raw/` (produced by the main pipeline) |
| Single-prompt runs       | `appendix/results/single_prompt/<model>/raw/` |
| American-persona runs    | `appendix/results/american/<model>/raw/` |

The score tables and figures shipped under `appendix/results/` were generated
from those raw outputs.

## Running

Run notebooks **from the repository root** (not from `appendix/`) so that
`data/...`, `results/...`, and `appendix/...` paths all resolve.

### Note on the American persona

The "American" experiment does **not** use a separate dataset — it applies an
American-framed system prompt to the same anonymized Japanese conversations.
