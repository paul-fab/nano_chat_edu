# Log-Prob MCQ Evaluation

This folder adds a base-model-friendly evaluation path for pedagogy MCQs.

## What it does
- Scores each answer option with normalized token log-probability:
  - `score(option) = mean(log p(tokens(option) | prompt))`
- Picks highest-scoring option as prediction.
- Supports few-shot prompting (default `k=3`) without requiring instruction-following generation.

## Files
- `eval/logprob_mcq_eval.py`: Single-model evaluator.
- `eval/logprob_mcq_eval_nanochat.py`: Evaluator for native nanochat checkpoints.
- `eval/run_public_comparison.py`: Multi-model comparison runner.
- `eval/combine_cdpk.py`: Utility to combine per-category CDPK files.
- `eval/public_model_sets.json`: Curated public-model presets.

## Input format
Evaluator accepts `.csv`, `.json`, or `.jsonl`.

Required fields per row/example:
- Question text in one of: `question`, `prompt`, `stem`, `item`, `query`
- Options as either:
  - `options` (array), or
  - letter columns `A`..`G`, or
  - numbered columns like `option1`, `option2`, ...
- Gold answer in one of: `answer`, `label`, `correct`, `gold`, `target`, `correct_answer`
  - letter (`A`..`G`) or index (`0..N-1` or `1..N`)

## Install
```bash
pip install torch transformers
```

For nanochat checkpoint eval, run from a nanochat environment where `nanochat` is importable (or set `PYTHONPATH`).

## Run one model
```bash
python eval/logprob_mcq_eval.py \
  --model gpt2 \
  --eval-file /path/to/send_test.csv \
  --dev-file /path/to/cdpk_dev.csv \
  --fewshot-k 3 \
  --output-json eval/results/gpt2.json
```

## Run public comparison set
```bash
python eval/run_public_comparison.py \
  --eval-file /path/to/send_test.csv \
  --dev-file /path/to/cdpk_dev.csv \
  --fewshot-k 3 \
  --output-csv eval/results/public_comparison.csv
```

## Run native nanochat checkpoint
```bash
PYTHONPATH=~/nanochat python eval/logprob_mcq_eval_nanochat.py \
  --source base \
  --model-tag d20 \
  --step 3000 \
  --eval-file /path/to/CDPK_all_test.csv \
  --dev-file /path/to/CDPK_all_dev.csv \
  --fewshot-k 3 \
  --device cuda \
  --output-json eval/results/cdpk_d20_step3000.json
```

`--step` is optional; if omitted the script loads the latest step for the tag.

## CDPK helper
If CDPK data is split by category, combine it first:
```bash
python eval/combine_cdpk.py
```

## Add your own model checkpoint
If your checkpoint is HF-compatible (local directory or model id), append it with:
```bash
python eval/run_public_comparison.py \
  --eval-file /path/to/send_test.csv \
  --dev-file /path/to/cdpk_dev.csv \
  --output-csv eval/results/public_plus_local.csv \
  --extra-model my_edu_model=/path/to/local_hf_model
```

## Notes
- Some models are gated on HF (for example `meta-llama/Llama-3.2-1B`) and require accepted licenses.
- For larger models, use GPU (`--device cuda`) and consider `--dtype bf16` where supported.
