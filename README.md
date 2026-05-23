# IR_Assign (Fully Script-Based Pipeline)

This repo contains a reproducible, **.py script only** workflow for the IR final assignment:

- **Task 1**: fine-tune 3 cross-encoder rerankers on MSMARCO and evaluate on **TREC DL'19** (NDCG@10 / Recall@100 / MAP@1000)
- **Task 2-3**: fuse run files using **ranx** and evaluate ensembles
- **Task 4-5 (optional)**: LLM query expansion (Ollama Zephyr) with and without pseudo-relevance feedback (PRF)

## Environment

We use `uv` (recommended) but plain `python` also works.

```bash
uv sync
```

Quick sanity check:
```bash
uv run scripts/verify_env.py
```

## Data layout

Datasets are stored under:
- `models/datasets/msmarco/`
- `models/datasets/trec_dl_2019/`

Models are stored under `models/<model_name>/...` and run files under `runs/`.

## Task 1: Fine-tune (1 hour) + Evaluate

### Fine-tune
Example (MiniLM):
```bash
uv run scripts/fine_tune.py \
  --model_name cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --output_dir models/minilm-l2 \
  --device auto \
  --download_mode auto \
  --corpus_mode auto
```

### Evaluate (TREC DL'19)
You must provide:
- `--model_dir` = a fine-tuned model folder
- `--output_run` = where to write the `.run`

Example (MiniLM):
```bash
mkdir -p runs/trec_dl19

uv run scripts/evaluate.py \
  --model_dir models/minilm-l2/<YOUR_CHECKPOINT_DIR> \
  --output_run runs/trec_dl19/minilm_l2.run \
  --device auto \
  --download_mode auto
```

## Task 2 + 3: Fusion (ranx)

Once you have the three Task 1 run files:

```bash
uv run scripts/fuse_and_eval.py \
  --qrels models/datasets/trec_dl_2019/2019qrels-pass.txt \
  --runs runs/trec_dl19/minilm_l2.run runs/trec_dl19/tinybert_l2.run runs/trec_dl19/distilroberta.run \
  --methods combSUM combMNZ combANZ borda rrf \
  --out_dir runs/fusion_results
```

Outputs:
- `runs/fusion_results/task2_fusions.csv`
- `runs/fusion_results/task3_pairs.csv`

## Task 4 + 5 (optional): Query Expansion (Ollama)

Prereq: install + run Ollama, and pull Zephyr:
```bash
ollama serve
ollama pull zephyr-7b-beta
```

Run experiments:
```bash
uv run scripts/run_query_expansion_experiments.py \
  --base_run runs/trec_dl19/tinybert_l2.run \
  --out_dir runs/query_expansion \
  --ollama_model zephyr-7b-beta \
  --prf_k 3
```

## Report

A ready-to-compile LaTeX report draft is in `report/report.tex`.

```bash
cd report
latexmk -pdf report.tex
```
