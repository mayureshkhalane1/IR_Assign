#!/usr/bin/env python3
"""Fuse multiple TREC run files with ranx and evaluate.

Used for:
- Task 2: fuse 3 runs with 5 different fusion methods
- Task 3: use best method from Task 2 on each pair of runs

Input run format: TREC run file (qid Q0 docid rank score tag)
Input qrels format: TREC qrels (qid 0 docid relevance)

This script prints a markdown-ready table and also writes CSVs.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from ranx import Qrels, Run, fuse, evaluate


DEFAULT_METRICS = ["ndcg@10", "recall@100", "map@1000"]


def load_runs(run_paths: list[Path]) -> dict[str, Run]:
    runs = {}
    for p in run_paths:
        runs[p.stem] = Run.from_file(str(p))
    return runs


def eval_run(qrels: Qrels, run: Run, metrics: list[str]) -> dict:
    res = evaluate(qrels, run, metrics)
    # ranx returns e.g. {'ndcg@10': 0.123}
    return {m: float(res[m]) for m in metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True, help="Path to qrels file")
    ap.add_argument("--runs", nargs="+", required=True, help="Run files (2+)")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["rrf", "comb_sum", "comb_mnz", "borda", "comb_anz"],
        help="Fusion methods to test (Task 2)",
    )
    ap.add_argument("--out_dir", default="runs/fusion_results", help="Output directory")
    ap.add_argument("--best_method", default=None, help="Override best method for Task 3")
    args = ap.parse_args()

    qrels = Qrels.from_file(args.qrels)
    run_paths = [Path(p) for p in args.runs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(run_paths)
    metrics = DEFAULT_METRICS

    # Baselines
    base_rows = []
    for name, r in runs.items():
        row = {"system": name}
        row.update(eval_run(qrels, r, metrics))
        base_rows.append(row)
    base_df = pd.DataFrame(base_rows).sort_values("ndcg@10", ascending=False)
    base_df.to_csv(out_dir / "base_models.csv", index=False)

    # Task 2: fuse all runs for each method
    fuse_rows = []
    for m in args.methods:
        fused = fuse(list(runs.values()), method=m)
        row = {"system": f"fuse:{m}"}
        row.update(eval_run(qrels, fused, metrics))
        fuse_rows.append(row)
    fuse_df = pd.DataFrame(fuse_rows).sort_values("ndcg@10", ascending=False)
    fuse_df.to_csv(out_dir / "task2_fusions.csv", index=False)

    best_method = args.best_method or fuse_df.iloc[0]["system"].split(":", 1)[1]

    # Task 3: best method on all 2-run combinations
    pair_rows = []
    for a, b in combinations(runs.keys(), 2):
        fused = fuse([runs[a], runs[b]], method=best_method)
        row = {"system": f"pair:{a}+{b} ({best_method})"}
        row.update(eval_run(qrels, fused, metrics))
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows).sort_values("ndcg@10", ascending=False)
    pair_df.to_csv(out_dir / "task3_pairs.csv", index=False)

    print("\n## Base models")
    print(base_df.to_markdown(index=False))
    print("\n## Task 2: Fusions (all 3 runs)")
    print(fuse_df.to_markdown(index=False))
    print(f"\nBest fusion method by NDCG@10: {best_method}")
    print("\n## Task 3: Best fusion on 2-run combinations")
    print(pair_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
