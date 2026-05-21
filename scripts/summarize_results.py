#!/usr/bin/env python3
"""Summarize evaluation results into markdown tables for the report.

This script is intentionally simple: it reads one or more CSV files (e.g.
from scripts/fuse_and_eval.py) and prints a markdown table.

Usage examples:
  python3 scripts/summarize_results.py runs/fusion_results/base_models.csv
  python3 scripts/summarize_results.py runs/fusion_results/task2_fusions.csv --sort ndcg@10

It also supports a lightweight "Task 1" mode where you manually provide model
names and metric numbers, and it writes a ready-to-paste table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", help="One or more CSVs to render as markdown")
    ap.add_argument("--sort", default=None, help="Column to sort by (descending)")
    ap.add_argument("--head", type=int, default=50, help="Max rows to print")
    ap.add_argument("--task1_template", action="store_true", help="Print Task 1 table template")
    args = ap.parse_args()

    if args.task1_template:
        df = pd.DataFrame(
            [
                {
                    "Model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
                    "Steps@1h": "(fill)",
                    "NDCG@10": "(fill)",
                    "Recall@100": "(fill)",
                    "MAP@1000": "(fill)",
                },
                {
                    "Model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                    "Steps@1h": "(fill)",
                    "NDCG@10": "(fill)",
                    "Recall@100": "(fill)",
                    "MAP@1000": "(fill)",
                },
                {
                    "Model": "distilroberta-base",
                    "Steps@1h": "(fill)",
                    "NDCG@10": "(fill)",
                    "Recall@100": "(fill)",
                    "MAP@1000": "(fill)",
                },
            ]
        )
        print(df.to_markdown(index=False))
        return

    if not args.csv:
        raise SystemExit("Provide at least one CSV path, or use --task1_template")

    for csv_path in args.csv:
        p = Path(csv_path)
        df = pd.read_csv(p)
        if args.sort and args.sort in df.columns:
            df = df.sort_values(args.sort, ascending=False)
        df = df.head(args.head)
        print(f"\n## {p.name}\n")
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
