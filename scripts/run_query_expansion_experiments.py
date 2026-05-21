#!/usr/bin/env python3
"""End-to-end Task 4/5 query expansion experiments.

Pipeline:
1) Generate expanded queries with Ollama Zephyr (CoT and CoT/PRF)
2) Evaluate with scripts/evaluate.py using --queries_path
3) Save run files and a small CSV summary + print markdown table

This avoids notebooks and is designed to be reproducible locally.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd


METRIC_PATTERNS = {
    "ndcg@10": re.compile(r"NDCG@10:\s*([0-9.]+)"),
    "recall@100": re.compile(r"Recall@100:\s*([0-9.]+)"),
    "map@1000": re.compile(r"MAP@1000:\s*([0-9.]+)"),
}


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout


def parse_metrics(stdout: str) -> dict[str, float]:
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(stdout)
        if not m:
            raise ValueError(f"Could not parse {k} from evaluate.py output")
        out[k] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Fine-tuned cross-encoder directory")
    ap.add_argument("--ollama_model", default="zephyr-7b-beta")
    ap.add_argument("--out_dir", default="runs/query_expansion")
    ap.add_argument("--prf_k", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries_cot = out_dir / "queries_cot.tsv"
    queries_cot_prf = out_dir / "queries_cot_prf.tsv"

    # 1) Expand queries
    run(
        [
            "python3",
            "scripts/expand_queries_ollama.py",
            "--model",
            args.ollama_model,
            "--mode",
            "cot",
            "--max_words",
            str(args.max_words),
            "--out",
            str(queries_cot),
        ]
    )
    run(
        [
            "python3",
            "scripts/expand_queries_ollama.py",
            "--model",
            args.ollama_model,
            "--mode",
            "cot_prf",
            "--prf_k",
            str(args.prf_k),
            "--max_words",
            str(args.max_words),
            "--out",
            str(queries_cot_prf),
        ]
    )

    # 2) Evaluate
    rows = []
    for label, qpath in [("cot", queries_cot), ("cot_prf", queries_cot_prf)]:
        run_path = out_dir / f"expanded_{label}.run"
        stdout = run(
            [
                "python3",
                "scripts/evaluate.py",
                "--model_dir",
                args.model_dir,
                "--output_run",
                str(run_path),
                "--queries_path",
                str(qpath),
                "--batch_size",
                str(args.batch_size),
            ]
        )
        metrics = parse_metrics(stdout)
        rows.append({"system": label, **metrics})

    df = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    df.to_csv(out_dir / "query_expansion_results.csv", index=False)

    print("\n## Query expansion results")
    print(df.to_markdown(index=False))
    print(f"\nSaved: {out_dir / 'query_expansion_results.csv'}")


if __name__ == "__main__":
    main()
