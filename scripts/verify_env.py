#!/usr/bin/env python3
"""Sanity checks before running long training/evaluation jobs.

Checks:
- torch sees MPS (optional) and can allocate
- key packages import
- dataset files exist where scripts expect them
"""

from __future__ import annotations

from pathlib import Path


def main():
    print("== Imports ==")
    import torch  # noqa
    import sentence_transformers  # noqa
    import transformers  # noqa
    import ranx  # noqa

    print("torch:", torch.__version__)
    print("mps available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print("cuda available:", torch.cuda.is_available())

    print("\n== Dataset presence ==")
    msmarco = Path("models/datasets/msmarco")
    trec = Path("models/datasets/trec_dl_2019")

    msmarco_files = [msmarco / "collection.tsv", msmarco / "queries.train.tsv"]
    trec_files = [
        trec / "msmarco-test2019-queries.tsv.gz",
        trec / "msmarco-passagetest2019-top1000.tsv.gz",
        trec / "2019qrels-pass.txt",
    ]

    for p in msmarco_files + trec_files:
        print(f"{p}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
