#!/usr/bin/env python3
"""Evaluate a (fine-tuned) cross-encoder on TREC DL 2019 (MSMARCO passage).

Local adaptation of the provided evaluation notebook.

Outputs:
- prints NDCG@10, Recall@100, MAP@1000 (matches notebook)
- writes a TREC run file (qid Q0 docid rank score STANDARD)

Assumptions:
- model_dir points to a sentence-transformers CrossEncoder directory (output_path from fine_tune.py)
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
import requests
import tqdm
from sentence_transformers.cross_encoder import CrossEncoder


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_if_missing(url: str, path: Path, chunk_size: int = 1024 * 1024):
    """Reliable downloader with .part file."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    logging.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm.tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.replace(path)


def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    ap.add_argument("--output_run", required=True, help="Output .run file path")
    ap.add_argument(
        "--data_dir",
        default="models/datasets/trec_dl_2019",
        help="Dataset directory (default: models/datasets/trec_dl_2019)",
    )
    ap.add_argument(
        "--download_mode",
        default="auto",
        choices=["auto", "never"],
        help="auto=download missing files; never=error if required files are missing",
    )
    ap.add_argument(
        "--queries_path",
        default=None,
        help="Optional path to msmarco-test2019-queries.tsv(.gz). If omitted, uses data_dir copy.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_run = Path(args.output_run)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s", model_dir)
    model = CrossEncoder(str(model_dir))

    # Files
    qrels = data_dir / "2019qrels-pass.txt"
    queries_tsv_gz = Path(args.queries_path) if args.queries_path else (data_dir / "msmarco-test2019-queries.tsv.gz")
    top1000 = data_dir / "msmarco-passagetest2019-top1000.tsv.gz"

    download_if_missing("https://trec.nist.gov/data/deep/2019qrels-pass.txt", qrels)
    if args.download_mode == "never" and not qrels.exists():
        raise FileNotFoundError(f"Missing {qrels}. Download it first or run with --download_mode auto")
    if args.download_mode == "auto":
        download_if_missing("https://trec.nist.gov/data/deep/2019qrels-pass.txt", qrels)

    if args.queries_path is None and args.download_mode == "auto":
        download_if_missing(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
            queries_tsv_gz,
        )
    elif args.queries_path is None and args.download_mode == "never" and not queries_tsv_gz.exists():
        raise FileNotFoundError(
            f"Missing {queries_tsv_gz}. Download it first or run with --download_mode auto"
        )
    if args.download_mode == "auto":
        download_if_missing(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
            top1000,
        )
    elif not top1000.exists():
        raise FileNotFoundError(
            f"Missing {top1000}. Download it first or run with --download_mode auto"
        )

    # Load qrels
    relevant_docs = defaultdict(lambda: defaultdict(int))
    with open(qrels) as f:
        for line in f:
            qid, _, pid, score = line.strip().split()
            relevant_docs[qid][pid] = int(score)

    # Load queries
    queries = {}
    open_fn = gzip.open if str(queries_tsv_gz).endswith(".gz") else open
    with open_fn(queries_tsv_gz, "rt") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    # Load top-1000 candidates
    # Format: qid \t pid \t query \t passage (in the MSMARCO file)
    # We'll read only qid->[(pid, passage)]
    candidates = defaultdict(list)
    with gzip.open(top1000, "rt") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) == 4:
                qid, pid, _query, passage = cols
            else:
                # be defensive
                qid, pid = cols[0], cols[1]
                passage = cols[-1]
            candidates[qid].append((pid, passage))

    # Score
    run = defaultdict(dict)
    ranking_lines = []

    for qid, qtext in queries.items():
        if qid not in candidates:
            continue
        pairs = [[qtext, passage] for _, passage in candidates[qid]]
        pids = [pid for pid, _ in candidates[qid]]

        scores = model.predict(pairs, batch_size=args.batch_size)
        for pid, s in zip(pids, scores):
            run[qid][pid] = float(s)

        # create sorted run lines now
        sorted_pids = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (pid, score) in enumerate(sorted_pids, start=1):
            ranking_lines.append(f"{qid} Q0 {pid} {rank} {score} STANDARD")

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {"ndcg_cut.10", "recall_100", "map_cut.1000"})
    scores = evaluator.evaluate(run)

    relevant_qid = [qid for qid in scores.keys()]
    print("Queries:", len(relevant_qid))
    print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()]) * 100))
    print("Recall@100: {:.2f}".format(np.mean([ele["recall_100"] for ele in scores.values()]) * 100))
    print("MAP@1000: {:.2f}".format(np.mean([ele["map_cut_1000"] for ele in scores.values()]) * 100))

    out_run.parent.mkdir(parents=True, exist_ok=True)
    out_run.write_text("\n".join(ranking_lines), encoding="utf-8")
    logging.info("Wrote run file: %s", out_run)


if __name__ == "__main__":
    main()
