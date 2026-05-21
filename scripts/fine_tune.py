#!/usr/bin/env python3
"""Fine-tune a cross-encoder on MSMARCO passage re-ranking.

This is a local (non-Colab) adaptation of the provided notebook.

Design goals:
- deterministic paths (./data, ./models)
- no notebook magics
- prints a final summary block to copy into the report

IMPORTANT: The assignment requires ~1 hour training per model.
This script does not hard-stop after 60 minutes (sentence-transformers does not
expose a clean time-based stopper here). Instead, run it and stop it manually
at ~60 minutes (Ctrl+C). The training loop prints progress; record the last
"Step" shown (global step).
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path

import tqdm
import requests
import time
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _acquire_lock(lock_path: Path, stale_seconds: int = 6 * 3600) -> None:
    """Tiny cross-process lock (atomic create).

    You are running 3 trainings in parallel; they must not download/extract the
    same MSMARCO files at the same time.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(1)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def download_file(url: str, path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Robust downloader safe for parallel processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    _acquire_lock(lock)
    try:
        # Another process might have finished while we waited.
        if path.exists() and path.stat().st_size > 0:
            return

        tmp = path.with_suffix(path.suffix + ".part")
        tmp.unlink(missing_ok=True)

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
                f.flush()
                os.fsync(f.fileno())

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download failed (no data written): {tmp}")

        tmp.replace(path)
    finally:
        _release_lock(lock)


def ensure_msmarco_queries(data_dir: Path) -> Path:
    """Download + extract MSMARCO queries tar (contains queries.train.tsv)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "queries.tar.gz"
    if not tar_path.exists():
        download_file(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            tar_path,
        )
    # The tar contains queries.train.tsv (and others) in root
    # Serialize extraction for parallel runs.
    extract_lock = data_dir / "queries.extract.lock"
    _acquire_lock(extract_lock)
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            wanted = {"queries.train.tsv"}
            names = {m.name for m in members}
            if wanted.issubset(names) and not (data_dir / "queries.train.tsv").exists():
                logging.info("Extracting queries.train.tsv")
                tar.extractall(path=data_dir)
    finally:
        _release_lock(extract_lock)
    qpath = data_dir / "queries.train.tsv"
    if not qpath.exists():
        raise FileNotFoundError(f"queries.train.tsv not found in {data_dir}.")
    return qpath


def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        required=True,
        help="HF model name, e.g. cross-encoder/ms-marco-MiniLM-L-2-v2",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write fine-tuned model (will be created)",
    )
    ap.add_argument(
        "--data_dir",
        default="models/datasets/msmarco",
        help="Dataset directory (default: models/datasets/msmarco)",
    )
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--evaluation_steps", type=int, default=1000)
    ap.add_argument("--warmup_steps", type=int, default=5000)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--pos_neg_ratio", type=int, default=4)
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=5_000_000,
        help="Upper bound on number of training samples loaded (time/memory control)",
    )
    ap.add_argument("--num_dev_queries", type=int, default=200)
    ap.add_argument("--num_max_dev_negatives", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--use_amp",
        default="auto",
        choices=["auto", "true", "false"],
        help="Mixed precision. 'auto' enables only on CUDA. MPS/CPU will disable.",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model: %s", args.model_name)
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)

    # AMP: accelerate's fp16 mixed precision is CUDA-only. On Apple MPS it raises.
    if args.use_amp == "auto":
        use_amp = torch.cuda.is_available()
    else:
        use_amp = args.use_amp == "true"

    if not torch.cuda.is_available() and use_amp:
        logging.warning("Forcing use_amp=False because CUDA is not available.")
        use_amp = False

    # ---------- Download / load MSMARCO corpus ----------
    corpus: dict[str, str] = {}
    collection_tsv = data_dir / "collection.tsv"
    if not collection_tsv.exists():
        tar_path = data_dir / "collection.tar.gz"
        if not tar_path.exists():
            logging.info("Downloading collection.tar.gz")
            download_file(
                "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
                tar_path,
            )
        extract_lock = data_dir / "collection.extract.lock"
        _acquire_lock(extract_lock)
        try:
            if not collection_tsv.exists():
                logging.info("Extracting collection.tsv (can take a while)")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=data_dir)
        finally:
            _release_lock(extract_lock)

    logging.info("Reading collection.tsv")
    with open(collection_tsv, "r", encoding="utf-8") as f:
        for line in f:
            pid, passage = line.rstrip("\n").split("\t", 1)
            corpus[pid] = passage

    # Integrity check: MSMARCO passage collection has ~8.8M passages.
    if len(corpus) < 8_000_000:
        raise RuntimeError(
            f"collection.tsv seems incomplete (loaded {len(corpus)} passages). "
            f"Delete {collection_tsv} and {data_dir / 'collection.tar.gz'} then re-run."
        )

    # ---------- Queries ----------
    qpath = ensure_msmarco_queries(data_dir)
    queries: dict[str, str] = {}
    logging.info("Reading %s", qpath.name)
    with open(qpath, "r", encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\n").split("\t", 1)
            queries[qid] = query

    # ---------- Build dev samples from train-eval triples ----------
    train_samples: list[InputExample] = []
    dev_samples: dict[str, dict] = {}

    train_eval_path = data_dir / "msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz"
    if not train_eval_path.exists():
        logging.info("Downloading %s", train_eval_path.name)
        download_file(
            "https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz",
            train_eval_path,
        )

    logging.info("Building dev set (%d queries)", args.num_dev_queries)
    missing_in_corpus = 0
    with gzip.open(train_eval_path, "rt") as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split()

            if qid not in dev_samples and len(dev_samples) < args.num_dev_queries:
                dev_samples[qid] = {
                    "query": queries[qid],
                    "positive": set(),
                    "negative": set(),
                }

            if qid in dev_samples:
                if pos_id in corpus:
                    dev_samples[qid]["positive"].add(corpus[pos_id])
                else:
                    missing_in_corpus += 1

                if len(dev_samples[qid]["negative"]) < args.num_max_dev_negatives:
                    if neg_id in corpus:
                        dev_samples[qid]["negative"].add(corpus[neg_id])
                    else:
                        missing_in_corpus += 1

    if missing_in_corpus:
        logging.warning(
            "Missing %d passage ids in corpus while building dev set (skipped).",
            missing_in_corpus,
        )

    # CERerankingEvaluator expects lists (it does `positive + negative`).
    # We used sets during construction to deduplicate; convert now.
    for _qid in list(dev_samples.keys()):
        dev_samples[_qid]["positive"] = list(dev_samples[_qid]["positive"])
        dev_samples[_qid]["negative"] = list(dev_samples[_qid]["negative"])

    # ---------- Training samples ----------
    train_path = data_dir / "msmarco-qidpidtriples.rnd-shuf.train.tsv.gz"
    if not train_path.exists():
        logging.info("Downloading %s", train_path.name)
        download_file(
            "https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz",
            train_path,
        )

    logging.info("Streaming training triples (max_train_samples=%d)", args.max_train_samples)
    cnt = 0
    missing_train = 0
    with gzip.open(train_path, "rt") as f:
        for line in tqdm.tqdm(f, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()

            if qid in dev_samples:
                continue

            query = queries[qid]
            if (cnt % (args.pos_neg_ratio + 1)) == 0:
                if pos_id not in corpus:
                    missing_train += 1
                    continue
                passage = corpus[pos_id]
                label = 1
            else:
                if neg_id not in corpus:
                    missing_train += 1
                    continue
                passage = corpus[neg_id]
                label = 0

            train_samples.append(InputExample(texts=[query, passage], label=label))
            cnt += 1
            if cnt >= args.max_train_samples:
                break

    if missing_train:
        logging.warning("Skipped %d training triples due to missing passage ids.", missing_train)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

    evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = out_dir / f"cross-encoder-{args.model_name.replace('/', '-')}-{run_id}"

    logging.info("\n===== START TRAINING =====\n")
    logging.info("STOP MANUALLY AFTER ~60 MINUTES (Ctrl+C).")
    logging.info("Output path: %s", output_path)

    try:
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.num_epochs,
            evaluation_steps=args.evaluation_steps,
            warmup_steps=args.warmup_steps,
            output_path=str(output_path),
            use_amp=use_amp,
        )
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user (expected for 1-hour stop).")

    print("\n\n===== REPORT SUMMARY (copy into report notes) =====")
    print(f"model_name: {args.model_name}")
    print(f"output_path: {output_path}")
    print(f"train_batch_size: {args.train_batch_size}")
    print(f"max_length: {args.max_length}")
    print(f"pos_neg_ratio: {args.pos_neg_ratio}")
    print(f"max_train_samples_loaded: {len(train_samples)}")
    print("NOTE: Record the last printed training step at ~60 minutes as 'training steps'.")


if __name__ == "__main__":
    main()
