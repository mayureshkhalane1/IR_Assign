#!/usr/bin/env python3
"""Query expansion for TREC DL 2019 using local Ollama (Zephyr-7B-beta).

Generates an expanded queries TSV (qid\tquery_text) that can be fed into
scripts/evaluate.py via --queries_path.

Two modes:
- cot: chain-of-thought style reasoning (we only keep the final expanded query)
- cot_prf: chain-of-thought + pseudo-relevance feedback: we pass top passages
  (from the candidate top1000 file) and ask the LLM to expand the query.

Notes:
- We do NOT save or require the model's reasoning in the report. We only keep
  the final expanded query string.
- We hard-truncate expansions to keep cross-encoder inputs within max length.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import requests
import tqdm


OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(model: str, prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


def normalize_one_line(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def truncate_tokens_rough(s: str, max_words: int) -> str:
    # cheap truncation: good enough to avoid super long expansions
    words = s.split()
    return " ".join(words[:max_words])


def load_queries(path: Path) -> dict[str, str]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    queries = {}
    with open_fn(path, "rt") as f:
        for line in f:
            qid, q = line.rstrip("\n").split("\t", 1)
            queries[qid] = q
    return queries


def load_prf_context(top1000_path: Path, k: int) -> dict[str, list[str]]:
    ctx = {}
    with gzip.open(top1000_path, "rt") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) == 4:
                qid, _pid, _q, passage = cols
            else:
                qid, passage = cols[0], cols[-1]
            if qid not in ctx:
                ctx[qid] = []
            if len(ctx[qid]) < k:
                ctx[qid].append(passage)
    return ctx


def prompt_cot(query: str) -> str:
    return f"""You are helping with information retrieval query reformulation.

Original query: {query}

Think step by step about the user intent and likely synonyms/related terms.
Then output ONLY the final expanded query as a single line, without quotes.
The expanded query should be concise and search-oriented.
"""


def prompt_cot_prf(query: str, passages: list[str]) -> str:
    joined = "\n\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))
    return f"""You are helping with information retrieval query reformulation.

Original query: {query}

Pseudo-relevant passages (top ranked candidates):
{joined}

Think step by step about the query intent and important terms appearing in the passages.
Then output ONLY the final expanded query as a single line, without quotes.
The expanded query should be concise and search-oriented.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zephyr-7b-beta")
    ap.add_argument("--mode", choices=["cot", "cot_prf"], required=True)
    ap.add_argument(
        "--queries",
        default="models/datasets/trec_dl_2019/msmarco-test2019-queries.tsv.gz",
        help="Input queries tsv(.gz)",
    )
    ap.add_argument(
        "--top1000",
        default="models/datasets/trec_dl_2019/msmarco-passagetest2019-top1000.tsv.gz",
        help="Top1000 candidates file (needed for cot_prf)",
    )
    ap.add_argument("--prf_k", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=32, help="Max words in expanded query")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", required=True, help="Output expanded queries TSV")
    args = ap.parse_args()

    queries_path = Path(args.queries)
    queries = load_queries(queries_path)

    prf_ctx = None
    if args.mode == "cot_prf":
        prf_ctx = load_prf_context(Path(args.top1000), args.prf_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for qid, q in tqdm.tqdm(queries.items(), desc="Expanding queries"):
        if args.mode == "cot":
            prompt = prompt_cot(q)
        else:
            passages = (prf_ctx or {}).get(qid, [])
            prompt = prompt_cot_prf(q, passages)

        resp = call_ollama(args.model, prompt, temperature=args.temperature)
        expanded = truncate_tokens_rough(normalize_one_line(resp), args.max_words)
        if not expanded:
            expanded = q
        lines.append(f"{qid}\t{expanded}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
