# Auto-generated from (Evaluating)_Final_assignment.ipynb# Code cells only; markdown removed.
# --- cell 1 ---# [ipynb-magic removed] !pip install sentence-transformers

# --- cell 2 ---# [ipynb-magic removed] !pip install pytrec_eval

# --- cell 3 ---"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import logging
from collections import defaultdict
import numpy as np
import sys
import pytrec_eval
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

# --- cell 4 ---from google.colab import drive
drive.mount('/content/gdrive')
base_path = "./gdrive/MyDrive/cross-encoder-reranker-ir-course-2026/"

# --- cell 5 ---# [ipynb-magic removed] !mkdir -p $base_path

# --- cell 6 ---model_save_path = "./gdrive/MyDrive/cross-encoder-reranker-ir-course-2023/finetuned_models/cross-encoder-cross-encoder-ms-marco-MiniLM-L-2-v2-2023-04-10_13-13-29/" #@param {type:"string"}

# --- cell 7 ---# [ipynb-magic removed] !wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
# [ipynb-magic removed] !tar -xvzf  queries.tar.gz

# --- cell 8 ---"""
This file evaluates CrossEncoder on the TREC 2019 Deep Learning (DL) Track: https://arxiv.org/abs/2003.07820

TREC 2019 DL is based on the corpus of MS Marco. MS Marco provides a sparse annotation, i.e., usually only a single
passage is marked as relevant for a given query. Many other highly relevant passages are not annotated and hence are treated
as an error if a model ranks those high.

TREC DL instead annotated up to 200 passages per query for their relevance to a given query. It is better suited to estimate
the model performance for the task of reranking in Information Retrieval.

Run:
python eval_cross-encoder-trec-dl.py cross-encoder-model-name

"""


data_folder = 'trec2019-data'
os.makedirs(data_folder, exist_ok=True)

#Read test queries
queries = {}
queries_filepath = os.path.join(data_folder, 'msmarco-test2019-queries.tsv.gz')
if not os.path.exists(queries_filepath):
    logging.info("Download "+os.path.basename(queries_filepath))
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz', queries_filepath)

with gzip.open(queries_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

#Read which passages are relevant
relevant_docs = defaultdict(lambda: defaultdict(int))
qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

if not os.path.exists(qrels_filepath):
    logging.info("Download "+os.path.basename(qrels_filepath))
    util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)


with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, score = line.strip().split()
        score = int(score)
        if score > 0:
            relevant_docs[qid][pid] = score

# Only use queries that have at least one relevant passage
relevant_qid = []
for qid in queries:
    if len(relevant_docs[qid]) > 0:
        relevant_qid.append(qid)


# Read the top 1000 passages that are supposed to be re-ranked
passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')

if not os.path.exists(passage_filepath):
    logging.info("Download "+os.path.basename(passage_filepath))
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz', passage_filepath)



passage_cand = {}
with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in passage_cand:
            passage_cand[qid] = []

        passage_cand[qid].append([pid, passage])

logging.info("Queries: {}".format(len(queries)))

# --- cell 9 ---queries_result_list = []
run = {}
model = CrossEncoder(model_save_path, max_length=512)

for qid in tqdm.tqdm(relevant_qid):
    query = queries[qid]

    cand = passage_cand[qid]
    pids = [c[0] for c in cand]
    corpus_sentences = [c[1] for c in cand]

    cross_inp = [[query, sent] for sent in corpus_sentences]

    if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
        cross_scores = model.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
    else:
        cross_scores = model.predict(cross_inp).tolist()

    cross_scores_sparse = {}
    for idx, pid in enumerate(pids):
        cross_scores_sparse[pid] = cross_scores[idx]

    sparse_scores = cross_scores_sparse
    run[qid] = {}
    for pid in sparse_scores:
        run[qid][pid] = float(sparse_scores[pid])

# --- cell 10 ---evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {'ndcg_cut.10', 'recall_100', 'map_cut.1000'})
scores = evaluator.evaluate(run)

print("Queries:", len(relevant_qid))
print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()])*100))
print("Recall@100: {:.2f}".format(np.mean([ele["recall_100"] for ele in scores.values()])*100))
print("MAP@1000: {:.2f}".format(np.mean([ele["map_cut_1000"] for ele in scores.values()])*100))

# --- cell 11 ---import operator
for qid in run.keys():
  run[qid] = sorted(run[qid].items(), key=operator.itemgetter(1), reverse = True)

# --- cell 12 ---ranking_lines = []
for qid in run.keys():
  for rank, did_pred_score in enumerate(run[qid]):
    did, pred_score = did_pred_score
    line = "{qid} Q0 {did} {rank} {pred_score} STANDARD".format(qid=qid, did=did, rank=rank, pred_score=str(pred_score))
    ranking_lines.append(line)

# --- cell 13 ---ranking_run_file_path = model_save_path + "ranking.run"
f_w = open(ranking_run_file_path, "w+")
f_w.write("\n".join(ranking_lines))
f_w.close()

# --- cell 14 ---# [ipynb-magic removed] !head -n 3 $ranking_run_file_path
