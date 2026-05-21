# INFORMATION RETRIEVAL


## PREPARATION

The objectives of this assignment are:

1. to get acquainted with fine-tuning and evaluating cross-encoder re-rankers
2. to learn about ensemble methods for combining the ranking run files of the
    different cross-encoder re-rankers

Additional objectives:

3. To learn how to implement query expansion with an LLM, with and without
    pseudo-relevance feedback

The assignment consists of 5 tasks.

**Note** : _Task 4 and 5 are not required for passing the assignment. For completing
task 1, 2, and 3 and the report your grade can be up to 7.5. For additionally
completing tasks 4 and 5 you can get up to 10._


## PREPARATION

➢ You will do the following tasks during this assignment:

```
➢ Task 1: working with two notebooks (prepared by us) that you need to understand
and run with the requested settings:
➢ Fine-tuning cross-encoders with the fine-tuning notebook:
https://colab.research.google.com/drive/1yXKZGGsMBSnaGFeKVR20FqRNwyYV6Nw6?usp=s
haring
➢ Performing inference with the evaluation notebook:
https://colab.research.google.com/drive/1tiVwPPmBaXCzbjr0fM44fiVFIyB-1lU_?usp=sharing
➢ Task 2 and 3: reading the ranx.fuse paper [1] and evaluating ensemble methods for
ranking
```
➢ You might do the following tasks, as a more challenging addition

```
➢ Task 4: Reading the paper “Query Expansion by Prompting Large Language Models”
and implementing the method for the given dataset
➢ Task 5: Adding pseudo-relevance feedback to your query expansion implementation
```

## PREPARATION

#### ➢ We assume you have completed the homework of week 7 before

#### starting the final assignment

```
➢ Some information such as the format of ranked lists, qrels files, and
how to use pytrec_eval for evaluation are explained in the week 7
exercise
➢ Before asking your questions, please be sure that the answers to them
are not provided in the homework exercise of week 7.
```

## TASK 1: FINE-TUNING CROSS-ENCODERS

➢ Familiarize yourself with the provided fine-tuning notebook by reviewing
the instructions and implementation.

➢ Use the fine-tuning notebook to fine-tune three different cross-encoders.
To do so, adjust the "model_name" parameter in the notebook. By
default, the parameter is set to "cross-encoder/ms-marco-MiniLM-L- 2 -
v2". You should fine-tune the three below models by modifying this
parameter:
➢ cross-encoder/ms-marco-MiniLM-L- 2 - v
➢ cross-encoder/ms-marco-TinyBERT-L- 2 - v
➢ distilroberta-base

➢ **Fine-tune each cross-encoder model for only 1 hour**. Note that you will
need to manually stop the notebook run after one hour.


## TASK 1: FINE-TUNING CROSS-ENCODERS

Some hints:

➢ You will find the fine-tuned model stored on your Google Drive. (Check the
model_save_path parameter).

➢ Please note that you must write down and keep the total number of steps each that the
model has been trained for because you need to provide this information into your
report.
➢ For instance, the below image shows a model trained for 15226 total number of steps out
of 625k steps.


## TASK 1: EVALUATING CROSS-ENCODERS

➢ By following the previous instructions, you have successfully fine-tuned 3
cross-encoders on the MSMARCO re-ranking dataset.

➢ Next, evaluate each of the fine-tuned cross-encoder models on the 43
queries of TREC DL’19 using the evaluation notebook and report
NDCG@10, Recall@100, and MAP@
➢ The evaluation notebook contains comments and instructions which give
you clues how this should be done.

➢ By running the evaluation notebook for a fine-tuned model, a ranking run
file will be generated named “ranking.run” in the folder of the fine-tuned
model. You need these ranking run files in order to do task 2.

_By analyzing and discussing the results of your evaluation and fine-tuning, you
will complete task 1 and gain a deeper understanding of the performance and
capabilities of each model._


## TASK 1: REPORTING RESULTS

In your report:

➢ **show a table** with your results obtained from fine-tuning and evaluation
for the three models (rows) on each of the three evaluation metrics
(columns). Also add a column with the number of training steps after one
hour training

➢ **include a discussion of the results**. Address the following questions:

```
➢ Why do you think one model outperforms another on a specific metric?
➢ Does this mean that the higher-performing model will be more effective for
all new, unseen queries?
➢ Are there any limitations or drawbacks to the models that should be
considered?
```

### TASK 2: SELECT AND APPLY FIVE ENSEMBLE

### METHODS

➢ Read the ranx.fuse paper [1].

➢ Select 5 ensemble methods (fusion
algorithms) from the paper and apply
them to the three model run files with
ranx. See the list of ensemble
approaches implemented by ranx:
https://amenra.github.io/ranx/fusion/

➢ Evaluate the effectiveness of the
ensembles with ranx on NDCG@10,
Recall@100, and MAP@


### TASK 2: DISCUSSION OVER ENSEMBLE

### METHODS

In your report,

➢ **show a table** with your results obtained with the five models (rows) on each of the three
evaluation metrics (columns).

➢ **include a discussion** of the effectiveness of different ensemble methods. You should address at
least three of the following questions in your discussion:

1. What are the differences in effectiveness between the various ensemble methods you tested?
2. If one ensemble method works better, why do you think it is more effective than the others?
3. What is your interpretation of the usefulness of ensemble (fusion) methods in general?
4. Consider a more sophisticated approach to performing ensemble methods. Think about the following
    questions:
➢ How could you leverage the strengths of each individual method to improve overall performance?
➢ Could you develop a weighted ensemble method that assigns different weights to different models based
on their strengths and weaknesses?
➢ Are there any other techniques or strategies you could use to optimize the ensemble process?


### TASK 3: ANALYZING MOST EFFECTIVE

### ENSEMBLE METHOD

#### ➢ Select the most effective ensemble method that you identified in

#### task 2 and use it for this task. If there are multiple equally effective

#### methods, feel free to choose the one that you prefer.

#### ➢ Apply the selected ensemble method to all possible combinations

#### of two models out of the three ranking run files.

#### By applying the ensemble method to different combinations of models

#### and comparing the results, you will gain a better understanding of the

#### strengths and weaknesses of each model and the effectiveness of the

#### ensemble method in different scenarios.


### TASK 3: DISCUSSION OVER MOST EFFECTIVE

### ENSEMBLE METHOD

In your report:

➢ **show a table** with your results for the ensembles in terms of nDCG@10,
Recall@100, and MAP@

➢ **include a discussion** and interpret the results obtained from applying the
ensemble method to different model combinations. Address the following
questions:
➢ Which combinations of models performed the best and which performed
the worst? Why do you think this is the case?
➢ Did the performance of the ensemble method vary significantly across
different combinations of models? If so, why?
➢ Were there any unexpected results or patterns in the data that you
observed? How do these results compare to your expectations?


### TASK 4: QUERY EXPANSION BY PROMPTING

### LARGE LANGUAGE MODELS

➢ Read the following paper: “Query Expansion by Prompting Large Language Models” [2]

➢ Queries: a set of 43 queries (link) and their top-1k candidate documents (link) and qrels
(link) are provided for expansion.

➢ Use the provided notebook (link) example to load an LLM. We use Zephyr-7b-beta in
this code. Feel free to change the LLM to your favourite (e.g. Llama, Gemma, Qwen)

➢ Implement the generation pipeline for using the designated prompt (see row "CoT" of
Table 3 in paper [2]) for query reformulation.
➢ Please note that since we use BERT-based re-rankers that have limited input length (
tokens), we advise you to prevent from repeating the original query for five times in
contrast to [2] for query expansion. This ensures that the query and candidate document
can fit into the input of BERT-based re-rankers.

➢ You may innovate with custom prompts and compare effectiveness.

_By completing this task, you will gain a deeper understanding of how LLM-driven query
reformulation impacts the effectiveness of cross-encoder re-rankers._


### TASK 4: QUERY EXPANSION BY PROMPTING

### LARGE LANGUAGE MODELS

#### ➢ Evaluation and reporting:

```
➢ Report on the effectiveness of the cross-encoder before and after query
expansion for these 43 queries. Report NDCG@10, Recall@100, and
MAP@1000.
➢ (Note: you might need to truncate the output of the LLM to fit into the cross-
encoder input.)
➢ Show a result table
➢ Discuss your result and in case your findings are different from the findings in
the paper, explain why you think that is the case
```

### TASK 5: QUERY EXPANSION WITH PSEUDO-

### RELEVANCE FEEDBACK

#### Summary: Pseudo Relevance Feedback (PRF) is a technique used in

#### information retrieval to improve the effectiveness without having extra

#### provided relevance assessments. The core idea is to automatically

#### refine a search query based on the initial set of top-retrieved

#### documents, under the assumption that these documents are relevant

#### ( see L05 on probabilistic models ).

#### Here, we aim to use PRF for query expansion with LLMs, as follows:

#### ➢ Use the top-3 documents retrieved from the original query for

#### query expansion.

#### ➢ For the prompt, use the prompt in Table 3, row "CoT/PRF" in [2]


### TASK 5: QUERY EXPANSION WITH PSEUDO-

### RELEVANCE FEEDBACK

#### ➢ Evaluation and reporting:

```
➢ Report on the effectiveness of the cross-encoder before and after query
expansion (with PRF) for the provided 43 queries and compare it with
original query, and expansion without PRF. Report NDCG@10,
Recall@100, and MAP@1000.
➢ Show a result table
➢ Discuss your result and in case your findings are different from the findings in
the paper, explain why you think that is the case.
```

## REPORT WRITING

Finally, the report should contain the following parts:
➢ (2 points) An introduction of the task and goals of your experiments
➢ (1.5 points) Task 1: Reporting the effectiveness of fine-tuning and evaluating cross-encoders, with a
discussion
➢ (2.5 points) Task 2: Reporting the effectiveness of five ensemble methods over the three fine-tuned
models’ run files, with a discussion.
➢ (1.5 points) Task 3: Reporting the effectiveness of applying the most effective ensemble method to
all possible combinations of two models out of the three ranking run files, with a discussion.
Additional points can be achieved by:
➢ (1 point) Task 4: Reporting the effectiveness of the cross-encoder with and without prompt-based
query expansion, and a description of these results.
➢ (1 point) Task 5: Reporting the effectiveness of the cross-encoder with prompt-based query
expansion with PRF, and a description of these results. If you completed both tasks 4 and 5, you can
show the results in one table and discuss.
➢ (0.5 points) for interesting results or analysis gained by custom prompts/strategies in task 4 or 5.


## REPORT WRITING

➢ You must work in **Overleaf with your student account** (ULCN)

```
➢ If you already have a private account, please connect it to ULCN through
the SSO login (this will also be a requirement for your master thesis)
```
➢ The report’s final length should **maximum 4 pages**. This is the hard
maximum. No appendix.
➢ Template: https://www.overleaf.com/latex/templates/association-for-
computing-machinery-acm-large- 1 - column-format-template/fsyrjmfzcwyy

➢ At the end of the report, add a **reflection paragraph** :

```
➢ Briefly describe the work division between the two team members.
➢ Did you use AI assistants in research or writing? If yes, please specify briefly how: (a)
which assistant did you use, (b) for which tasks (understanding, programming,
debugging, writing, editing), (c) and your reflection on the use: was it helpful, did it
work, did you notice any errors in the output? (maximum 1 paragraph)
```

## SUBMISSIONS

#### ➢ Please submit your report as PDF and your code separately. Don’t

#### upload a zip file.

#### ➢ Deadline: May 24, 2026

#### ➢ If you need help, contact the TAs on time before the deadline, at

#### ircourse@liacs.leidenuniv.nl

#### ➢ Use your own words in the reporting. Make sure you don’t copy

#### text from external sources, other student teams, or the output of AI

#### assistants such as ChatGPT.


## REFERENCES

#### ➢ [1] Bassani, Elias, et al. "ranx. fuse: A Python Library for

#### Metasearch." Proceedings of the 31st ACM International

#### Conference on Information & Knowledge Management. 2022.

#### https://dl.acm.org/doi/pdf/10.1145/3511808.

#### ➢ [2] Jagerman, Rolf, et al. "Query expansion by prompting large

#### language models." Generative Information Retrieval workshop at

#### ACM SIGIR 2023. https://arxiv.org/pdf/2305.03653.pdf
