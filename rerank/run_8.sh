#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_8
export SCORE_RUN_NAME=HLTRI_RERANK_1
export MODEL_NAME=pt-biobert-base-msmarco
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=consumer

python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${SCORE_RUN_NAME}.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt \
  --top_k 100

python rerank/epic_eval.py \
  data/epic_qa_prelim/prelim_judgments_corrected.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}