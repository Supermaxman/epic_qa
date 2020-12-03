#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_PRUNE_5
export MODEL_NAME=pt-biobert-base-msmarco-multi-sentence
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=consumer

python -m rerank_pruning.rerank_segments \
 --pred_path models/${MODEL_NAME}/${RUN_NAME}.pred \
 --output_path models/${MODEL_NAME}/${RUN_NAME}_PRUNED.pred \
 --threshold 1.0

python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${RUN_NAME}_PRUNED.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt

python rerank/epic_eval.py \
  data/epic_qa_prelim/prelim_judgments_corrected.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}