#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_PRUNE_3
export SCORE_RUN_NAME=HLTRI_RERANK_2
export MODEL_NAME=pt-biobert-base-msmarco-multi-sentence
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=consumer

python -m rerank_pruning.rerank_segments \
 --pred_path models/${MODEL_NAME}/${SCORE_RUN_NAME}.pred \
 --output_path models/${MODEL_NAME}/${RUN_NAME}_PRUNED.pred \
 --threshold 0.0 \
 --top_n_gram 1

python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${RUN_NAME}_PRUNED.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt

python rerank/epic_eval.py \
  data/epic_qa_prelim/prelim_judgments_corrected.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}