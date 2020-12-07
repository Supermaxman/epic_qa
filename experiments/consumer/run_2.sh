#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_2
export MODEL_NAME=pt-biobert-base-msmarco-multi-sentence
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=consumer
python -m rerank.rerank \
  --query_path data/epic_qa_prelim/${DATASET}/questions.json \
  --collection_path data/epic_qa_prelim/${DATASET}/data \
  --label_path data/epic_qa_prelim/prelim_judgments_corrected.json \
  --pre_model_name ${PRE_MODEL_NAME} \
  --model_name ${MODEL_NAME} \
  --multi_sentence \
  --max_seq_len 128

python -m rerank.format_preds \
  --model_path models/${MODEL_NAME} \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.pred

python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${RUN_NAME}.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt

python rerank/epic_eval.py \
  data/epic_qa_prelim/prelim_judgments_corrected.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}