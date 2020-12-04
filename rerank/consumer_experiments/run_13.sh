#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_EXPANDED_13
export SCORE_RUN_NAME=HLTRI_RERANK_1
export SCORE_MODEL_NAME=pt-biobert-base-msmarco
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
export PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000
export DATASET=consumer
export COLLECTION=epic_qa_prelim

python -m expand_query.expand \
 --input_path models/${SCORE_MODEL_NAME}/${SCORE_RUN_NAME}.txt \
 --collection_path data/${COLLECTION}/${DATASET}/data \
 --pre_model_name ${PRE_MODEL_NAME} \
 --model_name ${EXP_MODEL_NAME} \
 --top_k 10 \
 --num_samples 10

python -m expand_query.format_expand \
  --model_path models/${EXP_MODEL_NAME} \
  --output_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp

python -m rqe.rqe \
  --input_path models/${SCORE_MODEL_NAME}/${SCORE_RUN_NAME}.txt \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
  --model_name models/${RQE_MODEL_NAME}

python -m rqe.rqe_format \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe


python -m rqe.format_eval \
  --pred_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.txt

python rerank/epic_eval.py \
  data/${COLLECTION}/prelim_judgments_corrected.json \
  models/${RQE_MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}