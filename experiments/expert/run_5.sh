#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_EXPANDED_5
export SEARCH_RUN=passage-1200
export SCORE_RUN_NAME=HLTRI_RERANK_4
export SCORE_MODEL_NAME=pt-biobert-base-msmarco-expert-${SEARCH_RUN}
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000
export DATASET=expert
export COLLECTION=epic_qa_prelim

# needs to be run on GPU, TPUs do not like generator
python -m expand_query.expand \
 --input_path models/${SCORE_MODEL_NAME}/${SCORE_RUN_NAME}.txt \
 --collection_path data/${COLLECTION}/${DATASET}/data \
 --pre_model_name ${EXP_PRE_MODEL_NAME} \
 --model_name ${EXP_MODEL_NAME} \
 --top_k 10 \
 --num_samples 3 \
 --batch_size 16 \
 --max_seq_len 512

python -m expand_query.format_expand \
  --model_path models/${EXP_MODEL_NAME} \
  --output_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp

python -m rqe.rqe \
  --input_path models/${SCORE_MODEL_NAME}/${SCORE_RUN_NAME}.txt \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
  --model_name models/${RQE_MODEL_NAME}

python -m rqe.format_rqe \
  --model_path models/${RQE_MODEL_NAME} \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe \
  --threshold 0.10 \
  --num_samples 3

python -m rqe.format_run \
  --rqe_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe \
  --scores_path models/${SCORE_MODEL_NAME}/${SCORE_RUN_NAME}.txt \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored

python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored \
  --model_name models/${RQE_MODEL_NAME}

python -m rgqe.format_rgqe \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe \
  --threshold 0.5

python -m rgqe.format_eval \
  --results_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rqe_scored \
  --rgqe_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}_RGQE.txt \
  --threshold 0.8 \
  --overlap 1.0

python rerank/epic_eval.py \
  data/${COLLECTION}/prelim_judgments_corrected.json \
  models/${RQE_MODEL_NAME}/${RUN_NAME}_RGQE.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}


01d8cqn4-C000-S002
01d8cqn4-0-2
01d8cqn4-0-5