#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_18
export SEARCH_RUN=passage-large
export MODEL_NAME=rerank-expert-${SEARCH_RUN}-${RUN_NAME}
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=expert
export INDEX_NAME=passage_index
export COLLECTION=epic_qa_prelim
export SEARCH_TOP_K=1000
export NEGATIVE_SAMPLES=400

#python search/search_index.py \
#  --index_path data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
#  --query_path data/${COLLECTION}/${DATASET}/questions.json \
#  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
#  --output_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
#  --top_k ${SEARCH_TOP_K} \
#; \
#python search/search_passage_eval.py \
#  --input_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
#  --label_path data/${COLLECTION}/prelim_judgments_corrected.json
#
#python rerank/split_data.py \
#  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
#  --search_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
#  --output_path data/${COLLECTION}/${DATASET}/split \
#  --dataset ${DATASET}

python -m rerank.rerank_train \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --passage_search_run data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
  --split_path data/${COLLECTION}/${DATASET}/split \
  --pre_model_name ${PRE_MODEL_NAME} \
  --model_name ${MODEL_NAME} \
  --max_seq_len 96 \
  --batch_size 32 \
  --negative_samples ${NEGATIVE_SAMPLES} \
  --add_all_labels \
  --weighted_loss \
  --learning_rate 1e-5 \
  --epochs 5

python -m rerank.rerank \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --passage_search_run data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --label_path data/${COLLECTION}/${DATASET}/split/val.json \
  --pre_model_name ${PRE_MODEL_NAME} \
  --model_name ${MODEL_NAME} \
  --max_seq_len 96 \
  --load_trained_model

python -m rerank.format_preds \
  --model_path models/${MODEL_NAME} \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.pred \
; \
python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${RUN_NAME}.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt \
  --top_k 1000 \
; \
python rerank/rerank_sentence_eval.py \
  --input_path models/${MODEL_NAME}/${RUN_NAME}.txt \
  --label_path data/${COLLECTION}/${DATASET}/split/val.json \
; \
python rerank/epic_eval.py \
  data/${COLLECTION}/${DATASET}/split/val.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET} \
  | tail -n 3 \
  | awk \
    '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
    END{ for (i in RtoC) print RtoC[i] }' \
  | tail -n 2 > models/${MODEL_NAME}/${RUN_NAME}.eval \
; \
cat models/${MODEL_NAME}/${RUN_NAME}.eval
