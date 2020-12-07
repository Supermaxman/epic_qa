#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_7
export SEARCH_RUN=passage-large
export MODEL_NAME=rerank-expert-${SEARCH_RUN}
export PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export DATASET=expert
export INDEX_NAME=passage_index
export COLLECTION=epic_qa_prelim
export SEARCH_TOP_K=4000

python search/search_pass_index.py \
  --index_path data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
  --output_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --top_k ${SEARCH_TOP_K}

python search/search_passage_eval.py \
  --input_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json

python -m rerank.rerank \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --passage_search_run data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
  --pre_model_name ${PRE_MODEL_NAME} \
  --model_name ${MODEL_NAME}

python -m rerank.format_preds \
  --model_path models/${MODEL_NAME} \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.pred

python -m rerank.format_eval \
  --pred_path models/${MODEL_NAME}/${RUN_NAME}.pred \
  --output_path models/${MODEL_NAME}/${RUN_NAME}.txt

python rerank/rerank_sentence_eval.py \
  --input_path models/${MODEL_NAME}/${RUN_NAME}.txt \
  --label_path data/${COLLECTION}/prelim_judgments_corrected.json

python rerank/epic_eval.py \
  data/${COLLECTION}/prelim_judgments_corrected.json \
  models/${MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET}
