#!/usr/bin/env bash

export INDEX_NAME=passage_index
export DATASET=expert
export COLLECTION=epic_qa_prelim
export SEARCH_RUN=passage-baseline
export EXP_MODEL_NAME=docT5query-base-${DATASET}-${INDEX_NAME}
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000

python search/convert_passages_to_json.py \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --json_collection_path data/${COLLECTION}/${DATASET}/data_json

mkdir data/${COLLECTION}/${DATASET}/indices

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 12 \
  -input data/${COLLECTION}/${DATASET}/data_json \
  -index data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  -storePositions \
  -storeDocvectors \
  -storeRaw

mkdir data/${COLLECTION}/${DATASET}/search


python -m expand_query.expand \
 --collection_path data/${COLLECTION}/${DATASET}/data \
 --pre_model_name ${EXP_PRE_MODEL_NAME} \
 --model_name ${EXP_MODEL_NAME} \
 --top_k 10 \
 --num_samples 1 \
 --batch_size 16 \
 --max_seq_len 512

python -m expand_query.format_expand \
  --model_path models/${EXP_MODEL_NAME} \
  --output_path models/${EXP_MODEL_NAME}/${SEARCH_RUN}.exp