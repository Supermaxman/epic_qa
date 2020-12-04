#!/usr/bin/env bash

export INDEX_NAME=passage_index
export DATASET=expert
export COLLECTION=epic_qa_prelim
export SEARCH_RUN=passage-baseline

python search/convert_passages_to_json.py \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --json_collection_path data/${COLLECTION}/${DATASET}/data_json

mkdir data/${COLLECTION}/${DATASET}/indices

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input data/${COLLECTION}/${DATASET}/data_json \
  -index data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  -storePositions \
  -storeDocvectors \
  -storeRaw

mkdir data/${COLLECTION}/${DATASET}/search

python search_pass_index.py \
  --index_path data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  --query_path data/${COLLECTION}/${DATASET}/queries.json \
  --output_path data/${COLLECTION}/${DATASET}/search/${SEARCH_RUN} \
  --top_k 2000
