#!/usr/bin/env bash

export INDEX_NAME=passage_index
export DATASET=expert
export COLLECTION=epic_qa_prelim

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

python search_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name baseline_pass_doc
