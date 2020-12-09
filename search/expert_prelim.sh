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

python search/convert_sentences_to_json.py \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --json_collection_path data/${COLLECTION}/${DATASET}/data_sentence_json

export INDEX_NAME=sentence_index

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 12 \
  -input data/${COLLECTION}/${DATASET}/data_sentence_json \
  -index data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  -storePositions \
  -storeDocvectors \
  -storeRaw

python search/extract_collection.py \
 --collection_path data/${COLLECTION}/${DATASET}/data \
 --output_path data/${COLLECTION}/${DATASET}/data.jsonl


python expand_query/expand.py \
 --collection_path data/${COLLECTION}/${DATASET}/data.jsonl \
 --pre_model_name ${EXP_PRE_MODEL_NAME} \
 --model_name ${EXP_MODEL_NAME} \
 --top_k 10 \
 --num_samples 3 \
 --batch_size 64 \
 --max_seq_len 256 \
 --gpus 4,5,6,7 \
 --is_distributed \
; \
python expand_query/format_expand_jsonl.py \
  --model_path models/${EXP_MODEL_NAME} \
  --output_path models/${EXP_MODEL_NAME}/${SEARCH_RUN}.expl \
  --num_processes 16


export INDEX_NAME=passage_index_exp

mkdir data/${COLLECTION}/${DATASET}/data_expanded

python search/expand_passages_jsonl.py \
  --collection_path data/${COLLECTION}/${DATASET}/data.jsonl \
  --expand_path data/${COLLECTION}/${DATASET}/data.expl \
  --output_path data/${COLLECTION}/${DATASET}/data_expanded/data_expanded.jsonl

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 12 \
  -input data/${COLLECTION}/${DATASET}/data_expanded \
  -index data/${COLLECTION}/${DATASET}/indices/${INDEX_NAME} \
  -storePositions \
  -storeDocvectors \
  -storeRaw
