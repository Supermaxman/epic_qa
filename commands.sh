#!/usr/bin/env bash

# TODO add args
python convert_doc_to_json.py

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 8 -input data/expert/epic_qa_cord_2020-06-19_v2_doc_json/ \
 -index indices/expert/baseline_doc -storePositions -storeDocvectors -storeRaw


# TODO add args
python convert_doc_to_json.py

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 8 -input data/expert/epic_qa_cord_2020-06-19_v2_expanded_doc_json/ \
 -index indices/expert/expanded_doc -storePositions -storeDocvectors -storeRaw

alias trec_eval=/users/max/code/anserini-tools/eval/trec_eval.9.0.4/trec_eval

#trec_eval -m map qrels/expert/baseline_doc runs/expert/baseline_doc

# f'data/{doc_type}/expert_questions_prelim.json'
python search_index.py \
  --doc_type expert \
  --index baseline_doc \
  --query expert_questions_prelim.json \
  --run_name baseline_doc
trec_eval -m recip_rank data/expert/qrels-covid_d4_j3.5-4.txt runs/expert/baseline_doc

python search_index.py \
  --doc_type expert \
  --index expanded_doc \
  --query expert_questions_prelim.json \
  --run_name expanded_doc
trec_eval -m recip_rank data/expert/qrels-covid_d4_j3.5-4.txt runs/expert/expanded_doc

python rerank_run.py \
  --doc_type expert \
  --query expert_questions_prelim.json \
  --input_run_name baseline_doc \
  --run_name baseline_doc_rerank
trec_eval -m recip_rank data/expert/qrels-covid_d4_j3.5-4.txt runs/expert/baseline_doc_rerank
