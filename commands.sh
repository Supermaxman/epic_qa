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

python convert_passages_to_json.py \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --json_collection_path data/expert/epic_qa_cord_2020-06-19_v2_pass_json

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input data/expert/epic_qa_cord_2020-06-19_v2_pass_json/ \
  -index indices/expert/baseline_pass \
  -storePositions \
  -storeDocvectors \
  -storeRaw

python search_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name baseline_pass_doc

trec_eval -m recip_rank \
  data/expert/qrels-covid_d4_j3.5-4.txt \
  runs/expert/baseline_pass_doc

# expand
python convert_passages_to_json.py \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2_expanded \
  --json_collection_path data/expert/epic_qa_cord_2020-06-19_v2_pass_expanded_json \
  --ex

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input data/expert/epic_qa_cord_2020-06-19_v2_pass_expanded_json/ \
  -index indices/expert/expanded_pass \
  -storePositions \
  -storeDocvectors \
  -storeRaw

python search_index.py \
  --doc_type expert \
  --index expanded_pass \
  --query expert_questions_prelim.json \
  --run_name expanded_pass_doc

trec_eval -m recip_rank \
  data/expert/qrels-covid_d4_j3.5-4.txt \
  runs/expert/expanded_pass_doc




python search_pass_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name baseline_pass_full


python rerank_run.py \
  --doc_type expert \
  --query expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --input_run_name baseline_pass_full \
  --run_name baseline_pass_full_doc_rerank

trec_eval -m recip_rank \
  data/expert/qrels-covid_d4_j3.5-4.txt \
  runs/expert/baseline_pass_full_doc_rerank


python rerank_labels.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/expert/biobert_msmarco \
  --batch_size 16

python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/expert/biobert_msmarco_sentence \
  --batch_size 64


