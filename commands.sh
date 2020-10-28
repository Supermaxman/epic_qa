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
  --batch_size 64 \
  --max_length 128

python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/expert/biobert_msmarco_multi_sentence \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence


python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-bert-base-uncased-msmarco \
  --run_path runs/expert/bert_base_msmarco_multi_sentence \
  --batch_size 16 \
  --max_length 512 \
  --multi_sentence

python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-bert-large-msmarco \
  --run_path runs/expert/bert_large_msmarco_multi_sentence \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence


python rerank_labels_sentence.py \
  --query_path data/consumer/consumer_questions_prelim.json \
  --collection_path data/consumer/version_2_split \
  --label_path data/consumer/consumer_fake_qrels.txt \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/consumer/biobert_msmarco_multi_sentence \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence

python rerank_labels_sentence.py \
  --query_path data/consumer/consumer_questions_prelim.json \
  --collection_path data/consumer/version_2_split \
  --label_path data/consumer/consumer_fake_qrels.txt \
  --rerank_model nboost/pt-bert-base-uncased-msmarco \
  --run_path runs/consumer/bert_base_msmarco_multi_sentence \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence


python rerank_labels_sentence_splits.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/expert/biobert_msmarco_span_split \
  --batch_size 32 \
  --max_length 512 \
  --top_k 1000


python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model models/v1 \
  --run_path runs/expert/v1 \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence \
  --custom_model

# TODO running
python rerank_labels_sentence.py \
  --query_path data/consumer/consumer_questions_prelim.json \
  --collection_path data/consumer/version_2_split \
  --label_path data/consumer/consumer_fake_qrels.txt \
  --rerank_model models/consumer-v1 \
  --run_path runs/consumer/consumer-v1 \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence \
  --custom_model

# TODO waiting on model training on GCP TPUs
python rerank_labels_sentence.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --label_path data/expert/qrels-covid_d4_j3.5-4.txt \
  --rerank_model models/expert-v1 \
  --run_path runs/expert/expert-v1 \
  --batch_size 32 \
  --max_length 512 \
  --multi_sentence \
  --custom_model

python rerank.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --search_run runs/expert/baseline_pass_full \
  --search_top_k 1000 \
  --rerank_model models/expert-v3 \
  --run_path runs/expert/expert-v3-full \
  --batch_size 64 \
  --max_length 128 \
  --custom_model


python eval.py \
  --query_path data/expert/expert_questions_prelim.json \
  --run_path runs/expert/expert-v3-full \
  --top_k 10


python search_pass_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name bm25_pass_full \
  --top_k 1000

python rerank.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --search_run runs/expert/bm25_pass_full \
  --search_top_k 1000 \
  --rerank_model nboost/pt-biobert-base-msmarco \
  --run_path runs/expert/baseline-bm25-pass-full \
  --batch_size 64 \
  --max_length 128

python rerank.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --search_run runs/expert/bm25_pass_full \
  --search_top_k 1000 \
  --rerank_model models/expert-v3 \
  --run_path runs/expert/expert-v3-bm25-pass-full \
  --batch_size 64 \
  --max_length 128 \
  --custom_model

python eval.py \
  --query_path data/expert/expert_questions_prelim.json \
  --run_path runs/expert/baseline-bm25-pass-full \
  --top_k 10

#EQ001: P=0.000, R=0.000, F1=0.000
#EQ040: P=0.100, R=0.006, F1=0.001
#EQ002: P=0.100, R=0.012, F1=0.002
#EQ005: P=0.100, R=0.012, F1=0.002
#EQ007: P=0.200, R=0.011, F1=0.004
#EQ011: P=0.000, R=0.000, F1=0.000
#EQ013: P=0.100, R=0.007, F1=0.001
#EQ029: P=0.000, R=0.000, F1=0.000
#EQ018: P=0.100, R=0.017, F1=0.003
#EQ020: P=0.400, R=0.095, F1=0.076
#EQ021: P=0.000, R=0.000, F1=0.000
#EQ036: P=0.000, R=0.000, F1=0.000
#EQ022: P=0.300, R=0.018, F1=0.011
#EQ023: P=0.000, R=0.000, F1=0.000
#EQ041: P=0.400, R=0.027, F1=0.022
#EQ025: P=0.000, R=0.000, F1=0.000
#EQ026: P=0.200, R=0.018, F1=0.007
#EQ030: P=0.200, R=0.006, F1=0.002
#EQ034: P=0.100, R=0.036, F1=0.007
#EQ037: P=0.100, R=0.011, F1=0.002
#EQ045: P=0.100, R=0.005, F1=0.001
#TOTAL Micro: P=0.119, R=0.009, F1=0.002

python eval.py \
  --query_path data/expert/expert_questions_prelim.json \
  --run_path runs/expert/expert-v3-bm25-pass-full \
  --top_k 10
#EQ001: P=0.000, R=0.000, F1=0.000
#EQ040: P=0.000, R=0.000, F1=0.000
#EQ002: P=0.100, R=0.012, F1=0.002
#EQ005: P=0.000, R=0.000, F1=0.000
#EQ007: P=0.200, R=0.011, F1=0.004
#EQ011: P=0.000, R=0.000, F1=0.000
#EQ013: P=0.100, R=0.007, F1=0.001
#EQ029: P=0.000, R=0.000, F1=0.000
#EQ018: P=0.100, R=0.017, F1=0.003
#EQ020: P=0.400, R=0.095, F1=0.076
#EQ021: P=0.000, R=0.000, F1=0.000
#EQ036: P=0.100, R=0.011, F1=0.002
#EQ022: P=0.300, R=0.018, F1=0.011
#EQ023: P=0.200, R=0.015, F1=0.006
#EQ041: P=0.300, R=0.020, F1=0.012
#EQ025: P=0.100, R=0.011, F1=0.002
#EQ026: P=0.300, R=0.027, F1=0.016
#EQ030: P=0.200, R=0.006, F1=0.002
#EQ034: P=0.000, R=0.000, F1=0.000
#EQ037: P=0.100, R=0.011, F1=0.002
#EQ045: P=0.100, R=0.005, F1=0.001
#TOTAL Micro: P=0.124, R=0.009, F1=0.002

python search_pass_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name bm25_pass_100 \
  --top_k 100

python rerank.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --search_run runs/expert/bm25_pass_100 \
  --search_top_k 100 \
  --rerank_model models/expert-v3 \
  --run_path runs/expert/expert-v3-bm25-pass-100 \
  --batch_size 64 \
  --max_length 128 \
  --custom_model


python eval.py \
  --query_path data/expert/expert_questions_prelim.json \
  --run_path runs/expert/expert-v3-bm25-pass-100 \
  --top_k 1000
#EQ001: P=0.300, R=0.026, F1=0.015
#EQ040: P=0.000, R=0.000, F1=0.000
#EQ002: P=0.100, R=0.012, F1=0.002
#EQ005: P=0.000, R=0.000, F1=0.000
#EQ007: P=0.200, R=0.011, F1=0.004
#EQ011: P=0.000, R=0.000, F1=0.000
#EQ013: P=0.100, R=0.007, F1=0.001
#EQ029: P=0.000, R=0.000, F1=0.000
#EQ018: P=0.000, R=0.000, F1=0.000
#EQ020: P=0.400, R=0.095, F1=0.076
#EQ021: P=0.000, R=0.000, F1=0.000
#EQ036: P=0.100, R=0.011, F1=0.002
#EQ022: P=0.400, R=0.024, F1=0.019
#EQ023: P=0.200, R=0.015, F1=0.006
#EQ041: P=0.100, R=0.007, F1=0.001
#EQ025: P=0.200, R=0.022, F1=0.009
#EQ026: P=0.000, R=0.000, F1=0.000
#EQ030: P=0.200, R=0.006, F1=0.002
#EQ034: P=0.000, R=0.000, F1=0.000
#EQ037: P=0.200, R=0.022, F1=0.009
#EQ045: P=0.000, R=0.000, F1=0.000
#TOTAL Micro: P=0.119, R=0.009, F1=0.002


python search_pass_index.py \
  --doc_type expert \
  --index baseline_pass \
  --query expert_questions_prelim.json \
  --run_name bm25_pass_1000 \
  --top_k 1000 \
  --debug

python rerank.py \
  --query_path data/expert/expert_questions_prelim.json \
  --collection_path data/expert/epic_qa_cord_2020-06-19_v2 \
  --search_run runs/expert/bm25_pass_1000 \
  --rerank_model models/expert-v3 \
  --run_path runs/expert/expert-v3-bm25-pass \
  --batch_size 64 \
  --max_length 128 \
  --custom_model \
  --debug


python eval.py \
  --query_path data/expert/expert_questions_prelim.json \
  --run_path runs/expert/expert-v3-bm25-pass \
  --debug