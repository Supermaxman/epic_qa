#!/usr/bin/env bash

export RUN_NAME=HLTRI_RERANK_RQE_21
export SEARCH_RUN=passage-large
export RERANK_RUN_NAME=HLTRI_RERANK_15
export RERANK_MODEL_NAME=rerank-expert-${SEARCH_RUN}-${RERANK_RUN_NAME}
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000
export DATASET=expert
export COLLECTION=epic_qa_prelim
export RQE_THRESHOLD=0.9

python -m rerank.extract_answers \
  --search_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --collection_path data/${COLLECTION}/${DATASET}/data \
  --output_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.answers

# create expanded questions for every answer
#python expand_query/expand.py \
# --input_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
# --collection_path data/${COLLECTION}/${DATASET}/data \
# --pre_model_name ${EXP_PRE_MODEL_NAME} \
# --model_name ${EXP_MODEL_NAME} \
# --top_k 20 \
# --num_samples 20 \
# --batch_size 16 \
#; \
#python -m expand_query.format_expand \
#  --model_path models/${EXP_MODEL_NAME} \
#  --output_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp

# self entailment
python -m rgqe.rgqe \
  --input_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode self \
; \
python -m rgqe.format_rgqe_self \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_self \
; \
python -m rgqe.rgqe_self_components \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_self \
  --expand_path models/${EXP_MODEL_NAME}/${RUN_NAME}.exp \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --threshold ${RQE_THRESHOLD}

# top_k question entailment to filter out bad generated questions
python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --search_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --query_path data/${COLLECTION}/${DATASET}/questions.json \
  --label_path data/${COLLECTION}/${DATASET}/split/val.json \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode question \
  --top_k 100 \
; \
python -m rgqe.format_rgqe_question \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_question

# top_k set entailment
python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --qe_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_question \
  --search_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.txt \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode top \
  --top_k 100 \
  --threshold 0.01 \
; \
python -m rgqe.format_rgqe_top \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top \
; \
python -m rgqe.rgqe_top_components \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top \
  --cc_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc \
  --threshold ${RQE_THRESHOLD}


python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc \
  --cc_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --model_name models/${RQE_MODEL_NAME} \
  --mode all \
  --max_seq_len 64 \
  --batch_size 128 \
  --use_tpus

# all entailment against sets
python -m rgqe.rgqe \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc \
  --cc_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_cc \
  --model_name models/${RQE_MODEL_NAME} \
  --max_seq_len 64 \
  --mode all \
; \
python -m rgqe.format_rgqe_all \
  --model_path models/${RQE_MODEL_NAME} \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all \

python -m rgqe.rgqe_all_components \
  --input_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all \
  --cc_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_top_cc \
  --answers_path models/${RERANK_MODEL_NAME}/${RERANK_RUN_NAME}.answers \
  --queries_path data/${COLLECTION}/${DATASET}/questions.json \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all_cc_scored \
  --threshold ${RQE_THRESHOLD} \
  --ratio 0.8 \
; \
python -m rgqe.format_eval \
  --results_path models/${RQE_MODEL_NAME}/${RUN_NAME}.rgqe_all_cc_scored \
  --output_path models/${RQE_MODEL_NAME}/${RUN_NAME}.txt \
; \
python rerank/epic_eval.py \
  data/${COLLECTION}/${DATASET}/split/val.json \
  models/${RQE_MODEL_NAME}/${RUN_NAME}.txt \
  rerank/.${DATASET}_ideal_ranking_scores.tsv \
  --task ${DATASET} \
  | tail -n 3 \
  | awk \
    '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
    END{ for (i in RtoC) print RtoC[i] }' \
  | tail -n 2 > models/${RQE_MODEL_NAME}/${RUN_NAME}.eval \
; \
cat models/${RQE_MODEL_NAME}/${RUN_NAME}.eval


