#!/usr/bin/env bash

# run names
export EXP_DATA_RUN_NAME=HLTRI_REGEQUES_EXP_DATA_1
# 1 is expanded, 2 is not expanded
#export INDEX_NAME=HLTRI_REGEQUES_EXP_INDEX_1
export INDEX_NAME=HLTRI_REGEQUES_EXP_INDEX_2
export SEARCH_RUN_NAME=HLTRI_REGEQUES_SEARCH_1
export RERANK_RUN_NAME=HLTRI_REGEQUES_RERANK_1
export RERANK_RUN_MODEL_NAME=HLTRI_REGEQUES_RERANK_1
export EXP_ANSWER_RUN_NAME=HLTRI_REGEQUES_EXP_ANSWER_1
export RQE_RUN_NAME=HLTRI_REGEQUES_SEP_4

# collection and task names
export COLLECTION=epic_qa_prelim
export DATASET=expert

# major hyper-parameters for system
export SEARCH_TOP_K=500
export NEGATIVE_SAMPLES=800
export RGQE_TOP_K=100
export RGQE_SELF_THRESHOLD=0.8
export RGQE_TOP_C_THRESHOLD=0.8
export RGQE_ALL_THRESHOLD=0.8
export RQE_TOP_THRESHOLD=0.01
export RQE_ALL_THRESHOLD=0.1
export RGQE_RATIO=0.9
export RGQE_SEQ_LEN=96
export RGQE_BATCH_SIZE=64
export EXP_ANSWER_TOP_K=20
export EXP_ANSWER_NUM_SAMPLES=20
export EXP_ANSWER_BATCH_SIZE=16
export EXP_ANSWER_SEQ_LEN=96

export RERANK_SEQ_LEN=96
export RERANK_BATCH_SIZE=32

# flags to avoid re-running certain components
# index & search flags
export CREATE_INDEX=false
export EXPAND_INDEX=false
export SEARCH_INDEX=false

# rerank flags
# RERANK fine-tune reranking model using training set
export TRAIN_RERANK=false
# RERANK run rerank using trained model on validation set
export RUN_RERANK=false
# RERANK run evaluation script on validation set
export EVAL_RERANK=false

# rerank answer query expansion flags
export RUN_EXPAND_ANSWERS=true

# RGQE pairwise self-entailment to find entailed sets for each answer
export RUN_RGQE_SELF=false
# RGQE query-generated question entailment to filter poor generated questions
export RUN_RGQE_QUESTION=false
# RGQE full set-pairwise entailment for top_k answers for each query
export RUN_RGQE_TOP=false
# RGQE top_k set entailment to all set entailment to find entailed sets for all answers
export RUN_RGQE_ALL=false
# RGQE rerank answers based on generated question entailment sets
export RUN_RGQE_RERANK=true
# RGQE run evaluation script on validation set
export EVAL_RGQE=true

export RERANK_MODEL_NAME=rerank-${DATASET}-${RERANK_RUN_MODEL_NAME}
#export RERANK_MODEL_NAME=rerank-expert-passage-large-HLTRI_RERANK_15
export EXP_MODEL_NAME=docT5query-base
export RQE_MODEL_NAME=quora-seq-nboost-pt-bert-base-uncased-msmarco

export RERANK_PRE_MODEL_NAME=nboost/pt-biobert-base-msmarco
export EXP_PRE_MODEL_NAME=models/docT5query-base/model.ckpt-1004000

export DATASET_PATH=data/${COLLECTION}/${DATASET}
export JUDGEMENTS_PATH=data/${COLLECTION}/prelim_judgments_corrected.json
export QUERY_PATH=${DATASET_PATH}/questions.json
export LABEL_PATH=${DATASET_PATH}/split/val.json
export COLLECTION_PATH=${DATASET_PATH}/data
export COLLECTION_JSONL_PATH=${DATASET_PATH}/data_jsonl
export COLLECTION_JSONL_FILE_PATH=${COLLECTION_JSONL_PATH}/data.jsonl

export ARTIFACTS_PATH=artifacts/${COLLECTION}/${DATASET}

export EXP_DATA_PATH=${ARTIFACTS_PATH}/${EXP_DATA_RUN_NAME}
export EXP_DATA_FILE_PATH=${EXP_DATA_PATH}/data.expl

export INDEX_FILE_PATH=${ARTIFACTS_PATH}/${INDEX_NAME}

export SEARCH_PATH=${ARTIFACTS_PATH}/${SEARCH_RUN_NAME}.search
export RERANK_PATH=${ARTIFACTS_PATH}/${RERANK_RUN_NAME}
export RERANK_FILE_PATH=${RERANK_PATH}/${RERANK_RUN_NAME}.rerank
export EXP_ANSWER_PATH=${ARTIFACTS_PATH}/${EXP_ANSWER_RUN_NAME}
export EXP_ANSWER_FILE_PATH=${EXP_ANSWER_PATH}/${EXP_ANSWER_RUN_NAME}.exp

export RGQE_SELF_PATH=${ARTIFACTS_PATH}/${RQE_RUN_NAME}_SELF
export RGQE_SELF_FILE_PATH=${RGQE_SELF_PATH}/${RQE_RUN_NAME}.rgqe_self
export RGQE_CC_FILE_PATH=${RGQE_SELF_PATH}/${RQE_RUN_NAME}.rgqe_cc

export RGQE_QUESTION_PATH=${ARTIFACTS_PATH}/${RQE_RUN_NAME}_QUESTION
export RGQE_QUESTION_FILE_PATH=${RGQE_QUESTION_PATH}/${RQE_RUN_NAME}.rgqe_question

export RGQE_TOP_PATH=${ARTIFACTS_PATH}/${RQE_RUN_NAME}_TOP
export RGQE_TOP_FILE_PATH=${RGQE_TOP_PATH}/${RQE_RUN_NAME}.rgqe_top
export RGQE_TOP_CC_FILE_PATH=${RGQE_TOP_PATH}/${RQE_RUN_NAME}.rgqe_top_cc

export RGQE_ALL_PATH=${ARTIFACTS_PATH}/${RQE_RUN_NAME}_ALL
export RGQE_ALL_FILE_PATH=${RGQE_ALL_PATH}/${RQE_RUN_NAME}.rgqe_all
export RGQE_ALL_RERANK_FILE_PATH=${RGQE_ALL_PATH}/${RQE_RUN_NAME}.rgqe_rerank

export RERANK_RUN_PATH=${RERANK_PATH}/${RERANK_RUN_NAME}.txt
export ANSWERS_PATH=${RERANK_PATH}/${RERANK_RUN_NAME}.answers

export RERANK_EVAL_PATH=${RERANK_PATH}/${RERANK_RUN_NAME}.eval

export RUN_PATH=${RGQE_ALL_PATH}/${RQE_RUN_NAME}.txt
export EVAL_PATH=${RGQE_ALL_PATH}/${RQE_RUN_NAME}.eval


if [[ ${CREATE_INDEX} = true ]]; then
    echo "Creating index..."
    # create dataset split
    python rerank/split_data.py \
      --label_path data/${COLLECTION}/prelim_judgments_corrected.json \
      --output_path ${DATASET_PATH}/split \
      --dataset ${DATASET}

    if [[ ${EXPAND_INDEX} = true ]]; then
        mkdir ${COLLECTION_JSONL_PATH}
        # setup dataset
        python search/extract_collection.py \
         --collection_path ${COLLECTION_PATH} \
         --output_path ${COLLECTION_JSONL_FILE_PATH}


        # expand dataset for indexing
        python expand_query/expand.py \
         --collection_path ${COLLECTION_JSONL_FILE_PATH} \
         --output_path ${EXP_DATA_PATH} \
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
          --input_path ${EXP_DATA_PATH} \
          --output_path ${EXP_DATA_FILE_PATH} \
          --num_processes 16

        mkdir ${EXP_DATA_PATH}/data_expanded
        python search/expand_passages_jsonl.py \
          --collection_path ${COLLECTION_JSONL_FILE_PATH} \
          --expand_path ${EXP_DATA_FILE_PATH} \
          --output_path ${EXP_DATA_PATH}/data_expanded/data_expanded.jsonl

        # index dataset
        python -m pyserini.index \
          -collection JsonCollection \
          -generator DefaultLuceneDocumentGenerator \
          -threads 12 \
          -input ${EXP_DATA_PATH}/data_expanded \
          -index ${INDEX_FILE_PATH} \
          -storePositions \
          -storeDocvectors \
          -storeRaw
    else
#        mkdir ${COLLECTION_JSONL_PATH}/data
#        python search/expand_passages_jsonl.py \
#          --collection_path ${COLLECTION_JSONL_FILE_PATH} \
#          --output_path ${COLLECTION_JSONL_PATH}/data/data.jsonl
        # index dataset
#        python search/convert_passages_to_json.py \
#          --collection_path ${DATASET_PATH}/data \
#          --json_collection_path ${DATASET_PATH}/data_json
        python -m pyserini.index \
          -collection JsonCollection \
          -generator DefaultLuceneDocumentGenerator \
          -threads 12 \
          -input ${DATASET_PATH}/data_json \
          -index ${INDEX_FILE_PATH} \
          -storePositions \
          -storeDocvectors \
          -storeRaw
    fi
fi

if [[ ${SEARCH_INDEX} = true ]]; then
    echo "Searching index..."
    # search dataset
    python search/search_index.py \
      --index_path ${INDEX_FILE_PATH} \
      --query_path ${QUERY_PATH} \
      --label_path ${JUDGEMENTS_PATH} \
      --output_path ${SEARCH_PATH} \
      --top_k ${SEARCH_TOP_K}
fi

if [[ ${TRAIN_RERANK} = true ]]; then
    echo "Training rerank model..."
    python rerank/rerank_train.py \
      --query_path ${QUERY_PATH} \
      --collection_path ${COLLECTION_PATH} \
      --passage_search_run ${SEARCH_PATH} \
      --label_path ${JUDGEMENTS_PATH} \
      --split_path ${DATASET_PATH}/split \
      --pre_model_name ${RERANK_PRE_MODEL_NAME} \
      --model_name ${RERANK_MODEL_NAME} \
      --max_seq_len 96 \
      --batch_size 32 \
      --negative_samples ${NEGATIVE_SAMPLES} \
      --add_all_labels \
      --weighted_loss \
      --learning_rate 5e-6 \
      --epochs 5
fi

if [[ ${RUN_RERANK} = true ]]; then
    echo "Running rerank model..."
    python rerank/rerank.py \
      --query_path ${QUERY_PATH} \
      --collection_path ${COLLECTION_PATH} \
      --passage_search_run ${SEARCH_PATH} \
      --label_path ${LABEL_PATH} \
      --output_path ${RERANK_PATH} \
      --pre_model_name ${RERANK_PRE_MODEL_NAME} \
      --model_name ${RERANK_MODEL_NAME} \
      --max_seq_len ${RERANK_SEQ_LEN} \
      --batch_size ${RERANK_BATCH_SIZE} \
      --load_trained_model \
    ; \
    python rerank/format_rerank.py \
      --input_path ${RERANK_PATH} \
      --output_path ${RERANK_FILE_PATH} \
    ; \
    python rerank/format_eval.py \
      --input_path ${RERANK_FILE_PATH} \
      --output_path ${RERANK_RUN_PATH} \
      --top_k 1000

    python rerank/extract_answers.py \
      --search_path ${RERANK_RUN_PATH} \
      --collection_path ${COLLECTION_PATH} \
      --output_path ${ANSWERS_PATH}
fi


if [[ ${EVAL_RERANK} = true ]]; then
    echo "Evaluating rerank model..."
    python rerank/epic_eval.py \
      ${LABEL_PATH} \
      ${RERANK_RUN_PATH} \
      rerank/.${DATASET}_ideal_ranking_scores.tsv \
      --task ${DATASET} \
      > ${RERANK_EVAL_PATH} \
      ; \
      tail -n 3 ${RERANK_EVAL_PATH} \
      | awk \
        '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
        END{ for (i in RtoC) print RtoC[i] }' \
      | tail -n 2
fi


if [[ ${RUN_EXPAND_ANSWERS} = true ]]; then
    echo "Expanding answers..."
    python expand_query/expand.py \
     --input_path ${RERANK_RUN_PATH} \
     --output_path ${EXP_ANSWER_PATH} \
     --collection_path ${COLLECTION_PATH} \
     --pre_model_name ${EXP_PRE_MODEL_NAME} \
     --model_name ${EXP_MODEL_NAME} \
     --top_k ${EXP_ANSWER_TOP_K} \
     --num_samples ${EXP_ANSWER_NUM_SAMPLES} \
     --batch_size ${EXP_ANSWER_BATCH_SIZE} \
     --max_seq_len ${EXP_ANSWER_SEQ_LEN} \
     --run_top_k ${RGQE_TOP_K} \
    ; \
    python expand_query/format_expand.py \
      --input_path ${EXP_ANSWER_PATH} \
      --output_path ${EXP_ANSWER_FILE_PATH}
fi

if [[ ${RUN_RGQE_SELF} = true ]]; then
    echo "Running self RGQE..."
    # self entailment
    python rgqe/rgqe.py \
      --input_path ${EXP_ANSWER_FILE_PATH} \
      --output_path ${RGQE_SELF_PATH} \
      --model_name ${RQE_MODEL_NAME} \
      --max_seq_len ${RGQE_SEQ_LEN} \
      --batch_size ${RGQE_BATCH_SIZE} \
      --mode self \
    ; \
    python rgqe/format_rgqe_self.py \
      --input_path ${RGQE_SELF_PATH} \
      --output_path ${RGQE_SELF_FILE_PATH} \
    ; \
    python rgqe/rgqe_self_components.py \
      --input_path ${RGQE_SELF_FILE_PATH} \
      --expand_path ${EXP_ANSWER_FILE_PATH} \
      --output_path ${RGQE_CC_FILE_PATH} \
      --threshold ${RGQE_SELF_THRESHOLD}
fi

if [[ ${RUN_RGQE_QUESTION} = true ]]; then
    echo "Running question RGQE..."
    # query-question entailment to filter out bad generated questions
    python rgqe/rgqe.py \
      --input_path ${RGQE_CC_FILE_PATH} \
      --output_path ${RGQE_QUESTION_PATH} \
      --search_path ${RERANK_RUN_PATH} \
      --query_path ${QUERY_PATH} \
      --label_path ${LABEL_PATH} \
      --model_name ${RQE_MODEL_NAME} \
      --max_seq_len ${RGQE_SEQ_LEN} \
      --batch_size ${RGQE_BATCH_SIZE} \
      --top_k ${RGQE_TOP_K} \
      --mode question \
    ; \
    python rgqe/format_rgqe_question.py \
      --input_path ${RGQE_QUESTION_PATH} \
      --output_path ${RGQE_QUESTION_FILE_PATH}
fi

if [[ ${RUN_RGQE_TOP} = true ]]; then
    echo "Running top RGQE..."
    # top_k set entailment
    python rgqe/rgqe.py \
      --input_path ${RGQE_CC_FILE_PATH} \
      --output_path ${RGQE_TOP_PATH} \
      --qe_path ${RGQE_QUESTION_FILE_PATH} \
      --search_path ${RERANK_RUN_PATH} \
      --model_name ${RQE_MODEL_NAME} \
      --max_seq_len ${RGQE_SEQ_LEN} \
      --batch_size ${RGQE_BATCH_SIZE} \
      --mode top \
      --top_k ${RGQE_TOP_K} \
      --threshold ${RQE_TOP_THRESHOLD} \
    ; \
    python rgqe/format_rgqe_top.py \
      --input_path ${RGQE_TOP_PATH} \
      --output_path ${RGQE_TOP_FILE_PATH} \
    ; \
    python rgqe/rgqe_top_components.py \
      --input_path ${RGQE_TOP_FILE_PATH} \
      --cc_path ${RGQE_CC_FILE_PATH} \
      --output_path ${RGQE_TOP_CC_FILE_PATH} \
      --threshold ${RGQE_TOP_C_THRESHOLD}
fi


if [[ ${RUN_RGQE_RERANK} = true ]]; then
    echo "Running RGQE rerank..."
    python rgqe/rgqe_rerank.py \
      --cc_path ${RGQE_TOP_CC_FILE_PATH} \
      --answers_path ${ANSWERS_PATH} \
      --queries_path ${QUERY_PATH} \
      --output_path ${RGQE_ALL_RERANK_FILE_PATH} \
      --threshold ${RGQE_ALL_THRESHOLD} \
      --ratio ${RGQE_RATIO} \
    ; \
    python rgqe/format_eval.py \
      --results_path ${RGQE_ALL_RERANK_FILE_PATH} \
      --output_path ${RUN_PATH}
fi

if [[ ${EVAL_RGQE} = true ]]; then
    echo "Evaluating RGQE model..."
    python rerank/epic_eval.py \
      ${LABEL_PATH} \
      ${RUN_PATH} \
      rerank/.${DATASET}_ideal_ranking_scores.tsv \
      --task ${DATASET} \
      > ${EVAL_PATH} \
      ; \
      tail -n 3 ${EVAL_PATH} \
      | awk \
        '{ for (i=1; i<=NF; i++) RtoC[i]= (RtoC[i]? RtoC[i] FS $i: $i) }
        END{ for (i in RtoC) print RtoC[i] }' \
      | tail -n 2
fi

