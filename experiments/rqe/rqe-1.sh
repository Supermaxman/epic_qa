#!/usr/bin/env bash

filename=$(basename -- "$0")
# run names
RUN_ID=${filename::-3}
RUN_NAME=HLTRI_COVID_RQE
DATASET=q_hier

RQE_PRE_MODEL_NAME=models/quora-lm-models-mt-dnn-base-uncased
RQE_MODEL_NAME="RQE-${DATASET}-${RUN_NAME}_${RUN_ID}"
RQE_MODEL_TYPE=lm
RQE_BATCH_SIZE=8
RQE_NEG_SAMPLES=6000
RQE_LEARNING_RATE=5e-5
RQE_EPOCHS=10
RQE_SAVE_DIRECTORY=models/rqe

RQE_NUM_GPUS=1

RQE_TRAIN=true
RQE_EVAL=false

export TOKENIZERS_PARALLELISM=true

echo "Starting experiment ${RUN_NAME}_${RUN_ID}"
echo "Reserving ${RQE_NUM_GPUS} GPU(s)..."
RQE_GPUS=$(python gpu/request_gpus.py -r ${RQE_NUM_GPUS})
if [[ ${RQE_GPUS} -eq -1 ]]; then
    echo "Unable to reserve ${RQE_NUM_GPUS} GPU(s), exiting."
    exit 1
fi
echo "Reserved ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"
RQE_TRAIN_GPUS=${RQE_GPUS}
RQE_EVAL_GPUS=${RQE_GPUS}

handler()
{
    echo "Experiment aborted."
    echo "Freeing ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"
    python gpu/free_gpus.py -i "${RQE_GPUS}"
    exit 1
}
trap handler SIGINT

if [[ ${RQE_TRAIN} = true ]]; then
  python -m rqe.rqe_train \
    --dataset=${DATASET} \
    --pre_model_name=${RQE_PRE_MODEL_NAME} \
    --model_name="${RQE_MODEL_NAME}" \
    --model_type=${RQE_MODEL_TYPE} \
    --load_model \
    --learning_rate ${RQE_LEARNING_RATE} \
    --batch_size=${RQE_BATCH_SIZE} \
    --save_directory=${RQE_SAVE_DIRECTORY} \
    --neg_samples=${RQE_NEG_SAMPLES} \
    --epochs ${RQE_EPOCHS} \
    --gpus "${RQE_TRAIN_GPUS}"
fi


if [[ ${RQE_EVAL} = true ]]; then
  python -m rqe.rqe_eval \
    --dataset=${DATASET} \
    --model_name="${RQE_MODEL_NAME}" \
    --pre_model_name=${RQE_PRE_MODEL_NAME} \
    --model_type=${RQE_MODEL_TYPE} \
    --batch_size=${RQE_BATCH_SIZE} \
    --save_directory=${RQE_SAVE_DIRECTORY} \
    --gpus "${RQE_EVAL_GPUS}"
fi


echo "Freeing ${RQE_NUM_GPUS} GPUs: ${RQE_GPUS}"
python gpu/free_gpus.py -i "${RQE_GPUS}"
