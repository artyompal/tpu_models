#!/bin/bash


PART=$1

export TPU_NAME=$HOSTNAME

export PYTHONPATH=$HOME/dev/frameworks/tensorflow_models:$HOME/dev/frameworks/tensorflow_models/research:$HOME/dev/frameworks/tensorflow_models/research/slim:$HOME/tpu_models/models
export STORAGE_BUCKET=gs://ap_tpu_storage
export RESNET_CHECKPOINT=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602
export EVAL_SAMPLES=30
export NUM_STEPS_PER_EVAL=1000

export MODEL_DIR=${STORAGE_BUCKET}/saved/1.0.3-$PART
export TRAIN_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/train*
export EVAL_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/val*
export VAL_JSON_FILE=${STORAGE_BUCKET}/converted/$PART/validation_$PART.json


# tensorboard --logdir $MODEL_DIR &

python ../models/official/detection/main.py --use_tpu=True --tpu="${TPU_NAME?}" \
    --num_cores=8 --model_dir="${MODEL_DIR?}" --mode="train_and_eval" \
    --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, eval_samples: ${EVAL_SAMPLES?}, num_steps_per_eval: ${NUM_STEPS_PER_EVAL?} } }" \
    --config_file ../models/official/detection/configs/yaml/1.0.3_constant.yaml
