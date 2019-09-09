#!/bin/bash


set -e

PART=$1

USE_TPU=True
TPU_NAME=$HOSTNAME

PYTHONPATH=$HOME/tpu_models/models
STORAGE_BUCKET=gs://ap_tpu_storage
RESNET_CHECKPOINT=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602

MODEL_DIR=${STORAGE_BUCKET}/saved/1.0.3-$PART
TRAIN_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/train*.tfrecord
EVAL_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/val*.tfrecord
VAL_JSON_FILE=${STORAGE_BUCKET}/converted/$PART/validation_$PART.json

mkdir -p output
gsutil cp $VAL_JSON_FILE output/
LOCAL_VAL_JSON_FILE="output/validation_$PART.json"


NUM_STEPS_PER_EVAL=1000
EVAL_SAMPLES=$(cat $LOCAL_VAL_JSON_FILE | grep width | wc -l)
NUM_CLASSES=$(cat $LOCAL_VAL_JSON_FILE | grep name | wc -l)
((NUM_CLASSES++)) # add background class


python ../models/official/detection/main.py --use_tpu=$USE_TPU --tpu=$TPU_NAME \
    --num_cores=8 --model_dir=$MODEL_DIR --mode=train_and_eval \
    --params_override="{
        train: {
            checkpoint: {  
                path: $RESNET_CHECKPOINT, 
                prefix: resnet50/ 
            },
            train_file_pattern: $TRAIN_FILE_PATTERN
        }, 
        eval: { 
            val_json_file: $VAL_JSON_FILE, 
            eval_file_pattern: $EVAL_FILE_PATTERN, 
            eval_samples: $EVAL_SAMPLES, 
            num_steps_per_eval: $NUM_STEPS_PER_EVAL 
        },
        retinanet_head: {
            num_classes: $NUM_CLASSES,
        },
        retinanet_loss: {
            num_classes: $NUM_CLASSES,
        },
        postprocess: {
            num_classes: $NUM_CLASSES,
        }
    }" \
    --config_file ../models/official/detection/configs/yaml/1.0.3_constant.yaml


