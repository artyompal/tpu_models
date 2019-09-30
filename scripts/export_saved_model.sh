#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 dataset_code config_version step"
    exit
fi


set -e

PART=$1
VERSION=$2
STEP=$3

PYTHONPATH=$HOME/tpu_models/models

STORAGE_BUCKET=gs://new_tpu_storage
CHECKPOINT_PATH=$STORAGE_BUCKET/saved/$VERSION-$PART/model.ckpt-$STEP

VAL_DATASET=$PART
REGEX="v[0-9]+_(.+)"

if [[ $VAL_DATASET =~ $REGEX ]]
then
    VAL_DATASET="${BASH_REMATCH[1]}"
fi

echo "using validation dataset: $VAL_DATASET"

VAL_JSON_FILE=${STORAGE_BUCKET}/converted/$VAL_DATASET/validation_$VAL_DATASET.json
gsutil cp $VAL_JSON_FILE output/
LOCAL_VAL_JSON_FILE="output/validation_$VAL_DATASET.json"

NUM_CLASSES=$(cat $LOCAL_VAL_JSON_FILE | grep name | wc -l)
((NUM_CLASSES++)) # add background class


python ../models/official/detection/export_saved_model.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_normalized_coordinates=True --export_dir $STORAGE_BUCKET/final/$VERSION-$PART/ \
    --params_override="{
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
    --config_file `find ../models/official/detection/configs/yaml/ -name $VERSION*.yaml`
