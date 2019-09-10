#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 dataset_code config_version checkpoint_path "
    exit
fi


set -e

PART=$1
VERSION=$2
CHECKPOINT_PATH=$3

PYTHONPATH=$HOME/tpu_models/models

STORAGE_BUCKET=gs://ap_tpu_storage
VAL_JSON_FILE=${STORAGE_BUCKET}/converted/$PART/validation_$PART.json
gsutil cp $VAL_JSON_FILE output/
LOCAL_VAL_JSON_FILE="output/validation_$PART.json"

NUM_CLASSES=$(cat $LOCAL_VAL_JSON_FILE | grep name | wc -l)
((NUM_CLASSES++)) # add background class


python ../models/official/detection/export_saved_model.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_normalized_coordinates=True --export_dir gs://ap_tpu_storage/final/$VERSION-$PART/ \
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
