#!/bin/bash

PYTHON=${PYTHON:-python3}
PYTHONPATH=$PYTHONPATH:$HOME/dev/frameworks/tensorflow_models:$HOME/dev/frameworks/tensorflow_models/research

# python3.6 gen_tfrecords.py \
#     --image_dir data/validation/ \
#     --output_prefix output/val_human_parts \
#     --image_info_file output/val_human_parts.csv \
#     --classes_file output/classes_human_parts.csv \
#     --num_shards=10

$PYTHON gen_tfrecords.py \
    --image_dir data/train/ \
    --output_prefix output/train_human_parts \
    --image_info_file output/train_human_parts.csv \
    --classes_file output/classes_human_parts.csv \
    --num_shards=10
