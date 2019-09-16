#!/bin/bash

PYTHON=${PYTHON:-python3}
set -e

# $PYTHON gen_tfrecords.py \
#     --image_dir data/validation/ \
#     --output_prefix output/val_human_parts \
#     --image_info_file output/validation_human_parts.csv \
#     --classes_file output/classes_human_parts.csv \
#     --num_shards=1

$PYTHON gen_tfrecords.py \
    --image_dir data/train/ \
    --min_samples_per_class=56391 \
    --output_prefix output/balanced_train_human_parts \
    --image_info_file output/train_human_parts.csv \
    --classes_file output/classes_human_parts.csv \
    --num_shards=10

MIN_SAMPLES=(338 698 1489 8932 45938)

for i in {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

#     $PYTHON gen_tfrecords.py \
#         --image_dir data/validation/ \
#         --output_prefix output/val_part_$i \
#         --image_info_file output/validation_part_$i.csv \
#         --classes_file output/classes_part_${i}_of_5.csv \
#         --num_shards=1

#         --display_only=True \
    $PYTHON gen_tfrecords.py \
        --image_dir data/train/ \
        --min_samples_per_class=${MIN_SAMPLES[i]} \
        --output_prefix output/balanced_train_part_$i \
        --image_info_file data/challenge-2019-train-detection-bbox.csv \
        --classes_file output/classes_part_${i}_of_5.csv \
        --num_shards=10
done

