#!/bin/bash

PYTHON=${PYTHON:-python3}
set -e


if [ "$#" -ne 1 ]; then
    echo "usage: $0 fold_num"
    exit
fi


FOLD_NUM=$1
DISPLAY_ONLY=0

gsutil cp gs://new_tpu_storage/converted/train_human_parts_fold_$FOLD_NUM.csv output/
gsutil cp gs://new_tpu_storage/converted/train_boxes_fold_$FOLD_NUM.csv output/

$PYTHON gen_tfrecords.py \
    --display_only=$DISPLAY_ONLY \
    --image_dir data/train/ \
    --min_samples_per_class=100000 \
    --output_prefix output/train_human_parts_fold_$FOLD_NUM \
    --image_info_file output/train_human_parts_fold_$FOLD_NUM.csv \
    --classes_file output/classes_human_parts.csv \
    --num_shards=10

if [ $DISPLAY_ONLY -eq 0 ]
then
    gsutil -m cp output/train_human_parts_fold_$FOLD_NUM*.tfrecord gs://new_tpu_storage/converted/human_parts/
    rm output/train_human_parts_fold_$FOLD_NUM*.tfrecord
fi


MIN_SAMPLES=(338 0 0 0 0)

for i in {0..4}
do
    echo "========================================================================"
    echo "processing part $i"
    PART="part_$i"

    NUM_SAMPLES=${MIN_SAMPLES[i]}

    $PYTHON gen_tfrecords.py \
        --display_only=$DISPLAY_ONLY \
        --image_dir data/train/ \
        --min_samples_per_class=$NUM_SAMPLES \
        --output_prefix output/train_part_${i}_fold_$FOLD_NUM \
        --image_info_file output/train_boxes_fold_$FOLD_NUM.csv \
        --classes_file output/classes_part_${i}_of_5.csv \
        --num_shards=20

    if [ $DISPLAY_ONLY -eq 0 ]
    then
        gsutil -m cp output/train_part_${i}_fold_$FOLD_NUM*.tfrecord gs://new_tpu_storage/converted/$PART/
        rm output/train_part_${i}_fold_$FOLD_NUM*.tfrecord
    fi
done

