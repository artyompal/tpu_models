#!/bin/bash

PYTHON=${PYTHON:-python3}
set -e


if [ "$#" -ne 1 ]; then
    echo "usage: $0 fold_num"
    exit
fi


FOLD_NUM=$1

$PYTHON gen_tfrecords.py \
    --image_dir data/train/ \
    --min_samples_per_class=0 \
    --output_prefix output/train_human_parts_fold_${FOLD_NUM} \
    --image_info_file output/train_human_parts.csv \
    --classes_file output/classes_human_parts.csv \
    --num_shards=20 \
    --fold_num=$FOLD_NUM

gsutil -m cp output/train_human_parts_fold_${FOLD_NUM}*.tfrecord gs://ap_tpu_storage/converted/$PART/
rm output/train_human_parts_fold_${FOLD_NUM}*.tfrecord


MIN_SAMPLES=(338 0 0 0 0)

for i in {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

    NUM_SAMPLES=${MIN_SAMPLES[i]}

        # --display_only=True \
    $PYTHON gen_tfrecords.py \
        --image_dir data/train/ \
        --min_samples_per_class=$NUM_SAMPLES \
        --output_prefix output/train_part_${i}_fold_${FOLD_NUM} \
        --image_info_file data/challenge-2019-train-detection-bbox.csv \
        --classes_file output/classes_part_${i}_of_5.csv \
        --num_shards=20 \
        --fold_num=$FOLD_NUM

    gsutil -m cp output/train_part_${i}_fold_${FOLD_NUM}.tfrecord gs://ap_tpu_storage/converted/$PART/
    rm output/train_part_${i}_fold_${FOLD_NUM}.tfrecord
done
