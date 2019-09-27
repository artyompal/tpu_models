#!/bin/bash

PYTHON=${PYTHON:-python3}
set -e


DISPLAY_ONLY=0
VERSION="v4"


PART=human_parts
PREFIX=output/train_${VERSION}_$PART

$PYTHON gen_tfrecords.py \
    --display_only=$DISPLAY_ONLY \
    --image_dir data/train/ \
    --output_prefix $PREFIX \
    --image_info_file output/train_human_parts.csv \
    --classes_file output/classes_human_parts.csv \
    --num_shards=10

if [ $DISPLAY_ONLY -eq 0 ]
then
    gsutil -m cp $PREFIX*.tfrecord gs://new_tpu_storage/converted/${VERSION}_$PART/
    rm $PREFIX*.tfrecord
fi


for i in {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

    PART=part_$i
    PREFIX=output/train_${VERSION}_$PART

    $PYTHON gen_tfrecords.py \
        --display_only=$DISPLAY_ONLY \
        --image_dir data/train/ \
        --output_prefix $PREFIX \
        --image_info_file output/train_boxes.csv \
        --classes_file output/classes_part_${i}_of_5.csv \
        --num_shards=20

    if [ $DISPLAY_ONLY -eq 0 ]
    then
        gsutil -m cp $PREFIX*.tfrecord gs://new_tpu_storage/converted/${VERSION}_$PART/
        rm $PREFIX*.tfrecord
    fi
done
