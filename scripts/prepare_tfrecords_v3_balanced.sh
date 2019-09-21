#!/bin/bash

PYTHON=${PYTHON:-python3}
set -e


DISPLAY_ONLY=0
VERSION="v3"
MIN_SAMPLES=(338 0 0 8000 100000)

# PREFIX=output/train_${VERSION}_human_parts
# $PYTHON gen_tfrecords.py \
#     --display_only=$DISPLAY_ONLY \
#     --image_dir data/train/ \
#     --min_samples_per_class=100000 \
#     --output_prefix $PREFIX \
#     --image_info_file output/train_human_parts.csv \
#     --classes_file output/classes_human_parts.csv \
#     --num_shards=10
#
# if [ $DISPLAY_ONLY -eq 0 ]
# then
#     gsutil -m cp $PREFIX*.tfrecord gs://ap_tpu_storage/converted/${VERSION}_$PART/
#     rm $PREFIX*.tfrecord
# fi


for i in 0 3 4
do
    echo "========================================================================"
    echo "processing part $i"

    NUM_SAMPLES=${MIN_SAMPLES[i]}
    PREFIX=output/train_${VERSION}_part_$i

    $PYTHON gen_tfrecords.py \
        --display_only=$DISPLAY_ONLY \
        --image_dir data/train/ \
        --min_samples_per_class=$NUM_SAMPLES \
        --output_prefix $PREFIX \
        --image_info_file output/train_boxes.csv \
        --classes_file output/classes_part_${i}_of_5.csv \
        --num_shards=20

    if [ $DISPLAY_ONLY -eq 0 ]
    then
        gsutil -m cp $PREFIX*.tfrecord gs://ap_tpu_storage/converted/${VERSION}_$PART/
        rm $PREFIX*.tfrecord
    fi
done
