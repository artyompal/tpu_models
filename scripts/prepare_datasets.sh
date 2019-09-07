#!/bin/bash

set -e
mkdir -p output

PYTHON=${PYTHON:-python3.6}


#
# filter datasets
#

# $PYTHON filter_dataset.py \
#     output/validation_boxes.csv \
#     data/challenge-2019-validation-detection-bbox.csv
# $PYTHON filter_dataset.py \
#     output/train_boxes.csv \
#     data/challenge-2019-train-detection-bbox.csv
# $PYTHON filter_dataset.py \
#     output/train_boxes__no_inside.csv \
#     data/challenge-2019-train-detection-bbox.csv \
#     --remove_inside
# $PYTHON filter_dataset.py \
#     output/train_boxes__no_pics__no_inside.csv \
#     data/challenge-2019-train-detection-bbox.csv \
#     --remove_depicted --remove_inside


#
# split classes
#

# $PYTHON build_leaf_classes_list.py 500
#
# $PYTHON split_classes.py \
#     --gen_six_levels \
#     --gen_human_parts \
#     data/challenge-2019-train-detection-bbox.csv \
#     output/classes_leaf_443.csv
#
# $PYTHON build_validation.py \
#     output/validation_human_parts.csv \
#     output/validation_boxes.csv \
#     output/classes_human_parts.csv \
#     --num_samples=5
#
# $PYTHON gen_coco_val_json.py \
#     output/validation_human_parts.json \
#     output/validation_human_parts.csv \
#     output/classes_human_parts.csv


#
# build the validation set
#

for i in 0 # {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

    $PYTHON build_validation.py \
        output/validation_part_${i}.csv \
        output/validation_boxes.csv \
        output/classes_part_${i}_of_5.csv \
        --num_samples=5
        # --viz_directory=output/val_part_$i

    $PYTHON gen_coco_val_json.py \
        output/validation_part_${i}.json \
        output/validation_part_${i}.csv \
        output/classes_part_${i}_of_5.csv
done
