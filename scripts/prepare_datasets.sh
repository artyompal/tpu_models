#!/bin/bash

set -e
mkdir -p output

PYTHON=${PYTHON:-python3}

$PYTHON filter_dataset.py \
    output/validation_boxes.csv \
    data/challenge-2019-validation-detection-bbox.csv
$PYTHON filter_dataset.py \
    output/train_boxes.csv \
    data/challenge-2019-train-detection-bbox.csv
$PYTHON filter_dataset.py \
    output/train_boxes__no_inside.csv \
    data/challenge-2019-train-detection-bbox.csv \
    --remove_inside
$PYTHON filter_dataset.py \
    output/train_boxes__no_pics__no_inside.csv \
    data/challenge-2019-train-detection-bbox.csv \
    --remove_depicted --remove_inside

$PYTHON split_classes.py \
    data/challenge-2019-train-detection-bbox.csv \
    data/challenge-2019-classes-description-500.csv

$PYTHON build_validation.py \
    output/val_human_parts.csv \
    output/validation_boxes.csv \
    output/classes_human_parts.csv \
    5 \
    --viz_directory=output/human_parts_val


# ./build_validation.py validation_leaf_443.csv validation_boxes__no_grp.csv /home/cppg/dev/kaggle/open_images_2019/tpu_models/scripts/classes_leaf_443.csv 2
#
# for i in {0..5}
# do
#     ./build_validation.py \
#         validation_part_${i}.csv validation_boxes__no_grp.csv classes_part_${i}_of_6.csv 5
# done
