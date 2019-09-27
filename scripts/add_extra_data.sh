#!/bin/bash

set -e
mkdir -p output

PYTHON=${PYTHON:-python3.6}


$PYTHON add_data_from_testset.py \
    output/train_boxes_with_pseudo_labels.csv \
    output/train_boxes.csv \
    output/best_submission.csv

$PYTHON add_data_from_testset.py \
    output/train_human_parts_with_pseudo_labels.csv \
    output/train_human_parts.csv \
    output/best_submission.csv \
    --classes output/classes_human_parts.csv
