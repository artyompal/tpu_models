#!/bin/bash

set -e
mkdir -p output

PYTHON=${PYTHON:-python3.6}


$PYTHON add_data_from_testset.py \
    output/train_boxes_with_pseudo_labels.csv \
    output/train_boxes.csv \
    output/best_submission.csv
