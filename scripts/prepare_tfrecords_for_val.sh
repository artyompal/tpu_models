#!/bin/bash

export PYTHONPATH="$HOME/src/models:$HOME/src/models/research"
python3 convert_to_tfrecords.py --image_dir data/validation/ --output_prefix val --image_info_file data/challenge-2019-validation-detection-bbox.csv --num_shards=10

