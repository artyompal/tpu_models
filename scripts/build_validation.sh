#!/bin/bash

./build_validation.py validation_leaf_443.csv validation_boxes__no_grp.csv /home/cppg/dev/kaggle/open_images_2019/tpu_models/scripts/classes_leaf_443.csv 2

for i in {0..5}
do 
    ./build_validation.py \
        validation_part_${i}.csv validation_boxes__no_grp.csv classes_part_${i}_of_6.csv 5
done
