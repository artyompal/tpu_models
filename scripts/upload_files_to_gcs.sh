#!/bin/bash

for PART in human_parts part_0 part_1 part_2 part_3 part_4
do
    echo "========================================================================"
    echo processing $PART

    gsutil -m cp output/balanced_train_$PART*.tfrecord gs://new_tpu_storage/converted/$PART/
    rm output/balanced_train_$PART*.tfrecord 

    gsutil -m cp output/val_$PART*.tfrecord gs://new_tpu_storage/converted/$PART/
    rm output/val_$PART*.tfrecord 

    gsutil -m cp output/validation_$PART.json gs://new_tpu_storage/converted/$PART/
done
