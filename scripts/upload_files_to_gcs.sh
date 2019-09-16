#!/bin/bash

for PART in part_2 # human_parts # part_3 
do
    echo "========================================================================"
    echo processing $PART

    gsutil -m cp output/balanced_train_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
    rm output/balanced_train_$PART*.tfrecord 

#     gsutil -m cp output/val_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
#     gsutil -m cp output/validation_$PART.json gs://ap_tpu_storage/converted/$PART/
done
