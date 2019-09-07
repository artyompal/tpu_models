#!/bin/bash

for i in human_parts {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

    PART="part_$i"

    gsutil -m cp output/train_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
    gsutil -m cp output/val_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
    gsutil -m cp output/validation_$PART.json gs://ap_tpu_storage/converted/$PART/
done
