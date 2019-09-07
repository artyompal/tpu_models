#!/bin/bash


gsutil -m cp output/train_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp output/val_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp output/val_human_parts.json gs://ap_tpu_storage/converted/human_parts/

for i in {0..4}
do
    echo "========================================================================"
    echo "processing part $i"

    PART="part_$i"

    gsutil -m cp output/train_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
    gsutil -m cp output/val_$PART*.tfrecord gs://ap_tpu_storage/converted/$PART/
    gsutil -m cp output/val_$PART.json gs://ap_tpu_storage/converted/$PART/
done
