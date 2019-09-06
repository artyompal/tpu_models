#!/bin/bash

gsutil -m cp output/train_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp output/val_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp output/val_human_parts.json gs://ap_tpu_storage/converted/human_parts/
