#!/bin/bash

gsutil -m cp train_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp val_human_parts*.tfrecord gs://ap_tpu_storage/converted/human_parts/
gsutil -m cp val_human_parts.json gs://ap_tpu_storage/converted/human_parts/
