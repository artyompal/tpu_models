#!/bin/bash

gcloud config set project tpu-test-246716
ctpu up --tpu-size=v3-8 --machine-type n1-standard-8 --zone us-central1-a --name tpu3-1
ctpu up --tpu-size=v2-8 --machine-type n1-standard-8 --zone us-central1-b --name tpu2-1


# configuring:

ssh-keygen -t rsa -b 4096
cat .ssh/id_rsa.pub

git config --global core.editor "vim"
git clone git@github.com:artyompal/tpu_models.git

sudo apt-get install -y python-tk
pip install Cython matplotlib opencv-python-headless pyyaml Pillow
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

sudo /sbin/sysctl \
       -w net.ipv4.tcp_keepalive_time=60 \
       net.ipv4.tcp_keepalive_intvl=60 \
       net.ipv4.tcp_keepalive_probes=5


export TPU_NAME=tpu3-1

export PYTHONPATH=$HOME/dev/frameworks/tensorflow_models:$HOME/dev/frameworks/tensorflow_models/research:$HOME/dev/frameworks/tensorflow_models/research/slim:$HOME/tpu_models/models
export STORAGE_BUCKET=gs://ap_tpu_storage
export RESNET_CHECKPOINT=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602
export EVAL_SAMPLES=500
export NUM_STEPS_PER_EVAL=10000

# default config:
export PART=443_classes

# partial datasets config:
export PART=part_5
export MODEL_DIR=${STORAGE_BUCKET}/saved/1.0.3-$PART
export TRAIN_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/train*
export EVAL_FILE_PATTERN=${STORAGE_BUCKET}/converted/$PART/val*
export VAL_JSON_FILE=scripts/coco_validation_$PART.json

python models/official/detection/main.py --use_tpu=True --tpu="${TPU_NAME?}" \
    --num_cores=8 --model_dir="${MODEL_DIR?}" --mode="train_and_eval" \
    --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, eval_samples: ${EVAL_SAMPLES?}, num_steps_per_eval: ${NUM_STEPS_PER_EVAL?} } }" \
    --config_file models/official/detection/configs/yaml/1.0.3_constant.yaml

tensorboard --logdir gs://ap_tpu_storage/saved/1.0.0