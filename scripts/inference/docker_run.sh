#!/bin/bash

# docker run --runtime=nvidia \
docker run -u $(id -u):$(id -g) --runtime=nvidia \
    -e CUDA_VISIBLE_DEVICES \
    --shm-size 12G \
    -v `pwd`:/app \
    -v /mnt/ssd_fast/open_images:/app/data \
    -v `readlink -f ../best_models`:/app/best_models \
    -v `readlink -f ../predictions`:/app/predictions \
    --rm \
    -it open_images_objdet "$@"

