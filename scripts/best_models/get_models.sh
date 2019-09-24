#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 destination config_version"
    exit
fi

gsutil -m cp -r gs://new_tpu_storage/final/$2* $1
