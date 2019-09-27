#!/bin/bash

gcloud compute scp --project=tpuproj-245020 "$@" new2-5:~
