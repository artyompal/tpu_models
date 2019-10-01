#!/bin/bash

set -e

for part in human_parts part_0 part_1 part_2 part_3 part_4
do
    ./merge_subs.py sub_$part.csv ../csv/*$part*.csv
done
