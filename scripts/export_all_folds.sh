#!/bin/bash

set -e

./export_fold.sh human_parts 2.0.6 0 24000
./export_fold.sh human_parts 2.0.6 1 30000
./export_fold.sh human_parts 2.0.6 2 10000
./export_fold.sh human_parts 2.0.6 3 32000
./export_fold.sh human_parts 2.0.6 4 26000

./export_fold.sh part_0 2.0.4 0 60000
./export_fold.sh part_0 2.0.4 1 65000
./export_fold.sh part_0 2.0.4 2 50000
./export_fold.sh part_0 2.0.4 3 60000
./export_fold.sh part_0 2.0.4 4 55000

./export_fold.sh part_1 2.0.4 0 50000
./export_fold.sh part_1 2.0.4 1 50000
./export_fold.sh part_1 2.0.4 2 55000
./export_fold.sh part_1 2.0.4 3 55000
./export_fold.sh part_1 2.0.4 4 50000

./export_fold.sh part_2 2.0.4 0 120000
./export_fold.sh part_2 2.0.4 1 125000
./export_fold.sh part_2 2.0.4 2 115000
./export_fold.sh part_2 2.0.4 3 115000
./export_fold.sh part_2 2.0.4 4 125000

./export_fold.sh part_3 2.0.4 0 120000
./export_fold.sh part_3 2.0.4 1 140000
./export_fold.sh part_3 2.0.4 2 130000
./export_fold.sh part_3 2.0.4 3 135000
./export_fold.sh part_3 2.0.4 4 135000

./export_fold.sh part_4 2.0.4 0 150000
./export_fold.sh part_4 2.0.4 1 155000
./export_fold.sh part_4 2.0.4 2 150000
./export_fold.sh part_4 2.0.4 3 105000
./export_fold.sh part_4 2.0.4 4 180000

