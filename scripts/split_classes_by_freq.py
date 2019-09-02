#!/usr/bin/python3.6

import argparse
import json
import os
import sys

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bboxes_file', help='a file called something_bboxes.csv', type=str)
    parser.add_argument('classes_file', help='classes csv file', type=str)
    parser.add_argument('num_splits', help='number of class groups', type=int)
    args = parser.parse_args()

    classes = pd.read_csv(args.classes_file).classes.values
    print('classes:', classes.shape)

    df = pd.read_csv(args.bboxes_file)
    df = df.loc[df.LabelName.isin(classes)]

    stats = df.groupby('LabelName').ImageID.nunique()
    stats = stats.sort_values()
    image_counts = [(label, count) for label, count in zip(stats.index, stats.values)]
    print(image_counts)
    print('number of classes:', len(image_counts))

    part_len = (len(image_counts) + args.num_splits - 1) // args.num_splits
    for part in range(0, len(image_counts), part_len):
        part_end = min(len(image_counts), part + part_len)
        print(f'classes {part} to {part_end}')

        part_classes = stats.index[part : part_end]
        part_counts = stats.values[part : part_end]

        print(part_classes)
        print(part_counts)

        filename = f'classes_part_{part//part_len}_of_{args.num_splits}.csv'
        pd.DataFrame({'classes': part_classes, 'counts': part_counts}).to_csv(filename, index=None)
