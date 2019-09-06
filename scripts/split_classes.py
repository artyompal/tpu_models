#!/usr/bin/python3.6

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from debug import dprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bboxes', help='train bboxes, csv', type=str)
    parser.add_argument('classes', help='classes csv file', type=str)
    # parser.add_argument('num_splits', help='number of class groups', type=int)
    args = parser.parse_args()


    classes_df = pd.read_csv(args.classes, header=None)
    print('classes_df:', classes_df.shape)
    assert classes_df.shape[0] in [300, 500]

    df = pd.read_csv(args.bboxes)
    dprint(df.head())


    # generate human parts dataset
    human_parts_classes = pd.read_csv('class-ids-human-body-parts-and-mammal.txt',
                                      header=None).iloc[:, 0].values
    dprint(human_parts_classes)

    human_parts_classes_df = classes_df[classes_df.iloc[:, 0].isin(human_parts_classes)]
    dprint(human_parts_classes_df)
    human_parts_classes_df.to_csv('output/classes_human_parts.csv', header=False, index=False)

    human_parts_ids = pd.read_csv('train-image-ids-with-human-parts-and-mammal-boxes.txt',
                                  header=None).iloc[:, 0].values
    dprint(human_parts_ids)

    df = pd.read_csv('data/challenge-2019-train-detection-bbox.csv')
    df = df[df.ImageID.isin(human_parts_ids)]

    dprint(df.shape)
    df.to_csv('output/train_human_parts.csv', index=False)


'''
    part_len = (len(image_counts) + args.num_splits - 1) // args.num_splits
    for part in range(0, len(image_counts), part_len):
        part_end = min(len(image_counts), part + part_len)
        print(f'classes {part} to {part_end}')

        part_classes = stats.index[part : part_end]
        part_counts = stats.values[part : part_end]

        print(part_classes)
        print(part_counts)

        filename = f'classes_part_{part//part_len}_of_{args.num_splits}.csv'
        pd.DataFrame({'classes': part_classes, 'counts': part_counts}).to_csv(filename, header=None)
'''
