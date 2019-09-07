
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
    parser.add_argument('--gen_classes_stats', help='generate classes stats', action='store_true')
    parser.add_argument('--gen_human_parts', help='generate human parts classes', action='store_true')
    parser.add_argument('--gen_six_levels', help='generate 6 levels of classes', action='store_true')
    args = parser.parse_args()


    if args.gen_classes_stats or args.gen_six_levels:
        classes_df = pd.read_csv(args.classes, header=None, names=['classes', 'names'])
        print('classes_df:', classes_df.shape)

        df = pd.read_csv(args.bboxes)
        dprint(df.head())

        dprint(df.LabelName.nunique())
        assert df.LabelName.nunique() >= classes_df.shape[0]

        counts_df = pd.DataFrame(df.groupby('LabelName').ImageID.nunique())
        counts_df.columns = ['counts']
        dprint(counts_df.head())

        counts_df = classes_df.join(counts_df, on='classes')
        counts_df = counts_df.sort_values('counts')
        dprint(counts_df)
        counts_df.to_csv('output/classes_stats.csv', index=False)


    human_parts_classes = pd.read_csv('extra/class-ids-human-body-parts-and-mammal.txt',
                                      header=None).iloc[:, 0].values
    dprint(human_parts_classes)

    human_parts_classes_df = classes_df[classes_df.iloc[:, 0].isin(human_parts_classes)]
    dprint(human_parts_classes_df)
    human_parts_classes_df.to_csv('output/classes_human_parts.csv', header=False, index=False)

    human_parts_ids = pd.read_csv('extra/train-image-ids-with-human-parts-and-mammal-boxes.txt',
                                  header=None).iloc[:, 0].values
    dprint(human_parts_ids)


    if args.gen_human_parts:
        df = pd.read_csv('data/challenge-2019-train-detection-bbox.csv')
        df = df[df.ImageID.isin(human_parts_ids)]

        dprint(df.shape)
        df.to_csv('output/train_human_parts.csv', index=False)


    # remove human body part classes as they are not marked everywhere
    dprint(counts_df.shape)
    counts_df = counts_df[~counts_df.classes.isin(human_parts_classes)]
    dprint(counts_df.shape)

    if args.gen_six_levels:
        parts = [0, 100, 200, 300, 400, 450]
        counts_df = counts_df.drop(columns='counts')

        for i, (part, part_end) in enumerate(zip(parts, parts[1:])):
            print(f'classes {part} to {part_end}')
            filename = f'output/classes_part_{i}_of_{len(parts) - 1}.csv'

            classes_part_df = counts_df.iloc[part : part_end]
            classes_part_df.to_csv(filename, header=False, index=False)
