#!/usr/bin/python3.6

import argparse
import os

from typing import Any, Dict, List
from glob import glob

import numpy as np
import pandas as pd

from debug import dprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output filename', type=str)
    parser.add_argument('input', help='description file, csv format', type=str)
    parser.add_argument('classes', help='classes list, in csv format with header', type=str)
    parser.add_argument('num_samples', help='number of samples per class', type=int)
    parser.add_argument('--viz_directory', help='directory to save debug symlinks', type=str)
    args = parser.parse_args()


    blacklist = [os.path.splitext(os.path.basename(f))[0] for f in glob('data/blacklist/*.jpg')]

    classes_df = pd.read_csv(args.classes)
    classes = classes_df.classes.values
    dprint(classes.shape)

    classes_desc_df = pd.read_csv('data/challenge-2019-classes-description-500.csv', header=None)
    classes_desc = {row[1]: row[2] for row in classes_desc_df.itertuples()}

    df = pd.read_csv(args.input)
    dprint(df.shape)
    dprint(df.head())

    df = df[df.LabelName.isin(classes)]
    df = df[~df.ImageID.isin(blacklist)]

    selected_dfs = []
    # dprint(df)

    for label_name, label_df in df.groupby('LabelName'):
        samples = label_df.ImageID.unique()
        samples = samples[:args.num_samples]

        samples_df = label_df[label_df.ImageID.isin(samples)]
        selected_dfs.append(samples_df)

    dprint(len(selected_dfs))
    df = pd.concat(selected_dfs)

    dprint(classes_df.shape)
    dprint(df.ImageID.unique().shape)

    dprint(df.shape)
    dprint(df.head())
    df.to_csv(args.output, index=None)


    if args.viz_directory:
        os.makedirs(args.viz_directory, exist_ok=True)

        for f in glob(args.viz_directory + '/*.jpg'):
            os.remove(f)

        for label_name, label_df in df.groupby('LabelName'):
            for image_id in label_df.ImageID.unique():
                os.symlink(os.path.abspath(f'validation/{image_id}.jpg'),
                           f'{args.viz_directory}/{classes_desc[label_name]}_{image_id}.jpg')
