
import argparse
import os

from glob import glob

import numpy as np
import pandas as pd

from debug import dprint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output filename', type=str)
    parser.add_argument('input', help='description file, csv format', type=str)
    parser.add_argument('classes', help='classes list, in csv format with header', type=str)
    parser.add_argument('--num_samples', help='number of samples per class', type=int)
    parser.add_argument('--viz_directory', help='directory to save debug symlinks', type=str)
    args = parser.parse_args()


    with open('blacklist.txt') as f:
        blacklist = [os.path.splitext(name)[0] for name in f]

    classes_df = pd.read_csv(args.classes, header=None, names=['classes', 'names'])
    classes = classes_df.classes.values
    dprint(classes.shape)

    class_names = {row[1]: row[2] for row in classes_df.itertuples()}

    df = pd.read_csv(args.input)
    dprint(df.shape)
    dprint(df.head())

    df = df[df.LabelName.isin(classes)]
    df = df[~df.ImageID.isin(blacklist)]

    selected_dfs = []

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
    df.to_csv(args.output, index=False)


    if args.viz_directory:
        os.makedirs(args.viz_directory, exist_ok=True)

        for img_file in glob(args.viz_directory + '/*.jpg'):
            os.remove(img_file)

        for label_name, label_df in df.groupby('LabelName'):
            for image_id in label_df.ImageID.unique():
                os.symlink(os.path.abspath(f'data/validation/{image_id}.jpg'),
                           f'{args.viz_directory}/{class_names[label_name]}_{image_id}.jpg')
