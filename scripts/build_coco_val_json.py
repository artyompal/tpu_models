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
    parser.add_argument('output', help='output filename', type=str)
    parser.add_argument('type', help='train|validation|test', type=str)
    parser.add_argument('classes', help='classes list, in csv format', type=str)
    args = parser.parse_args()
    assert args.type in ['train', 'validation', 'test']

    # read classes table
    classes = pd.read_csv(args.classes).classes.values
    print(classes.shape)

    # read labels table
    classes_df = pd.read_csv('data/challenge-2019-classes-description-500.csv', header=None)
    classes_df = classes_df[classes_df.iloc[:, 0].isin(classes)]
    print('number of classes:', classes_df.shape[0])
    assert classes_df.shape[0] == classes.size

    classes_table = {row[1]: row[0] + 1 for row in classes_df.itertuples()}
    print(dict(list(classes_table.items())[:10]))


    if args.type == 'train':
        df = pd.read_csv('data/challenge-2019-train-detection-bbox.csv')
    elif args.type == 'validation':
        df = pd.read_csv('data/challenge-2019-validation-detection-bbox.csv')
    else:
        assert False

    df = df[df.LabelName.isin(classes)]

    unique_ids = sorted(df.ImageID.unique())
    image2id = {image_id: i for i, image_id in enumerate(unique_ids)}


    # build categories table
    categories = []

    for row in classes_df.itertuples():
        cat = {}

        cat['id'] = row[0] + 1
        cat['name'] = row[2]
        cat['supercategory'] = 'object'

        categories.append(cat)


    # build images table
    images = {}

    for image_id in tqdm(unique_ids):
        path = f'data/{args.type}/{image_id}.jpg'
        img = Image.open(path)

        image = {}
        image['id'] = image2id[image_id]
        image['width'] = img.width
        image['height'] = img.height

        images[image_id] = image


    # make annotations in json
    annotations = []

    for i, row in enumerate(tqdm(df.itertuples(), total=df.shape[0])):
        # for i, row in enumerate(instances_df.itertuples()):
        #     bboxes.append(np.array([
        #         row.XMin * img.width,
        #         row.YMin * img.height,
        #         row.XMax * img.width,
        #         row.YMax * img.height]))
        #     labels.append(classes_table[row.LabelName])

        image = images[row.ImageID]
        x = int(row.XMin * image['width'])
        y = int(row.YMin * image['height'])
        w = int((row.XMax - row.XMin) * image['width'])
        h = int((row.YMax - row.YMin) * image['height'])

        ann: Dict[str, Any] = {}
        ann['id'] = i
        ann['image_id'] = image2id[row.ImageID]
        ann['category_id'] = classes_table[row.LabelName]
        ann['area'] = (row.XMax - row.XMin) * (row.YMax - row.YMin)
        ann['bbox'] = [x, y, w, h]
        ann['iscrowd'] = 0

        annotations.append(ann)
        # break


    out: Dict[str, Any] = {}
    out['categories'] = categories
    out['images'] = list(images.values())
    out['annotations'] = annotations

    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
