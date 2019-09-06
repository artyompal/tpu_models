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
    parser.add_argument('output', help='output filename, csv format', type=str)
    parser.add_argument('input', help='description file, csv format', type=str)
    parser.add_argument('classes', help='classes list, csv format w/ header', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    classes_df = pd.read_csv(args.classes, header=None)
    print('classes_df:', classes_df.shape)
    assert classes_df.shape[0] in [300, 500]

    df = df[df.LabelName.isin(classes_df.iloc[:, 0])]
    df.to_csv(args.output, index=False)
