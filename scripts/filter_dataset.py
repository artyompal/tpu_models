#!/usr/bin/python3.6

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output filename, csv format', type=str)
    parser.add_argument('input', help='description file, csv format', type=str)
    parser.add_argument('--remove_groups', help='remove IsGroup', default=True, type=bool)
    parser.add_argument('--remove_depicted', help='remove IsDepicted', action='store_true')
    parser.add_argument('--remove_inside', help='remove IsInside', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    print('df.shape before after cleaning', df.shape)

    if args.remove_groups:
        df = df[df.IsGroupOf == 0]

    if args.remove_depicted:
        df = df[df.IsDepiction == 0]

    if args.remove_inside:
        df = df[df.IsInside == 0]

    print('df.shape after cleaning', df.shape)
    df.to_csv(args.output)
