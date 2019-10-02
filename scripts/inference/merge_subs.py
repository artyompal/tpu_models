#!/usr/bin/python3.6
# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import multiprocessing
import sys

import numpy as np

from tqdm import tqdm
from collections import defaultdict

import pyximport; pyximport.install()
from soft_nms import cpu_nms, cpu_soft_nms


def apply_nms(row, thresh_iou, soft_nms=False, sigma=0.5, thresh_score=1e-3, method=2, verbose=False):
    #          0      1    2   3   4   5
    # row = [label,score, x1, y1, x2, y2, ...]

    boxes_by_class = defaultdict(list)
    res = []
    for i, cls in enumerate(row[0::6]):
        bbox = [float(f) for f in row[i*6+1: i*6+6]]
        boxes_by_class[cls].append(bbox)

    for cls, bboxes in boxes_by_class.items():
        bboxes_arr = np.array(bboxes, dtype=np.float32)
        # swap: first coordinates, then score
        bboxes_arr[:,[4,0,1,2,3]] = bboxes_arr[:,[0,1,2,3,4]]

        if soft_nms:
            nms_bboxes = cpu_soft_nms(bboxes_arr, Nt=thresh_iou, sigma=sigma,
                                      threshold=thresh_score, method=method)

        else:
            keep_idx = cpu_nms(bboxes_arr, thresh_iou)
            nms_bboxes = bboxes_arr[keep_idx]
            if verbose:
                print(cls, keep_idx)

        # swap back: first score, then coordinates
        nms_bboxes[:,[0,1,2,3,4]] = nms_bboxes[:,[4,0,1,2,3]]
        if verbose:
            print(cls, nms_bboxes)

        for i in range(nms_bboxes.shape[0]):
            str_bboxes = [f'{bbox:.5f}' for bbox in nms_bboxes[i]]
            res.extend([cls] + str_bboxes)

    return res


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} result.csv sub1.csv sub2.csv ...')
        sys.exit()

    fnames = sys.argv[2:]
    print('subs to combine:', fnames)

    lines = []
    for fn in tqdm(fnames):
        with open(fn, 'r') as f_dest:
            l = f_dest.readlines()

            # chop off the header, sort and append
            lines.append(sorted(l[1:]))


    subs_num = len(lines)
    rows_num = len(lines[0])
    thresh_iou = 0.3        # IoU threshold for method = 1
    sigma = 0.5             # default
    thresh_score = 2e-2     # score threshold, can't be zero
    method = 2              # default
    soft_nms = True
    verbose = False


    def process_line(lines):
        row = []

        for line in lines:
            file_name, row_ = line.strip().split(',')
            row.extend(row_.split())

        new_row = apply_nms(row, thresh_iou=thresh_iou, soft_nms=soft_nms, sigma=sigma,
                            thresh_score=thresh_score, method=method, verbose=verbose)
        return file_name + ',' + ' '.join(new_row) + '\n'


    pool = multiprocessing.Pool()
    lines = list(zip(*lines))
    res = list(tqdm(pool.imap(process_line, lines), total=rows_num))

    with open(sys.argv[1], 'w') as f_dest:
        f_dest.write('ImageId,PredictionString\n')
        f_dest.writelines(res)
