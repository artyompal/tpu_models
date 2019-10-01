''' Universal inference script for a saved TF model. '''

import argparse
import os
import pickle

import tensorflow as tf
import pandas as pd

from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='TF SavedModel directory', type=str)
    parser.add_argument('part_idx', help='part index', type=int)
    parser.add_argument('num_parts', help='total number of parts', type=int)
    parser.add_argument('--destination', help='prediction file', type=str, default=None)

    args = parser.parse_args()
    assert args.part_idx >= 0 and args.part_idx < args.num_parts
    assert args.num_parts >= 2 and args.num_parts <= 10

    model_dir = args.directory[args.directory.find('models/'):]
    model_dir = glob(model_dir + '/*')[0]
    dirs = os.path.normpath(model_dir).split('/')
    assert len(dirs) > 1

    if args.destination is None:
        dest_filename = dirs[-2] + '_part_%d_of_%d.pkl' % (args.part_idx, args.num_parts)
    else:
        dest_filename = args.destination

    print('model_dir', model_dir, 'dest_filename', dest_filename)

    tf.enable_eager_execution()
    model = tf.saved_model.load_v2(model_dir, ['serve'])

    signature = model.signatures['serving_default']
    print('supported outputs:', signature.structured_outputs)

    data_dir = 'data/' if len(glob('data/*.csv')) else '../data/'

    sample_submission_df = pd.read_csv(data_dir + '/OBJDET_sample_submission.csv')
    image_ids = sample_submission_df['ImageId']

    part_size = image_ids.shape[0] // args.num_parts

    if args.part_idx != args.num_parts - 1:
        images_ids = image_ids.iloc[args.part_idx * part_size : (args.part_idx + 1) * part_size]
    else:
        images_ids = image_ids.iloc[args.part_idx * part_size :]


    predictions = []
    for image_id in tqdm(image_ids):
        img_filename = data_dir + '/test/' + image_id + '.jpg'

        with open(img_filename, 'rb') as f:
            data = f.read()
            pred = signature(tf.constant(data))
            predictions.append(pred)

    with open(dest_filename, 'wb') as f:
        pickle.dump(predictions, f)
