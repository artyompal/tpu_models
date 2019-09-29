# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw Open Images dataset to TFRecord for object_detection.

Example usage:
    python convert_to_tfrecords.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
import hashlib
import io
import multiprocessing
import os
import random
import sys

from collections import Counter, defaultdict
from functools import partial
from typing import Any, Tuple

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import PIL.Image

from tqdm import tqdm

from research.object_detection.utils import dataset_util
from research.object_detection.utils import label_map_util

import tensorflow as tf


flags.DEFINE_boolean('include_masks', False,
                     'Whether to include instance segmentations masks '
                     '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('image_dir', '', 'Directory containing images.')
flags.DEFINE_string('image_dir2', '', 'Additional directory with images.')
flags.DEFINE_string('image_info_file', '', 'File containing image information')
flags.DEFINE_string('classes_file', '', 'CSV file with allowed classes')
flags.DEFINE_string('classes_replace_table', '', 'CSV file with class mapping')
flags.DEFINE_string('output_prefix', '', 'Path to output file')
flags.DEFINE_integer('num_shards', 10, 'Number of shards for output file.')
flags.DEFINE_integer('min_samples_per_class', 0, 'Minimum number of samples per class.')
flags.DEFINE_boolean('display_only', False, 'Don\'t write any file, just show what will be done.')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image_df, image2idx):
  """Converts image and annotations to a tf.Example proto.

  Args - OLD DESCRIPTION FOR THE REFERENCE, IGNORE IT.
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official Open Images dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    category_index: a dict containing Open Images category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    caption_annotations:
      list of dict with keys: [u'id', u'image_id', u'str'].
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_id = image_df.ImageID.values[0]

  # some settings here
  bbox_annotations = True
  include_masks = False

  filename = image_id + '.jpg'

  full_path = os.path.join(FLAGS.image_dir, filename)
  if not os.path.exists(full_path):
    full_path = os.path.join(FLAGS.image_dir2, filename)

  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  pil_image = PIL.Image.open(full_path)
  image_height = pil_image.height
  image_width = pil_image.width

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  key = hashlib.sha256(encoded_jpg).hexdigest()

  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image2idx[image_id]).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
  }

  # num_annotations_skipped = 0
  if bbox_annotations:
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    # encoded_mask_png = []

    for ann in image_df.itertuples():
      xmin.append(ann.XMin)
      xmax.append(ann.XMax)
      ymin.append(ann.YMin)
      ymax.append(ann.YMax)

      # is_crowd.append(object_annotations['iscrowd'])
      is_crowd.append(bool(ann.IsGroupOf))

      # category_id = int(object_annotations['category_id'])
      category_id = class_indices[ann.LabelName]
      # print(category_id)
      category_ids.append(category_id)

      category_name = class_labels[ann.LabelName].encode('utf8')
      # print(category_name)
      category_names.append(category_name)

      # area.append(object_annotations['area'])
      area.append(abs((ann.XMax - ann.XMin) * (ann.YMax - ann.YMin)))

      # if include_masks:
      #   run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
      #                                       image_height, image_width)
      #
      #   binary_mask = mask.decode(run_len_encoding)
      #
      #   if not object_annotations['iscrowd']:
      #     binary_mask = np.amax(binary_mask, axis=2)
      #
      #   pil_image = PIL.Image.fromarray(binary_mask)
      #   output_io = io.BytesIO()
      #   pil_image.save(output_io, format='PNG')
      #   encoded_mask_png.append(output_io.getvalue())

    feature_dict.update({
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    })

    # if include_masks:
    #   feature_dict['image/object/mask'] = (
    #       dataset_util.bytes_list_feature(encoded_mask_png))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example # key, example, num_annotations_skipped


def get_classes_stats(all_samples):
  print('gathering sample stats')
  stats = None

  for sample_df in all_samples:
    counter = Counter(sample_df.LabelName.values)
    stats = stats + counter if stats else counter # type: ignore

  imbalance = max(stats.values()) / min(stats.values())
  return stats, imbalance

def _load_images_info(images_info_file, classes):
  df = pd.read_csv(images_info_file)

  if classes is not None:
    print('annotations before filtering:', df.shape)
    print(df.head())

    df = df[df.LabelName.isin(classes)]

    print('annotations after filtering:', df.shape)
    print(df.head())

  return df

def _create_tf_record_from_oid_annotations(
    images_info_file,
    output_path,
    num_shards,
    include_masks=False,
    classes=None):
  """Loads Open Images annotation csv files and converts to tf.Record format.

  Args:
    images_info_file: CSV file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val annotation csv files.
    output_path: Path to output tfrecord file.
    num_shards: Number of output files to create.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    classes: np.array of classes or None
  """

  if not FLAGS.display_only:
    tf.logging.info('writing to the output path: %s', output_path)
    writers = [tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                           (i, num_shards)) for i in range(num_shards)]

  df = _load_images_info(images_info_file, classes)

  unique_ids = sorted(df.ImageID.unique())
  image2idx = {image: i for i, image in enumerate(unique_ids)}
  unique_ids_count = len(unique_ids)


  num_classes = df.LabelName.nunique()

  stats, imbalance = get_classes_stats([s for _, s in tqdm(df.groupby('ImageID'),
                                                           total=df.ImageID.nunique())])
  print('class imbalance before:', imbalance, stats)
  print('total samples before:', df.ImageID.nunique())

  # get class count column
  counts_df = pd.DataFrame(df.groupby('LabelName').ImageID.nunique())
  counts_df.columns = ['count']
  counts_df.sort_values('count', inplace=True)
  df = df.join(counts_df, on='LabelName')
  df = df.sort_values('count')
  print(df.head())

  # find the least frequent class for every sample
  least_freq_cls_df = pd.DataFrame(df.groupby('ImageID').LabelName.first())
  least_freq_cls_df.columns = ['LeastFreqClass']
  print(least_freq_cls_df.head())
  df = df.join(least_freq_cls_df, on='ImageID')
  print(df.head())

  # group by the least frequent class, then group by sample
  all_samples = []
  samples_per_class = defaultdict(int)


  for class_, class_df in tqdm(df.groupby('LeastFreqClass'), total=num_classes):
    samples = [sample_df for _, sample_df in class_df.groupby('ImageID')]

    def add_samples(samples_list):
      all_samples.extend(samples_list)

      for sample in samples_list:
          for label in sample.LabelName.values:
              samples_per_class[label] += 1

    samples_needed = max(FLAGS.min_samples_per_class - samples_per_class[class_], 0)

    if len(samples) >= samples_needed:
      # I always add all samples, maybe I shouldn't
      add_samples(samples)
    else:
      quot = samples_needed // len(samples)
      mod = samples_needed % len(samples)

      add_samples(samples * quot)

      if mod:
        add_samples(random.sample(samples, mod))

  random.shuffle(all_samples)


  stats, imbalance = get_classes_stats(all_samples)
  print('class imbalance after:', imbalance, stats)
  print('total samples after:', len(all_samples))

  if FLAGS.display_only:
    return


  print('writing tfrecords')

  # Multiprocessing implementation is actually slower because we're SSD-bound here,
  # so more random reads we do, the poorer performance will be.
  #
  # pool = multiprocessing.Pool()
  # for idx, tf_example in enumerate(tqdm(pool.imap(partial(create_tf_example, image2idx=image2idx),
  #                                                 df.groupby('ImageID')),
  #                                  total=unique_ids_count)):
  #     writers[idx % num_shards].write(tf_example.SerializeToString())

  for idx, sample_df in enumerate(tqdm(all_samples)):
    tf_example = create_tf_example(sample_df, image2idx=image2idx)
    writers[idx % num_shards].write(tf_example.SerializeToString())

  # pool.close()
  # pool.join()

  for writer in writers:
    writer.close()


  tf.logging.info('Finished writing')


def main(_):
  assert FLAGS.image_dir, '"image_dir" is missing.'
  assert FLAGS.output_prefix, '"output_prefix" is missing.'
  assert FLAGS.image_info_file, 'annotation file is missing.'
  assert FLAGS.classes_file, 'classes file is missing.'

  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file

  global classes_df, class_indices, class_labels, classes
  classes_df = pd.read_csv(FLAGS.classes_file, header=None, names=['classes', 'names'])
  class_indices = {row[1]: row[0] + 1 for row in classes_df.itertuples()}
  class_labels = {row[1]: row[2] for row in classes_df.itertuples()}
  classes = classes_df.classes.values

  directory = os.path.dirname(FLAGS.output_prefix)
  if not tf.gfile.IsDirectory(directory):
    tf.gfile.MakeDirs(directory)

  _create_tf_record_from_oid_annotations(
      images_info_file,
      FLAGS.output_prefix,
      FLAGS.num_shards,
      include_masks=FLAGS.include_masks,
      classes=classes)


if __name__ == '__main__':
  classes_df = None
  class_indices = dict()
  class_labels = dict()
  classes = None

  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
