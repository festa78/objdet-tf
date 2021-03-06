#!/usr/bin/python3 -B
"""The script to make TFRecord datasets from {image, bounding box} pair file paths.
"""

import argparse

import project_root

from src.data.voc import get_file_path, parse_label_files
from src.data.tfrecord import write_tfrecord

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to make TFRecord datasets from {image, bounding box} pair file paths."
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help=
        'Path to the voc data directory. The directory should have JPEGImages/ and Annotations/ subfolders.'
    )
    parser.add_argument(
        'train_list',
        type=str,
        help=
        'Path to the voc train data file list. The each row in the file should contain a file basename.'
    )
    parser.add_argument(
        'val_list',
        type=str,
        help=
        'Path to the voc validation data file list. The each row in the file should contain a file basename.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to directory to save the created .tfrecord data.')
    options = parser.parse_args()

    with open(options.train_list) as f:
        train_list = f.read().splitlines()
    with open(options.val_list) as f:
        val_list = f.read().splitlines()
    data_list = get_file_path(options.input_dir, train_list, val_list)
    for v in data_list.values():
        v['labels'] = parse_label_files(v['label_list'])
    write_tfrecord(data_list, options.output_dir, normalize=True)
