#!/usr/bin/python3 -B
"""The script to train VOC2007 data by YOLOV2.
"""

import argparse
from datetime import datetime
from functools import partial
import logging
logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import tensorflow as tf
import yaml

import project_root

from src.data.data_preprocessor import DataPreprocessor
from src.data.tfrecord import read_tfrecord
from src.data.voc import id2trainId
from src.models.yolo_v2 import YOLOV2
from src.pipelines.trainer import Trainer
from src.utils.bbox_handler import convert_labelid, AnchorConverter, generate_anchor_priors
from src.utils.detection_losses import yolo_detection_loss
from src.utils.image_processing import (random_flip_left_right_image_and_label,
                                        random_crop_image_and_label,
                                        resize_image_and_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "The script to train voc data by a fully convolutional network (aka FCN)."
    )
    parser.add_argument(
        'params_yaml',
        type=str,
        help=
        'Path to the parameter setting yaml file. Examples in scripts/params')

    args = parser.parse_args()
    options = yaml.load(open(args.params_yaml))
    print(yaml.dump(options))

    # Initializations.
    tf.set_random_seed(1234)
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    save_dir_fullpath = os.path.expanduser(options['save_dir'])
    if not os.path.isdir(save_dir_fullpath):
        raise AttributeError('--save_dir should be existing directory path.')
    save_dir_fullpath = os.path.join(save_dir_fullpath, str(datetime.now()))
    os.makedirs(save_dir_fullpath)
    logging.info('Created save directory to {}'.format(save_dir_fullpath))

    resume_fullpath = None
    if options['resume_path']:
        resume_fullpath = os.path.expanduser(options['resume_path'])
    finetune_from = None
    if options['finetune_from']:
        finetune_from = options['finetune_from']

    # Data part should live in cpu.
    with tf.device('/cpu:0'):
        grid_size = (YOLOV2.GRID_H, YOLOV2.GRID_W)
        anchor_priors = generate_anchor_priors(
            grid_size, options['anchor_scale'], options['anchor_aspect_ratio'])
        anchor_converter = AnchorConverter(anchor_priors)

        # Read from tfrecord format data made by src.data.tfrecord.TFRecordWriter.
        train_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'train_*.tfrecord'))
        train_data_processor = DataPreprocessor(
            dataset=train_dataset,
            anchor_converter=anchor_converter,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=options['batch_size'],
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'])

        val_dataset = read_tfrecord(
            os.path.join(options['tfdata_dir'], 'val_*.tfrecord'))
        val_data_processor = DataPreprocessor(
            dataset=val_dataset,
            anchor_converter=anchor_converter,
            num_parallel_calls=options['num_parallel_calls'],
            batch_size=1,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1)

        # Pre-process training data.
        # Add more pre-procesing blocks.
        logging.info('Preprocess train data')

        logging.info('Convert label ontology from Id to trainId.')
        id2trainid_tensor = partial(convert_labelid, lookup_table=id2trainId)
        train_data_processor.process_label(id2trainid_tensor)

        if options['train_resized_height'] and options['train_resized_width']:
            logging.info('Resize image to ({}, {})'.format(
                options['train_resized_height'],
                options['train_resized_width']))
            train_data_processor.process_image_and_label(
                resize_image_and_label,
                size=tf.constant((options['train_resized_height'],
                                  options['train_resized_width'])))
        if options['random_saturation'] > 0. and options[
                'random_saturation'] < 1.:
            logging.info('Randomly adjust saturation by factor {}'.format(
                options['random_saturation']))
            lower = 1. - options['random_saturation']
            upper = 1. + options['random_saturation']
            train_data_processor.process_image(
                tf.image.random_saturation, lower=lower, upper=upper)
        if options['random_contrast'] > 0. and options['random_contrast'] < 1.:
            logging.info('Randomly adjust contrast by factor {}'.format(
                options['random_contrast']))
            lower = 1. - options['random_contrast']
            upper = 1. + options['random_contrast']
            train_data_processor.process_image(
                tf.image.random_contrast, lower=lower, upper=upper)
        if options['random_hue'] > 0. and options['random_hue'] < .5:
            logging.info('Randomly adjust hue by factor {}'.format(
                options['random_hue']))
            train_data_processor.process_image(
                tf.image.random_hue, max_delta=options['random_hue'])
        if options['random_brightness'] > 0. and options[
                'random_brightness'] < 1.:
            logging.info('Randomly adjust brightness by factor {}'.format(
                options['random_brightness']))
            train_data_processor.process_image(
                tf.image.random_brightness,
                max_delta=options['random_brightness'])

        if options['train_crop_height'] and options['train_crop_width']:
            logging.info('Random crop image and label to ({}, {})'.format(
                options['train_crop_height'], options['train_crop_width']))
            train_data_processor.process_image_and_label(
                random_crop_image_and_label,
                crop_size=(options['train_crop_height'],
                           options['train_crop_width']))

        if options['random_flip_left_right']:
            logging.info('Randomly horizontal flip image and label.')
            train_data_processor.process_image_and_label(
                random_flip_left_right_image_and_label)

        logging.info('Squash image pixel scale to [0,1].')
        train_data_processor.process_image(lambda image: image / 255.)

        train_iterator = train_data_processor.get_iterator()

        # Pre-process validation data.
        logging.info('Preprocess validation data')
        if options['val_resized_height'] and options['val_resized_width']:
            logging.info('Resize validation image to ({}, {})'.format(
                options['val_resized_height'], options['val_resized_width']))
            val_data_processor.process_image_and_label(
                resize_image_and_label,
                size=tf.constant((options['val_resized_height'],
                                  options['val_resized_width'])))

        logging.info('Convert label ontology from Id to trainId.')
        val_data_processor.process_label(id2trainid_tensor)

        logging.info('Squash image pixel scale to [0,1].')
        val_data_processor.process_image(lambda image: image / 255.)

        val_iterator = val_data_processor.get_iterator()

        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        num_anchors = len(options['anchor_scale']) * len(
            options['anchor_aspect_ratio'])
        model = YOLOV2(options['num_classes'], num_anchors)

        # TODO: Add more optimizers.
        learning_rate = tf.train.polynomial_decay(
            learning_rate=options['learning_rate'],
            global_step=global_step,
            decay_steps=options['num_epochs'] * options['num_train_samples'] /
            options['batch_size'],
            end_learning_rate=options['end_learning_rate'],
            power=options['power'])
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=learning_rate)

        trainer = Trainer(
            model=model,
            num_classes=options['num_classes'],
            train_num_batch=options['batch_size'],
            val_num_batch=1,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            anchor_converter=anchor_converter,
            loss_fn=yolo_detection_loss,
            optimizer=optimizer,
            global_step=global_step,
            save_dir=save_dir_fullpath,
            resume_path=resume_fullpath,
            num_epochs=options['num_epochs'],
            evaluate_epochs=options['evaluate_epochs'],
            finetune_from=finetune_from,
            verbose_steps=options['verbose_steps'])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        trainer.train(sess)
