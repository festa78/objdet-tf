"""Test set for image processing utils.
"""

import numpy as np
import tensorflow as tf

import project_root

from src.utils.image_processing import (random_flip_left_right_image_and_label,
                                        random_crop_image_and_label,
                                        resize_image_and_label)


class Test(tf.test.TestCase):

    def test_random_flip_left_right_image_and_label(self):
        """Test flipping.
        """
        with self.test_session():
            image = np.zeros((10, 10, 3))
            image[:, 5:, :] = 1
            image = tf.convert_to_tensor(image, dtype=tf.int64)
            label = np.array([
                [.5, .3, .8, .8, 0],
            ])
            label = tf.convert_to_tensor(label, dtype=tf.float32)

            # No flipping.
            image_flip, label_flip = random_flip_left_right_image_and_label(
                image, label, 0.)
            self.assertAllEqual(image, image_flip)
            self.assertAllClose(label, label_flip)

            # With flipping.
            image_gt = np.zeros((10, 10, 3))
            image_gt[:, :5, :] = 1
            image_gt = tf.convert_to_tensor(image_gt, dtype=tf.int64)
            label_gt = np.array([
                [.2, .3, .5, .8, 0],
            ])
            label_gt = tf.convert_to_tensor(label_gt, dtype=tf.float32)
            image_flip, label_flip = random_flip_left_right_image_and_label(
                image, label, 1.)
            self.assertAllEqual(image_gt, image_flip)
            self.assertAllClose(label_gt, label_flip)

    def test_random_crop_image_and_label(self):
        """Test cropping image and label.
        """
        with self.test_session():
            image = np.zeros((10, 10, 3))
            image[:, 5:, :] = 1
            image = tf.convert_to_tensor(image, dtype=tf.int64)
            label = np.array([
                [.4, .4, .6, .5, 0],
            ])
            label = tf.convert_to_tensor(label, dtype=tf.float32)

            # Cropping without affecting box sizes.
            image_crop, label_crop = random_crop_image_and_label(
                image, label, tf.constant([6, 6]))
            self.assertAllEqual(tf.shape(image_crop)[:2], tf.constant([6, 6]))
            self.assertAllClose(label_crop[0, 2] - label_crop[0, 0], .2)
            self.assertAllClose(label_crop[0, 3] - label_crop[0, 1], .1)

            # Cropping will affect box sizes.
            label = np.array([
                [.8, .8, 1.2, 1.5, 0],
            ])
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            image_crop, label_crop = random_crop_image_and_label(
                image, label, tf.constant([10, 10]))
            self.assertAllEqual(tf.shape(image_crop)[:2], tf.constant([10, 10]))
            self.assertAllClose(label_crop[0, 2] - label_crop[0, 0], .2)
            self.assertAllClose(label_crop[0, 3] - label_crop[0, 1], .2)

            # Cropping will remove boxes.
            label = np.array([
                [1., 1., 1.2, 1.5, 0],
            ])
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            image_crop, label_crop = random_crop_image_and_label(
                image, label, tf.constant([10, 10]))
            self.assertAllEqual(tf.shape(image_crop)[:2], tf.constant([10, 10]))
            self.assertAllEqual(tf.shape(label_crop)[0], 0)
