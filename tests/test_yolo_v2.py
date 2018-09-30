"""Test set for YOLOV2 classes.
"""

import unittest

import numpy as np
import tensorflow as tf

import project_root

from src.utils.detection_losses import yolo_detection_loss
from src.models.yolo_v2 import YOLOV2


class Test(unittest.TestCase):
    NUM_CLASSES = 3
    NUM_ANCHORS = 1

    def test_network_update(self):
        """Test networks surely updates the parameters.
        cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
        """
        MODELS = (YOLOV2,)
        np.random.seed(1234)
        tf.set_random_seed(1234)

        for model in MODELS:
            image_height = model.GRID_H * model.SCALE
            image_width = model.GRID_W * model.SCALE
            with tf.Graph().as_default():
                with tf.device("/cpu:0"):
                    dummy_in = tf.placeholder(
                        tf.float32, (None, image_height, image_width, 3))
                    dummy_gt = tf.placeholder(
                        tf.float32, (1, model.GRID_H, model.GRID_W, 1, 6))

                with tf.device("/gpu:0"):
                    dut = model(self.NUM_CLASSES, self.NUM_ANCHORS)
                    out = dut(dummy_in)

                with tf.device("/cpu:0"):
                    loss = yolo_detection_loss(out, dummy_gt)

                with tf.device("/gpu:0"):
                    # Use large learning rate to avoid vanishing gradient (especially for PSPNet).
                    optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate=1.e40)
                    grads = optimizer.compute_gradients(
                        loss, var_list=tf.trainable_variables())
                    train_op = optimizer.apply_gradients(grads)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    target_np = np.zeros((1, model.GRID_H, model.GRID_W, 1, 6))
                    target_np[0, 5, 5, 0, 4] = np.array([1.])
                    target_np[0, 5, 5, 0, 5:] = np.array([1.])

                    before = sess.run(tf.trainable_variables())
                    sess.run(
                        train_op,
                        feed_dict={
                            dummy_in:
                            np.random.rand(1, image_height, image_width, 3),
                            dummy_gt:
                            target_np
                        })
                    after = sess.run(tf.trainable_variables())
                    for i, (b, a) in enumerate(zip(before, after)):
                        # Make sure something changed.
                        # assert (b != a).any()
                        assert not np.allclose(b, a)
