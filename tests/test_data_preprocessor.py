"""Test set for DataPreprocessor class.
"""

import numpy as np
import tensorflow as tf

import project_root

from src.data.data_preprocessor import DataPreprocessor
from src.utils.bbox_handler import AnchorConverter, generate_anchor_priors


class Test(tf.test.TestCase):

    def test_process_image(self):
        """Test DataPreprocessor.process_image.
        """
        anchor_priors = generate_anchor_priors()
        anchor_converter = AnchorConverter(anchor_priors)
        dataset = tf.data.Dataset.from_tensor_slices({
            'image':
            tf.constant(np.zeros((3, 3, 3))),
            'label':
            tf.constant(np.zeros((3, 1)))
        })

        dut = DataPreprocessor(dataset, anchor_converter)

        # Lambda function with the external parameter @p a.
        dut.process_image(lambda image, a: image + a + 1, a=1)

        batch = dut.dataset.make_one_shot_iterator().get_next()

        image_equal_op = tf.equal(batch['image'],
                                  tf.constant(np.ones((3, 3)) * 2))
        label_equal_op = tf.equal(batch['label'], tf.constant(np.zeros((1,))))

        with self.test_session() as sess:
            for _ in range(3):
                for op in sess.run((image_equal_op, label_equal_op)):
                    assert np.all(op)

    def test_process_label(self):
        """Test DataPreprocessor.process_label.
        """
        anchor_priors = generate_anchor_priors()
        anchor_converter = AnchorConverter(anchor_priors)
        dataset = tf.data.Dataset.from_tensor_slices({
            'image':
            tf.constant(np.zeros((3, 3, 3))),
            'label':
            tf.constant(np.zeros((3, 1)))
        })

        dut = DataPreprocessor(dataset, anchor_converter)

        # Lambda function with the external parameter @p a.
        dut.process_label(lambda label, a: label + a + 1, a=1)

        batch = dut.dataset.make_one_shot_iterator().get_next()

        image_equal_op = tf.equal(batch['image'], tf.constant(np.zeros((3, 3))))
        label_equal_op = tf.equal(batch['label'], tf.constant(
            np.ones((1,)) * 2))

        with self.test_session() as sess:
            for _ in range(3):
                for op in sess.run((image_equal_op, label_equal_op)):
                    assert np.all(op)

    def test_process_image_and_label(self):
        """Test DataPreprocessor.process_image_and_label.
        """
        anchor_priors = generate_anchor_priors()
        anchor_converter = AnchorConverter(anchor_priors)
        dataset = tf.data.Dataset.from_tensor_slices({
            'image':
            tf.constant(np.zeros((3, 3, 3))),
            'label':
            tf.constant(np.zeros((3, 1)))
        })

        dut = DataPreprocessor(dataset, anchor_converter)

        # Lambda function with the external parameter @p a.
        dut.process_image_and_label(
            lambda image, label, a: (image - a - 1, label + a + 1), a=1)

        batch = dut.dataset.make_one_shot_iterator().get_next()

        image_equal_op = tf.equal(batch['image'],
                                  tf.constant(-np.ones((3, 3)) * 2))
        label_equal_op = tf.equal(batch['label'], tf.constant(
            np.ones((1,)) * 2))

        with self.test_session() as sess:
            for _ in range(3):
                for op in sess.run((image_equal_op, label_equal_op)):
                    assert np.all(op)
