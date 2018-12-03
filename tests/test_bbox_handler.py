"""Test set for bounding box handling utility classes.
"""

import numpy as np
import tensorflow as tf

import project_root

from src.utils.bbox_handler import (AnchorConverter, compute_iou,
                                    generate_anchor_priors, ltrb_to_xywh,
                                    xywh_to_ltrb)


class Test(tf.test.TestCase):

    def test_generate_anchor_targets(self):
        """Test it can generate anchor targets as desired.
        """
        with self.test_session():
            # Regression.
            grid_size = (1, 1)
            num_anchors = 1
            anchor_priors = np.array([15., 15., 30., 30.]).reshape(
                [grid_size[0], grid_size[1], num_anchors, 4]) / 100.

            # Single anchor and target.
            gt_bboxes = tf.constant([1., 1., 31., 31., 400], shape=[1, 5
                                                                   ]) / 100.
            dut = AnchorConverter(anchor_priors, iou_threshold=.5)
            anchor_targets = dut.generate_anchor_targets(gt_bboxes)
            self.assertAllClose(
                anchor_targets,
                tf.constant([0.02, 0.02, 0., 0., 1., 4.],
                            shape=[grid_size[0], grid_size[1], num_anchors, 6]))

            # Assigned as non-object with a higher iou_threshold.
            # Still it's an object.
            dut = AnchorConverter(anchor_priors, iou_threshold=.8769)
            anchor_targets = dut.generate_anchor_targets(gt_bboxes)
            self.assertAllClose(
                anchor_targets,
                tf.constant([0.02, 0.02, 0., 0., 1., 4.],
                            shape=[grid_size[0], grid_size[1], num_anchors, 6]))
            # Not an object.
            dut = AnchorConverter(anchor_priors, iou_threshold=.8770)
            anchor_targets = dut.generate_anchor_targets(gt_bboxes)
            self.assertAllClose(
                anchor_targets,
                tf.constant([0., 0., 0., 0., 0., 0.],
                            shape=[grid_size[0], grid_size[1], num_anchors, 6]))

    def test_encode_regression(self):
        """Test it can encode regression targets.
        """
        with self.test_session():
            grid_size = (1, 1)
            num_anchors = 1
            anchor_priors = np.array([15., 15., 30., 30.]).reshape(
                [grid_size[0], grid_size[1], num_anchors, 4]) / 100.
            dut = AnchorConverter(anchor_priors)

            # Single assigned index.
            target_locations_xywh = tf.constant([16., 16., 30., 30.],
                                                shape=[1, 4]) / 100.
            assigned = dut.encode_regression(target_locations_xywh,
                                             tf.constant([0]))
            self.assertAllClose(
                assigned,
                tf.constant([0.02, 0.02, 0., 0.], shape=[num_anchors, 4]))

            # Multiple assigned indices.
            num_anchors = 2
            anchor_priors = np.array([[15., 15., 30., 30.], [
                15., 15., 30., 30.
            ]]).reshape([grid_size[0], grid_size[1], num_anchors, 4]) / 100.
            dut = AnchorConverter(anchor_priors)
            target_locations_xywh = tf.constant(
                [[16., 16., 30., 30.], [15., 15., 29., 29.]], shape=[2, 4
                                                                    ]) / 100.
            assigned = dut.encode_regression(target_locations_xywh,
                                             tf.constant([0, 1]))
            self.assertAllClose(
                assigned,
                tf.constant(
                    [[0.02, 0.02, 0., 0.], [0., 0., -0.033902, -0.033902]],
                    shape=[num_anchors, 4]))

    def test_compute_iou(self):
        """Test it can compute IoU properly.
        """
        with self.test_session():
            # Regression tests on two boxes.
            box1 = tf.constant([15., 15., 30., 30.], shape=[1, 4])
            box2 = tf.constant([30., 25., 30., 30.], shape=[1, 4])
            iou = compute_iou(box1, box2)
            self.assertAllEqual(iou, tf.constant([.2], shape=[1, 1]))

            # Single vs multiple boxes.
            box1 = tf.constant([15., 15., 30., 30.], shape=[1, 4])
            box2 = tf.constant([[30., 25., 30., 30.], [35., 25., 30., 30.]],
                               shape=[2, 4])
            iou = compute_iou(box1, box2)
            self.assertAllEqual(iou, tf.constant([[.2, .125]], shape=[1, 2]))
            iou = compute_iou(box2, box1)
            self.assertAllEqual(iou, tf.constant([[.2, .125]], shape=[2, 1]))

            # Disjoint boxes return 0 IoU.
            # Non-disjoint.
            box1 = tf.constant([15.00001, 15.00001, 30., 30.], shape=[1, 4])
            box2 = tf.constant([45., 45., 30., 30.], shape=[1, 4])
            iou = compute_iou(box1, box2)
            self.assertAllGreater(iou, 0.)
            # Disjoint.
            box1 = tf.constant([15., 15., 30., 30.], shape=[1, 4])
            box2 = tf.constant([45., 45., 30., 30.], shape=[1, 4])
            iou = compute_iou(box1, box2)
            self.assertAllEqual(iou, tf.constant([0.], shape=[1, 1]))
            # Far away disjoint.
            box1 = tf.constant([15., 15., 30., 30.], shape=[1, 4])
            box2 = tf.constant([80., 80., 30., 30.], shape=[1, 4])
            iou = compute_iou(box1, box2)
            self.assertAllEqual(iou, tf.constant([0.], shape=[1, 1]))

    def test_ltrb_xywh_conversions(self):
        """Test box location coordinate conversions.
        """
        with self.test_session():
            # Regression tests on a single box.
            ltrb = tf.constant([0., 0., 100., 100.], shape=[1, 4])
            xywh = ltrb_to_xywh(ltrb)
            self.assertAllEqual(
                xywh, tf.constant([50., 50., 100., 100.], shape=[1, 4]))

            ltrb_recon = xywh_to_ltrb(xywh)
            self.assertAllEqual(ltrb, ltrb_recon)

            # Multiple boxes.
            ltrb = tf.constant([[0., 0., 100., 100.], [50., 50., 50., 50.]],
                               shape=[2, 4])
            ltrb_recon = xywh_to_ltrb(ltrb_to_xywh(ltrb))
            self.assertAllEqual(ltrb, ltrb_recon)

            # Handle negative value boxes.
            ltrb = tf.constant([-100., -100., 100., 100.], shape=[1, 4])
            ltrb_recon = xywh_to_ltrb(ltrb_to_xywh(ltrb))
            self.assertAllEqual(ltrb, ltrb_recon)

    def test_generate_anchor_priors(self):
        """Check that it can generate expected anchors.
        """
        # Regression.
        gt = np.array([0.5, 0.5, 1.414214, 2.828427])
        grid_size = (1, 1)
        scale = (2,)
        aspect_ratio = (2.,)
        anchor_priors = generate_anchor_priors(grid_size, scale, aspect_ratio)
        np.testing.assert_array_almost_equal(np.squeeze(anchor_priors), gt)

        # Multiple scale and aspect ratio.
        gt = np.array([[0.5, 0.5, 2.82842712, 1.41421356],
                       [0.5, 0.5, 1.41421356, 2.82842712],
                       [0.5, 0.5, 5.65685425, 2.82842712],
                       [0.5, 0.5, 2.82842712, 5.65685425],
                       [0.5, 0.5, 11.3137085, 5.65685425],
                       [0.5, 0.5, 5.65685425, 11.3137085]])
        scale = (2, 4, 8)
        aspect_ratio = (.5, 2.)
        anchor_priors = generate_anchor_priors(grid_size, scale, aspect_ratio)
        np.testing.assert_array_almost_equal(np.squeeze(anchor_priors), gt)

        # With grid.
        gt = np.array([[[0.25, 0.25, .707107, 1.414214],
                        [0.75, 0.25, .707107, 1.414214]],
                       [
                           [0.25, 0.75, .707107, 1.414214],
                           [0.75, 0.75, .707107, 1.414214],
                       ]])
        grid_size = (2, 2)
        scale = (2,)
        aspect_ratio = (2.,)
        anchor_priors = generate_anchor_priors(grid_size, scale, aspect_ratio)
        np.testing.assert_array_almost_equal(np.squeeze(anchor_priors), gt)
