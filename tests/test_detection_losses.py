"""Test set for detection loss computation.
"""

import numpy as np
import tensorflow as tf

import project_root

from src.utils.detection_losses import yolo_detection_loss


class Test(tf.test.TestCase):
    GRID_SIZE = 10
    NUM_CLASSES = 3

    def test_yolo_detection_loss(self):
        """Test to ensure it can properly compute detction loss
        for YOLO.
        """
        with tf.Graph().as_default():
            pred = tf.placeholder(
                tf.float32,
                (1, self.GRID_SIZE, self.GRID_SIZE, 1, 5 + self.NUM_CLASSES))
            target = tf.placeholder(tf.float32,
                                    (1, self.GRID_SIZE, self.GRID_SIZE, 1, 6))

            with self.test_session() as sess:
                loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss = yolo_detection_loss(
                    pred,
                    target,
                    weight_regression=1.,
                    weight_nonobjectness=1.,
                    return_all_losses=True)

                pred_np = np.zeros((1, self.GRID_SIZE, self.GRID_SIZE, 1,
                                    5 + self.NUM_CLASSES))
                pred_np[0, :, :, 0, 4] = np.array([-1000.])
                pred_np[0, 5, 5, 0, 4] = np.array([1000.])
                pred_np[0, :, :, 0, 5:] = np.array([1000., 0., 0.])
                pred_np[0, 5, 5, 0, 5:] = np.array([0., 1000., 0.])
                target_np = np.zeros((1, self.GRID_SIZE, self.GRID_SIZE, 1, 6))
                target_np[0, 5, 5, 0, 4] = np.array([1.])
                target_np[0, 5, 5, 0, 5:] = np.array([1.])

                # Only regression loss.
                pred_np[0, 5, 5, 0, :4] = np.array([.5, .5, 1., 1.])
                target_np[0, 5, 5, 0, :4] = np.array([.5, .5, .8, .8])
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, reg_val)

                # Only objectness loss.
                target_np[0, 5, 5, 0, :4] = np.array([.5, .5, 1., 1.])
                pred_np[0, :, :, 0, 4] = np.array([-1000.])
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, obj_val)

                # Only non-objectness loss.
                pred_np[0, 5, 5, 0, 4] = np.array([1000.])
                pred_np[0, 5, 6, 0, 4] = np.array([1000.])
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, nonobj_val)

                # Only classification loss.
                pred_np[0, 5, 6, 0, 4] = np.array([-1000.])
                pred_np[0, 5, 5, 0, 5:] = np.array([1000., 0., 0.])
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, cls_val)

                # Weights to adjust regression loss.
                pred_np[0, 5, 5, 0, 5:] = np.array([0., 1000., 0.])
                target_np[0, 5, 5, 0, :4] = np.array([.5, .5, .8, .8])
                loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss = yolo_detection_loss(
                    pred,
                    target,
                    weight_regression=.5,
                    weight_nonobjectness=1.,
                    return_all_losses=True)
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, reg_val / 2.)

                # Weights to adjust non-objectness loss.
                target_np[0, 5, 5, 0, :4] = np.array([.5, .5, 1., 1.])
                pred_np[0, 5, 6, 0, 4] = np.array([1000.])
                loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss = yolo_detection_loss(
                    pred,
                    target,
                    weight_regression=1.,
                    weight_nonobjectness=.5,
                    return_all_losses=True)
                all_val, reg_val, obj_val, nonobj_val, cls_val = sess.run(
                    (loss_all, reg_loss, obj_loss, nonobj_loss, cls_loss),
                    feed_dict={
                        pred: pred_np,
                        target: target_np
                    })
                self.assertAlmostEqual(all_val, nonobj_val / 2.)
