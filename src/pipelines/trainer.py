"""Training pipeline for object detection.
"""

import logging
logging.basicConfig(level=logging.INFO)
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import project_root

from src.utils.bbox_handler import xywh_to_ltrb


class Trainer:
    """Basic training pipeline class which integrate
    model, data, loss, and optimizer and
    start object detection training.
    This class supposed to be instantiated on gpu.

    Parameters
    ----------
    model: object
        A semantic segmentation model object which
        has .__call__(input) method to get model output.

    num_classes: int
        The number of output classes of the model.

    train_num_batch: int
        The batch size of train inputs.

    val_num_batch: int
        The batch size of validation inputs.

    train_iterator: tf.Tensor
        The initializable iterator for training.
        .get_next() is used to create train batch operator.

    val_iterator: tf.Tensor
        The initializable iterator for validation.
        .get_next() is used to create validation batch operator.

    anchor_converter: AnchorConverter class
        The closs to encode/decode anchors.

    loss_fn: functional
        A functional which outputs loss value according to
        the same sized inputs tensors: predicted output
        tf.Tensor, ground truth output tf.Tensor,
        and weight tf.Tensor which weights losses
        on each pixel when conducting reduce mean operation.

    optimizer: tf.Train.Optimizer
        A optimizer class which optimizes parameters of
        the @p model with losses computed by @loss_fn.

    global_step: tf.Variable
        A global step value to use with optimizer and
        logging purpose.

    save_dir: str
        A path to the directory to save logs and models.

    score_threshold: float, default: .2
        Objectness score threshold to ignore boxes as non-object.

    iou_threshold: float, default: .2
        IoU threshold for non-max suppression.

    num_epochs: int, default: 200
        The number epochs to train.

    evaluate_epochs: int, default: 10
        Evaluate model by validation dataset and save the session
        every @p evaluate_epochs epochs.

    verbose_steps: int, default: 10
        Show metric every @p verbose_step.

    resume_path: str, default: None
        The path to resume session from.

    finetune_from: str, default: None
        If specified, resume only model weights from the architecture.
    """

    def __init__(self,
                 model,
                 num_classes,
                 train_num_batch,
                 val_num_batch,
                 train_iterator,
                 val_iterator,
                 anchor_converter,
                 loss_fn,
                 optimizer,
                 global_step,
                 save_dir,
                 score_threshold=.2,
                 iou_threshold=.5,
                 num_epochs=200,
                 evaluate_epochs=10,
                 verbose_steps=10,
                 resume_path=None,
                 finetune_from=None):
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.num_classes = num_classes
        self.train_num_batch = train_num_batch
        self.val_num_batch = val_num_batch
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.anchor_converter = anchor_converter
        self.train_batch = self.train_iterator.get_next()
        self.val_batch = self.val_iterator.get_next()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.global_step = global_step
        self.save_dir = save_dir
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.num_epochs = num_epochs
        self.evaluate_epochs = evaluate_epochs
        self.verbose_steps = verbose_steps
        self.resume_path = resume_path
        self.finetune_from = finetune_from

        # Inspect inputs.
        if hasattr(model, '__call__') is False:
            raise AttributeError('model object should have .__call__() method.')
        if hasattr(optimizer, 'minimize') is False:
            raise AttributeError(
                'optimizer object should have .minimize() method.')
        if any(key not in self.train_batch for key in ('image', 'label')):
            raise AttributeError(
                'train_batch object should have "image" and "label" keys')
        if any(key not in self.val_batch for key in ('image', 'label')):
            raise AttributeError(
                'val_batch object should have "image" and "label" keys')

        # Set up metrics.
        self.train_loss, \
            self.train_mean_loss, \
            self.train_mean_loss_update_op, \
            self.train_metric_reset_op, \
            self.train_step_summary_op, \
            self.train_epoch_summary_op = self.compute_metrics(
                self.train_batch['image'], self.train_batch['label'], self.train_num_batch, 'train')

        self.val_loss, \
            self.val_mean_loss, \
            self.val_mean_loss_update_op, \
            self.val_metric_reset_op, \
            self.val_step_summary_op, \
            self.val_epoch_summary_op = self.compute_metrics(
                self.val_batch['image'], self.val_batch['label'], self.val_num_batch, 'val')

        self.train_op = self.optimizer.minimize(
            self.train_loss,
            var_list=tf.trainable_variables(scope='model'),
            global_step=self.global_step)

        # Epoch ops and a saver should live in cpu.
        with tf.device('/cpu'):
            # In the training loop, it will increment epoch first.
            # So set -1 as the initial value to start from 0.
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)
            self.epoch_less_than_max = tf.less(self.epoch,
                                               tf.constant(self.num_epochs))
            self.saver = tf.train.Saver()

        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.ckpt_dir = os.path.join(self.save_dir, 'ckpts')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)

    def compute_metrics(self, images, labels, num_batch, name):
        """Compute necessary metics: loss, weights, IoU, and summaries.

        Parameters
        ----------
        images: (N, H, W, C=3) tf.tensor
            Image batch used as an input.
        labels: (N, grid_height, grid_width, num_anchors, 6) tf.Tensor
            Regression target for anchors.
            In the last dimension, the first 4 elements are the ground truth
            regresssion target (tx, ty, tw, th), the 5th element represents
            ground truth objectness, and the last element represents the ground truth
            class id.
            If there is no target assigned to an anchor, it will have all zeros
            for the 6 elements in the last dimension.
        num_batch: int
            The batch size of inputs.
        name: string
            Variable scope prefix for the metrics.

        Returns
        -------
        loss: Scalar tensor
            Loss value.
        image_summary: (H, 3 * W, C=3) tf.tensor
            Set of {input, prediction, ground truth} image
            used for visualization.
        mean_loss: Scalar tensor
            Mean loss value per epoch.
        mean_loss_update_op:
            Operator to update mean loss.
        mean_iou: Scalar tensor
            Mean IoU per epoch.
        mean_iou_update_op:
            Operator to update mean IoU.
        metric_reset_op:
            Operator to reset mean_loss and mean_iou values.
            Supposed to be called every epoch.
        step_summary_op:
            Operator to compute metrics for each step.
        epoch_summary_op:
            Operator to compute metrics for each epoch.
        """
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            predictions = self.model(images)

        # Metric computations should live in cpu.
        with tf.device('cpu:0'):
            with tf.variable_scope('{}_step_metrics'.format(name)) as scope:
                loss = self.loss_fn(predictions, labels)

            with tf.variable_scope('{}_epoch_metrics'.format(name)) as scope:
                mean_loss, mean_loss_update_op = tf.metrics.mean(loss)

                regression_predictions_nms, \
                    objectness_predictions_nms, \
                    class_predictions_nms, \
                    regression_labels_valid, \
                    class_labels_valid = self._parse_outputs(
                        predictions, labels, num_batch)

                # TODO: support random indexing.
                image_summary = tf.cast(images[0] * 255., tf.uint8)
                regression_prediction_image = regression_predictions_nms[0]
                objectness_prediction_image = objectness_predictions_nms[0]
                class_prediction_image = class_predictions_nms[0]
                regression_label_image = regression_labels_valid[0]
                class_label_image = class_labels_valid[0]

                image_summary = self._draw_boxes(
                    image_summary, regression_prediction_image,
                    objectness_prediction_image, class_prediction_image,
                    regression_label_image, class_label_image)

                var_list = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
                metric_reset_op = tf.variables_initializer(var_list)

            # Add to summary.
            # NOTE: Needs to separate epoch metric summaries as they need to compute
            # after the update operations.
            step_summaries = []
            step_summaries.append(
                tf.summary.scalar('{}_mean_loss'.format(name), mean_loss))

            epoch_summaries = []
            epoch_summaries.append(
                tf.summary.image('{}_visualization'.format(name),
                                 image_summary))

            step_summary_op = tf.summary.merge(step_summaries)
            epoch_summary_op = tf.summary.merge(epoch_summaries)

        return loss, mean_loss, mean_loss_update_op, metric_reset_op, \
            step_summary_op, epoch_summary_op

    def _draw_boxes(self, image_summary, regression_prediction,
                    objectness_prediction, class_prediction, regression_label,
                    class_label):
        """Draw boxes and classes of prediction and label.
        Also show objectness for prediction.
        """

        def cv_draw(image, box_preds, box_gts):
            image = np.squeeze(image)
            height = image.shape[0]
            width = image.shape[1]
            box_preds = np.clip(box_preds, 0., 1.)
            box_preds = np.stack(
                (box_preds[:, 0] * width, box_preds[:, 1] * height,
                 box_preds[:, 2] * width, box_preds[:, 3] * height), 1)
            box_gts = np.stack((box_gts[:, 0] * width, box_gts[:, 1] * height,
                                box_gts[:, 2] * width, box_gts[:, 3] * height),
                               1)
            box_preds = box_preds.astype(np.int)
            box_gts = box_gts.astype(np.int)
            for box in box_preds:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
            for box in box_gts:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                              (0, 0, 0), 2)

            image = np.reshape(image, (1, height, width, 3))
            return image

        regression_prediction = tf.reshape(regression_prediction, (-1, 4))
        regression_label = tf.reshape(regression_label, (-1, 4))
        image_summary = tf.py_func(
            cv_draw, [image_summary, regression_prediction, regression_label],
            tf.uint8)

        return image_summary

    def _parse_outputs(self, predictions, labels, num_batch):
        """Parse model outputs and labels for evaluations.
        """
        regression_predictions = predictions[..., :4]
        objectness_predictions = predictions[..., 4]
        class_logits_predictions = predictions[..., 5:]
        class_predictions = tf.argmax(class_logits_predictions, axis=-1)
        regression_labels = labels[..., :4]
        objectness_labels = labels[..., 4]
        class_labels = labels[..., 5]

        shape = tf.shape(objectness_predictions)
        grid_h, grid_w, num_anchors = shape[1], shape[2], shape[3]
        num_anchors_grid = grid_h * grid_w * num_anchors

        regression_predictions = tf.reshape(regression_predictions,
                                            [num_batch, num_anchors_grid, 4])
        objectness_predictions = tf.reshape(objectness_predictions,
                                            [num_batch, num_anchors_grid])
        class_predictions = tf.reshape(class_predictions,
                                       [num_batch, num_anchors_grid])

        # Apply NMS on predictions per class per sample.
        regression_predictions_nms = []
        objectness_predictions_nms = []
        class_predictions_nms = []
        for regression_prediction, objectness_prediction, class_prediction in zip(
                tf.unstack(regression_predictions, num=num_batch),
                tf.unstack(objectness_predictions, num=num_batch),
                tf.unstack(class_predictions, num=num_batch)):
            # Decode regressions.
            regression_prediction = self.anchor_converter.decode_regression(
                regression_prediction)
            regression_prediction = xywh_to_ltrb(regression_prediction)

            selected_indices_predictions = []
            for i in range(self.num_classes):
                class_mask = tf.equal(class_prediction, i)

                regression_predictions_class = tf.boolean_mask(
                    regression_prediction, class_mask)
                objectness_class = tf.boolean_mask(objectness_prediction,
                                                   class_mask)

                # Returns an empty tensor if inputs are empty.
                selected_indices = tf.image.non_max_suppression(
                    regression_predictions_class,
                    objectness_class,
                    num_anchors_grid,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold),
                selected_indices = tf.reshape(
                    tf.gather(tf.where(class_mask), selected_indices), [-1])

                selected_indices_predictions.append(selected_indices)

            selected_indices_predictions = tf.concat(
                selected_indices_predictions, axis=0)
            regression_prediction_nms = tf.gather(regression_prediction,
                                                  selected_indices_predictions)
            objectness_prediction_nms = tf.gather(objectness_prediction,
                                                  selected_indices_predictions)
            class_prediction_nms = tf.gather(class_prediction,
                                             selected_indices_predictions)
            regression_predictions_nms.append(regression_prediction_nms)
            objectness_predictions_nms.append(objectness_prediction_nms)
            class_predictions_nms.append(class_prediction_nms)

        regression_labels = tf.reshape(regression_labels,
                                       [num_batch, num_anchors_grid, 4])
        objectness_labels = tf.reshape(objectness_labels,
                                       [num_batch, num_anchors_grid])
        class_labels = tf.reshape(class_labels, [num_batch, num_anchors_grid])

        # Extract only valid boxes which has a target assigned.
        regression_labels_valid = []
        class_labels_valid = []
        for regression_label, objectness_label, class_label in zip(
                tf.unstack(regression_labels, num=num_batch),
                tf.unstack(objectness_labels, num=num_batch),
                tf.unstack(class_labels, num=num_batch)):
            # Decode regressions.
            regression_label = self.anchor_converter.decode_regression(
                regression_label)
            regression_label = xywh_to_ltrb(regression_label)

            valid_mask = tf.not_equal(objectness_label, 0)
            regression_label = tf.boolean_mask(regression_label, valid_mask)
            class_label = tf.boolean_mask(regression_label, valid_mask)

            regression_labels_valid.append(regression_label)
            class_labels_valid.append(class_label)

        return regression_predictions_nms, objectness_predictions_nms, \
            class_predictions_nms, regression_labels_valid, class_labels_valid

    def train(self, sess):
        """Execute train loop.

        Parameters
        ----------
        sess: tf.Session
            TensorFlow session to run train loop.
        """
        if self.resume_path:
            if self.finetune_from:
                self.model._restore_model_variables(sess, self.resume_path,
                                                    self.finetune_from)
                self.logger.info('Finetune from {}.'.format(self.finetune_from))
            else:
                self.saver.restore(sess, self.resume_path)
                self.logger.info('The session restored from {}.'.format(
                    self.resume_path))

        self.logger.info('Start training.')
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        while sess.run(self.epoch_less_than_max):
            sess.run(self.increment_epoch_op)
            ep = sess.run(self.epoch)

            sess.run((self.train_iterator.initializer,
                      self.train_metric_reset_op))

            step_op = (self.global_step, self.train_step_summary_op,
                       self.train_mean_loss_update_op, self.train_op)

            start = time.clock()
            while True:
                try:
                    out = sess.run(step_op)
                    step = out[0]
                    if step % self.verbose_steps == 0:
                        train_mloss = sess.run(self.train_mean_loss)
                        self.logger.info(
                            'Train step: {}, mean loss: {:06f}'.format(
                                step, train_mloss))
                except tf.errors.OutOfRangeError:
                    break
            proc_time = time.clock() - start

            # Avoid sess.run(self.train_step_summary_op) here, otherwise get OutOfRangeError.
            train_step_summary = out[1]
            train_mloss = sess.run(self.train_mean_loss)
            self.logger.info(
                'Train epoch: {},\tproc time: {:06f}\tmean loss: {:06f}'.format(
                    ep, proc_time, train_mloss))
            summary_writer.add_summary(train_step_summary, ep)

            sess.run(self.train_iterator.initializer)
            train_epoch_summary = sess.run(self.train_epoch_summary_op)
            summary_writer.add_summary(train_epoch_summary, ep)

            if (ep + 1) % self.evaluate_epochs == 0:
                with tf.device('/cpu'):
                    save_path = '{:08d}.ckpt'.format(ep)
                    self.saver.save(sess, os.path.join(self.ckpt_dir,
                                                       save_path))
                    self.logger.info('The session saved')

                self.validate(sess, summary_writer, ep)

    def validate(self, sess, summary_writer, epoch):
        """Execute validation loop.

        Parameters
        ----------
        summary_writer: tf.summary.FileWriter
            The summary writer to add summary metrics of validation.
        """
        self.logger.info('Start evaluation.')
        sess.run((self.val_iterator.initializer, self.val_metric_reset_op))
        step_op = (self.val_mean_loss_update_op, self.val_step_summary_op)

        start = time.clock()
        while True:
            try:
                out = sess.run(step_op)
            except tf.errors.OutOfRangeError:
                break
        proc_time = time.clock() - start

        self.logger.info('Validation proc time: {:06f}'.format(proc_time))
        _, val_step_summary = out

        val_mloss = sess.run(self.val_mean_loss)
        self.logger.info('Validation mean loss: {:06f}'.format(val_mloss))

        sess.run(self.val_iterator.initializer)
        val_epoch_summary = sess.run(self.val_epoch_summary_op)

        summary_writer.add_summary(val_step_summary, epoch)
        summary_writer.add_summary(val_epoch_summary, epoch)
