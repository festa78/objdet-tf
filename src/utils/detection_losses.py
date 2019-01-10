"""Loss implementations for detection.
"""

import numpy as np
import tensorflow as tf


def _parse_yolo_prediction_and_target(prediction, target):
    """Parse YOLO detector's prediction to
    regression, objectness, and class logits.
    """
    regression_prediction = prediction[..., :4]
    objectness_prediction = prediction[..., 4]
    class_prediction = prediction[..., 5:]
    regression_target = target[..., :4]
    objectness_target = target[..., 4]
    class_target = target[..., 5]

    with tf.control_dependencies([
            tf.assert_equal(
                tf.shape(regression_prediction), tf.shape(regression_target))
    ]):
        # Foreground has objectness = 1, otherwise background.
        foreground = tf.cast(objectness_target, tf.bool)
        background = tf.logical_not(foreground)

    # Only cares foreground for regression and classification.
    foreground_regression_prediction \
        = tf.reshape(tf.boolean_mask(regression_prediction, foreground), shape=(-1, 4))
    foreground_regression_target \
        = tf.reshape(tf.boolean_mask(regression_target, foreground), shape=(-1, 4))
    foreground_class_prediction \
        = tf.boolean_mask(class_prediction, foreground)
    foreground_class_target \
        = tf.boolean_mask(class_target, foreground)

    num_classes = tf.shape(class_prediction)[-1]
    foreground_class_onehot_target = tf.reshape(
        tf.one_hot(
            tf.cast(foreground_class_target, tf.int32), depth=num_classes),
        shape=(-1, num_classes))

    # Separate to objectness and nonobjectness
    # to weight losses.
    objectness_sigmoid_prediction = tf.nn.sigmoid(objectness_prediction)
    objectness_sigmoid_prediction = tf.clip_by_value(
        objectness_sigmoid_prediction,
        np.finfo(np.float32).eps, 1. - np.finfo(np.float32).eps)
    nonobjectness_sigmoid_prediction = 1. - objectness_sigmoid_prediction
    # Convert to logits to use tf loss function.
    nonobjectness_prediction = -tf.log((1. - nonobjectness_sigmoid_prediction) /
                                       nonobjectness_sigmoid_prediction)

    # Create logits expression.
    objectness_stack_prediction = tf.stack(
        [nonobjectness_prediction, objectness_prediction], axis=-1)
    objectness_prediction = tf.reshape(
        tf.boolean_mask(objectness_stack_prediction, foreground), shape=(-1, 2))
    nonobjectness_prediction = tf.reshape(
        tf.boolean_mask(objectness_stack_prediction, background), shape=(-1, 2))

    # Create onehot targets.
    nonobjectness_target = tf.boolean_mask(objectness_target, background)
    objectness_target = tf.boolean_mask(objectness_target, foreground)
    nonobjectness_onehot_target = tf.reshape(
        tf.one_hot(tf.cast(nonobjectness_target, tf.int32), depth=2),
        shape=(-1, 2))
    objectness_onehot_target = tf.reshape(
        tf.one_hot(tf.cast(objectness_target, tf.int32), depth=2),
        shape=(-1, 2))

    parsed_prediction = {
        'foreground_regression': foreground_regression_prediction,
        'objectness': objectness_prediction,
        'nonobjectness': nonobjectness_prediction,
        'foreground_class': foreground_class_prediction
    }
    parsed_target = {
        'foreground_regression': foreground_regression_target,
        'objectness': objectness_onehot_target,
        'nonobjectness': nonobjectness_onehot_target,
        'foreground_class': foreground_class_onehot_target
    }
    return parsed_prediction, parsed_target, foreground, background


def yolo_detection_loss(prediction,
                        target,
                        weight_regression=5.,
                        weight_nonobjectness=.5,
                        return_all_losses=False):
    """Loss for YOLO detector.

    Parameters
    ----------
    prediction: (N, GRID_H, GRID_W, NUM_ANCHORS, 5 + NUM_CLASSES) tf.Tensor
        Output from YOLO detector network.
        In the last dimension, the first 4 elements predict
        regresssion (tx, ty, tw, th), the 5th element predicts object-ness,

    target: (N, GRID_H, GRID_W, NUM_ANCHORS, 6) tf.Tensor
        Ground truth for YOLO detector.
        In the last dimension, the first 4 elements are the ground truth
        regresssion target (tx, ty, tw, th), the 5th element represents
        ground truth objectness, and the last element represents the ground truth
        class id.

    weight_regression: float, default 5.
        Weight parameter for regresssion loss.

    weight_nonobjectness: float, default .5
        Weight parameter for non-objectness loss.

    return_all_losses: bool, default False
        If True, return loss for each component in addition to a total loss.

    Returns
    -------
    loss: tf.float
        Total loss of detection.

    regression_loss: tf.float
        Loss for regression part.
        Returns if @p return_all_losses is True.

    objectness_loss: tf.float
        Loss for objectness part.
        Returns if @p return_all_losses is True.

    nonobjectness_loss: tf.float
        Loss for non-objectness part.
        Returns if @p return_all_losses is True.

    classification_loss: tf.float
        Loss for classification part.
        Returns if @p return_all_losses is True.
    """
    parsed_prediction, parsed_target, foreground, background \
        = _parse_yolo_prediction_and_target(prediction, target)

    regression_loss = tf.losses.huber_loss(
        parsed_target['foreground_regression'],
        parsed_prediction['foreground_regression'])
    objectness_loss = tf.losses.softmax_cross_entropy(
        parsed_target['objectness'], parsed_prediction['objectness'])
    nonobjectness_loss = tf.losses.softmax_cross_entropy(
        parsed_target['nonobjectness'], parsed_prediction['nonobjectness'])
    classification_loss = tf.losses.softmax_cross_entropy(
        parsed_target['foreground_class'],
        parsed_prediction['foreground_class'])

    loss = weight_regression * regression_loss + \
           objectness_loss + \
           weight_nonobjectness * nonobjectness_loss + \
           classification_loss

    if return_all_losses:
        return loss, regression_loss, objectness_loss, \
            nonobjectness_loss, classification_loss
    return loss
