"""Loss implementations for detection.
"""

import numpy as np
import tensorflow as tf


def _parse_yolo_prediction_and_target(prediction, target):
    """Parse YOLO detector's prediction to
    regression, objectiveness, and class logits.
    """
    regression_prediction = prediction[..., :4]
    objectiveness_prediction = prediction[..., 4]
    class_prediction = prediction[..., 5:]
    regression_target = target[..., :4]
    objectiveness_target = target[..., 4]
    class_target = target[..., 5]

    # Foreground has objectiveness = 1, otherwise background.
    foreground = tf.cast(objectiveness_target, tf.bool)
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

    # Separate to objectiveness and nonobjectiveness
    # to weight losses.
    objectiveness_sigmoid_prediction = tf.nn.sigmoid(objectiveness_prediction)
    objectiveness_sigmoid_prediction = tf.clip_by_value(
        objectiveness_sigmoid_prediction,
        np.finfo(np.float32).eps, 1. - np.finfo(np.float32).eps)
    nonobjectiveness_sigmoid_prediction = 1. - objectiveness_sigmoid_prediction
    # Convert to logits to use tf loss function.
    nonobjectiveness_prediction = -tf.log(
        (1. - nonobjectiveness_sigmoid_prediction) /
        nonobjectiveness_sigmoid_prediction)

    objectiveness_stack_prediction = tf.stack(
        [nonobjectiveness_prediction, objectiveness_prediction], axis=-1)
    objectiveness_prediction = tf.reshape(
        tf.boolean_mask(objectiveness_stack_prediction, foreground),
        shape=(-1, 2))
    nonobjectiveness_prediction = tf.reshape(
        tf.boolean_mask(objectiveness_stack_prediction, background),
        shape=(-1, 2))

    nonobjectiveness_target = tf.boolean_mask(objectiveness_target, background)
    objectiveness_target = tf.boolean_mask(objectiveness_target, foreground)
    objectiveness_onehot_target = tf.reshape(
        tf.one_hot(tf.cast(objectiveness_target, tf.int32), depth=2),
        shape=(-1, 2))
    nonobjectiveness_onehot_target = tf.reshape(
        tf.one_hot(tf.cast(nonobjectiveness_target, tf.int32), depth=2),
        shape=(-1, 2))

    parsed_prediction = {
        'foreground_regression': foreground_regression_prediction,
        'objectiveness': objectiveness_prediction,
        'nonobjectiveness': nonobjectiveness_prediction,
        'foreground_class': foreground_class_prediction
    }
    parsed_target = {
        'foreground_regression': foreground_regression_target,
        'objectiveness': objectiveness_onehot_target,
        'nonobjectiveness': nonobjectiveness_onehot_target,
        'foreground_class': foreground_class_onehot_target
    }
    return parsed_prediction, parsed_target, foreground, background


def yolo_detection_loss(prediction,
                        target,
                        weight_regression=5.,
                        weight_nonobjectiveness=.5,
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
        ground truth objectiveness, and the last element represents the ground truth
        class id.

    weight_regression: float, default 5.
        Weight parameter for regresssion loss.

    weight_nonobjectiveness: float, default .5
        Weight parameter for non-objectiveness loss.

    return_all_losses: bool, default False
        If True, return loss for each component in addition to a total loss.

    Returns
    -------
    loss: tf.float
        Total loss of detection.

    regression_loss: tf.float
        Loss for regression part.

    objectiveness_loss: tf.float
        Loss for objectiveness part.

    nonobjectiveness_loss: tf.float
        Loss for non-objectiveness part.

    classification_loss: tf.float
        Loss for classification part.
    """
    parsed_prediction, parsed_target, foreground, background \
        = _parse_yolo_prediction_and_target(prediction, target)

    regression_loss = tf.losses.huber_loss(
        parsed_target['foreground_regression'],
        parsed_prediction['foreground_regression'])
    objectiveness_loss = tf.losses.softmax_cross_entropy(
        parsed_target['objectiveness'], parsed_prediction['objectiveness'])
    nonobjectiveness_loss = tf.losses.softmax_cross_entropy(
        parsed_target['nonobjectiveness'],
        parsed_prediction['nonobjectiveness'])
    classification_loss = tf.losses.softmax_cross_entropy(
        parsed_target['foreground_class'],
        parsed_prediction['foreground_class'])

    loss = weight_regression * regression_loss + \
           objectiveness_loss + \
           weight_nonobjectiveness * nonobjectiveness_loss + \
           classification_loss

    if return_all_losses:
        return loss, regression_loss, objectiveness_loss, \
            nonobjectiveness_loss, classification_loss
    return loss
