"""Image processing utils.
"""

import tensorflow as tf


def random_flip_left_right_image_and_label(image, label, prob=.5):
    """Randomly flip image and label with probbality @p prob.

    Parameters
    ----------
    image: (H, W, C=3) tf.Tensor
        Image tensor.
    label: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Label tensor.
        The location (l, t, r, b) must be normalized.
    prob: float
        Probability to flip image and label.

    Returns
    -------
    image_crop: (H, W, C=3) tf.Tensor.
        Cropped image tensor.
    label_crop: (num_boxes, (l, t, r, b, class_id)) tf.Tensor.
        Cropped label tensor.
        The location (l, t, r, b) is normalized.
    """
    rand = tf.reshape(tf.random_uniform(tf.constant([1])), [])
    image_flip, label_flip = tf.cond(
        tf.less(rand, prob),
        lambda: (tf.image.flip_left_right(image), _box_flip_left_right(label)),
        lambda: (image, label))
    return image_flip, label_flip


def _box_flip_left_right(box):
    """flip boxes location from left to right.
    Parameters
    ----------
    box: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Label tensor.
        The location (l, t, r, b) must be normalized.

    Returns
    -------
    box_flipped: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Flipped label tensor.
        The location (l, t, r, b) is normalized.
    """
    with tf.control_dependencies([
            tf.assert_less_equal(tf.reduce_max(box), 1.),
            tf.assert_greater_equal(tf.reduce_min(box), 0.)
    ]):
        box_flipped = tf.stack(
            [1. - box[:, 2], box[:, 1], 1. - box[:, 0], box[:, 3]], axis=1)
    return box_flipped


def random_crop_image_and_label(image, label, crop_size):
    """Randomly crops `image` together with `label`.

    Parameters
    ----------
    image: (H, W, C=3) tf.Tensor
        Image tensor.
    label: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Label tensor.
        The location (l, t, r, b) must be normalized.
    crop_size: (H, W) 1D tf.Tensor
        Height (H) and Width (W) of crop size.

    Returns
    -------
    image_crop: (H, W, C=3) tf.Tensor.
        Cropped image tensor.
    label_crop: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Cropped label tensor.
        The location (l, t, r, b) is normalized.
    """
    image_shape = tf.shape(image)[:2]
    image_height, image_width = image_shape[0], image_shape[1]
    with tf.control_dependencies([
            tf.assert_greater(crop_size[0], 0),
            tf.assert_greater(crop_size[1], 0),
            tf.assert_less(crop_size[0], image_width + 1),
            tf.assert_less(crop_size[1], image_height + 1)
    ]):
        l_max = image_width - crop_size[0] + 1
        t_max = image_height - crop_size[1] + 1

    l_crop = tf.random_uniform([1], maxval=l_max, dtype=tf.int32)[0]
    t_crop = tf.random_uniform([1], maxval=t_max, dtype=tf.int32)[0]
    image_crop = image[l_crop:l_crop + crop_size[0], t_crop:t_crop +
                       crop_size[1]]

    # Crop boxes in normalized coordinate.
    label_crop_l = label[:, 0] - tf.cast(l_crop / image_width, label.dtype)
    label_crop_t = label[:, 1] - tf.cast(t_crop / image_height, label.dtype)
    label_crop_r = label[:, 2] - tf.cast(l_crop / image_width, label.dtype)
    label_crop_b = label[:, 3] - tf.cast(t_crop / image_height, label.dtype)
    label_crop = tf.stack(
        (label_crop_l, label_crop_t, label_crop_r, label_crop_b), axis=1)
    # Remove boxes which are outside the cropped region.
    label_mask = tf.logical_not(
        tf.logical_or(
            tf.logical_or(
                tf.greater_equal(label_crop[:, 0], 1.),
                tf.greater_equal(label_crop[:, 1], 1.)),
            tf.logical_or(
                tf.less_equal(label_crop[:, 2], 0.),
                tf.less_equal(label_crop[:, 3], 0.))))
    label_crop = tf.boolean_mask(label_crop, label_mask)
    # Squash box into the image region.
    label_crop = tf.clip_by_value(
        label_crop, clip_value_min=0., clip_value_max=1.)

    label_crop = tf.concat(
        (label_crop,
         tf.expand_dims(tf.boolean_mask(label[:, -1], label_mask), axis=1)),
        axis=1)

    return image_crop, label_crop


def resize_image_and_label(image,
                           label,
                           size,
                           image_method=tf.image.ResizeMethod.BILINEAR,
                           align_corners=False):
    """Resize `image` together with `label`.

    Parameters
    ----------
    image: (H, W, C=3) tf.Tensor
        Image tensor.
    label: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Label tensor.
        The location (l, t, r, b) must be normalized.
    size: tuple of (image_height, image_width)
        The size of resized image.
    image_method: functional, default tf.image.ResizeMethod.BILINEAR
        The parameter of tf.image.resize_image.
        Interpolation method for image resizeing.
    align_corners: boolean, default: False
        The parameter of tf.image.resize_image.

    Returns
    -------
    image_resize: (H, W, C=3) tf.Tensor.
        Resized image tensor.
    label_resize: (num_boxes, (l, t, r, b, class_id)) tf.Tensor
        Resized label tensor.
        The location (l, t, r, b) is normalized.
    """
    image_resize = tf.image.resize_images(
        image, size, method=image_method, align_corners=align_corners)

    # As the labels are normalized, just return the same values.
    label_resize = label

    return image_resize, label_resize
