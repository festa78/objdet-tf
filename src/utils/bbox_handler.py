"""Utility class for bounding box handling.
"""

import numpy as np
import tensorflow as tf

import project_root


class AnchorConverter:
    """Manage to generate anchor targets from anchor priors.
    Make sure consistent conversions between absolute bounding box
    location and anchor target translation from anchor_priors.

    Parameters
    ----------
    anchor_priors: (grid_height, grid_width, num_anchors, (x, y, w, h) box location)
                   numpy float array
        Anchor priors in (x, y, w ,h) coordinate.
        The coordinate must be normalized to [0., 1.].
        Will be converted to tf.Tensor internally.
    """

    def __init__(self, anchor_priors):
        assert anchor_priors[..., :2].max() <= 1.
        assert anchor_priors[..., :2].min() >= 0.

        self.grid_height, self.grid_width, self.num_anchors = anchor_priors.shape[:
                                                                                  3]
        self.total_anchors = self.grid_height * self.grid_width * self.num_anchors
        self.anchor_priors = tf.convert_to_tensor(anchor_priors)
        # Concatenate anchors for convenience.
        self.anchor_priors = tf.reshape(anchor_priors, [self.total_anchors, 4])
        self.anchor_priors = tf.cast(self.anchor_priors, tf.float32)

    def generate_anchor_targets(self, gt_bboxes, iou_threshold=.5):
        """Given ground truth bounding boxes and anchor priors,
        compute IoU between them. If IoU is larger than @p iou_threshold,
        assign the corresponding anchor to object and regression.

        Parameters
        ----------
        gt_bboxes: (num_boxes (l, t, r, b, class_id)) tf.Tensor
            Ground truth bounding box locations and its class ids.
            The coordinate must be normalized to [0, 1.].

        iou_threshold: float, default: 5
            When an anchor prior and a ground truth bounding boxes have
            IoU larger than this value, the anchor prior is assigned as
            foreground and (tx, ty, tw, th) translation between them
            are computed as a regression target.

        Returns
        -------
        anchor_targets: (grid_height, grid_width, num_anchors, 6) tf.Tensor
            Regression target for anchors.
            In the last dimension, the first 4 elements are the ground truth
            regresssion target (tx, ty, tw, th), the 5th element represents
            ground truth objectness, and the last element represents the ground truth
            class id.
            If there is no target assigned to an anchor, it will have all zeros
            for the 6 elements in the last dimension.
        """
        gt_locations_ltrb, gt_classes = gt_bboxes[:, :4], gt_bboxes[:, 4]

        with tf.control_dependencies([
                tf.assert_less_equal(tf.reduce_max(gt_locations_ltrb), 1.),
                tf.assert_greater_equal(tf.reduce_min(gt_locations_ltrb), 0.)
        ]):
            gt_locations_xywh = ltrb_to_xywh(gt_locations_ltrb)

        # Assign ground truths to anchors which have IoU larger than
        # the threshold.
        iou = compute_iou(self.anchor_priors, gt_locations_xywh)
        iou_masked = tf.cast(iou > iou_threshold, tf.float32)
        iou *= iou_masked

        # Assign ground truth which has the largest IoU.
        # assigned_idx for non-object will be ignored later.
        assigned_idx = tf.argmax(iou, axis=1)

        # Compute regression target from prior anchors.
        assigned_gt_regressions = self.encode_regression(gt_locations_xywh, assigned_idx)
        assigned_gt_classes = tf.gather(gt_classes, assigned_idx)
        assigned_gt_classes = tf.reshape(assigned_gt_classes, [-1, 1])

        # If there is no ground truth assigned to an anchor, it is
        # dealt as non-object.
        objectness = tf.cast(tf.reduce_sum(iou_masked, axis=1) > 0, tf.float32)
        objectness = tf.reshape(objectness, [-1, 1])

        assigned_gt_regressions *= tf.tile(objectness, [1, 4])
        assigned_gt_classes *= objectness

        anchor_targets = tf.concat([assigned_gt_regressions, objectness, assigned_gt_classes], 1)

        anchor_targets = tf.reshape(
            anchor_targets,
            [self.grid_height, self.grid_width, self.num_anchors, 6])

        return anchor_targets

    def encode_regression(self, target_locations_xywh, assigned_idx):
        """Compute regression targets from anchor priors to targets.
        Use expressions in the figure 3. in YOLO v2 paper.
        cf. https://arxiv.org/abs/1612.08242

        Parameters
        ----------
        target_locations_xywh: (num_ground_truths, (x, y, w, h) location) tf.Tensor
            (x, y, w, h) target bounding box locations.
            The coordinate must be normalized.

        assigned_idx: (num_anchors,) tf.Tensor
            Indices of @p target_location_xywh assigned to each anchor prior.
            The order must be aligned with @p anchor_priors.

        Returns
        -------
        assigned_gt_regressions: (num_anchors, (tx, ty, tw, th) translation) tf.Tensor
            Regression target for each anchor.
        """
        target_locations_ltrb = xywh_to_ltrb(target_locations_xywh)
        assert_ops = [
            tf.assert_less_equal(tf.reduce_max(target_locations_ltrb), 1.),
            tf.assert_greater_equal(tf.reduce_min(target_locations_ltrb), 0.),
            tf.assert_equal(self.total_anchors,
                            tf.shape(assigned_idx)[0])
        ]
        with tf.control_dependencies(assert_ops):
            assigned_target_locations = tf.gather(target_locations_xywh,
                                                  assigned_idx)
        assigned_gt_regressions_xy \
            = assigned_target_locations[:, :2] - self.anchor_priors[:, :2]
        # Squash to [0., 1.]
        # TODO: Should use a top left point of grid instead?
        assigned_gt_regressions_xy = (assigned_gt_regressions_xy + 1.) / 2.
        assigned_gt_regressions_xy = logit(assigned_gt_regressions_xy)
        assigned_gt_regressions_wh = tf.log(
            assigned_target_locations[:, 2:] / self.anchor_priors[:, 2:])
        assigned_gt_regressions = tf.concat(
            [assigned_gt_regressions_xy, assigned_gt_regressions_wh], axis=1)

        return assigned_gt_regressions

    def decode_regression(self):
        pass


def logit(x):
    """Compute logit.
    """
    return -tf.log(1. / x - 1.)


def generate_anchor_priors(image_shape,
                           grid_size=(16, 16),
                           scale=(8, 16, 32),
                           aspect_ratio=(.5, 1., 2.)):
    """Generate anchor priors for images.

    Parameters
    ----------
    image_shape: (height, width) int tuple
        Shape of image on which anchors will be generated.

    grid_size: (height, width) int tuple
        Size of grid to separate image.
        Anchors will be generated on each grid center.

    scale: int tuple, default: (8, 16, 32)
        Scales of anchor boxes to generate.

    aspect_ratio: float tuple, default: (.5, 1., 2.)
        Aspect ratios of anchor boxes to generate.
        NOTE: aspect_ratio = width / height.

    Returns
    -------
    anchor_priors: (grid_height, grid_width, num_anchors, (x, y, w, h) box coordinate)
                   numpy float array
        Generated anchors for each grid cell.
        @p num_anchors = len(scale) * len(aspect_ratio)
    """
    assert len(scale) > 0
    assert len(aspect_ratio) > 0

    grid_height, grid_width = grid_size
    image_height, image_width = image_shape
    grid_height_pixel = image_height // grid_height
    grid_width_pixel = image_width // grid_width
    num_anchors = len(scale) * len(aspect_ratio)

    # Compute anchor centers for each grid cell.
    one_row = np.arange(grid_width) * grid_width_pixel + grid_width_pixel // 2
    one_col = np.arange(
        grid_height) * grid_height_pixel + grid_height_pixel // 2
    anchor_priors = np.array(np.meshgrid(one_row, one_col), dtype=np.float)
    # Convert to the format compatible with the return value.
    anchor_priors = np.transpose(anchor_priors, (1, 2, 0))
    anchor_priors = np.expand_dims(anchor_priors, 2)
    anchor_priors = np.tile(anchor_priors, (1, 1, num_anchors, 2))
    # Set zeros to w and h values.
    anchor_priors[..., 2:] = 0.

    # Generate anchor shapes relative to the grid cell centers.
    # Compute anchor shape by aspect_ratio using image area size.
    grid_area = grid_width_pixel * grid_height_pixel
    anchor_width = np.round(np.sqrt([grid_area / r for r in aspect_ratio]))
    anchor_height = np.round(anchor_width * aspect_ratio)
    # Scale shapes according to the shape parameters.
    anchor_width_scaled = np.repeat(scale, (len(aspect_ratio),)) * np.tile(
        anchor_width, (len(scale),))
    anchor_height_scaled = np.repeat(scale, (len(aspect_ratio),)) * np.tile(
        anchor_height, (len(scale),))
    anchor_shapes = np.stack((anchor_width_scaled, anchor_height_scaled), 1)

    # Integrate into anchor_priors.
    anchor_priors[..., 2:] = anchor_shapes.reshape((1, 1, num_anchors, 2))

    return anchor_priors


def compute_iou(boxes1, boxes2):
    """Compute IoU between two bounding boxes.

    Parameters
    ----------
    boxes1: (num_boxes1, (x, y, w, h) location) tf.Tensor
        (x, y, w, h) box locations.
    boxes2: (num_boxes2, (x, y, w, h) location) tf.Tensor
        (x, y, w, h) box locations.

    Returns
    -------
    iou: (num_boxes1, num_boxes2) tf.Tensor
        IoU between boxes1 and boxes2.
    """
    num_boxes1 = tf.shape(boxes1)[0]
    num_boxes2 = tf.shape(boxes2)[0]

    boxes1_ltrb = xywh_to_ltrb(boxes1)
    boxes2_ltrb = xywh_to_ltrb(boxes2)
    boxes1_ltrb_tile = tf.tile(
        tf.reshape(boxes1_ltrb, [num_boxes1, 1, 4]), [1, num_boxes2, 1])
    boxes2_ltrb_tile = tf.tile(
        tf.reshape(boxes2_ltrb, [1, num_boxes2, 4]), [num_boxes1, 1, 1])

    l_max = tf.reduce_max(
        tf.stack([boxes1_ltrb_tile[..., 0], boxes2_ltrb_tile[..., 0]], axis=2),
        axis=2)
    t_max = tf.reduce_max(
        tf.stack([boxes1_ltrb_tile[..., 1], boxes2_ltrb_tile[..., 1]], axis=2),
        axis=2)
    r_min = tf.reduce_min(
        tf.stack([boxes1_ltrb_tile[..., 2], boxes2_ltrb_tile[..., 2]], axis=2),
        axis=2)
    b_min = tf.reduce_min(
        tf.stack([boxes1_ltrb_tile[..., 3], boxes2_ltrb_tile[..., 3]], axis=2),
        axis=2)
    intersection = (r_min - l_max) * (b_min - t_max)
    # If boxes are disjoint, set intersection 0.
    intersection *= tf.cast(tf.less(l_max, r_min), tf.float32)
    intersection *= tf.cast(tf.less(t_max, b_min), tf.float32)

    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    boxes1_area_tile = tf.tile(
        tf.reshape(boxes1_area, [num_boxes1, 1]), [1, num_boxes2])
    boxes2_area_tile = tf.tile(
        tf.reshape(boxes2_area, [1, num_boxes2]), [num_boxes1, 1])
    union = boxes1_area_tile + boxes2_area_tile - intersection

    iou = intersection / union
    return iou


def ltrb_to_xywh(ltrb):
    """Convert a box coordinate from ltrb to xywh.

    Parameters
    ----------
    ltrb: (num_boxes, (l, t, r, b) location)
        The box location in (l, t, r, b) coordinate.

    Returns
    -------
    xywh: (num_boxes, (x, y, w, h) location)
        The box location in (x, y, w, h) coordinate.
    """
    x = (ltrb[:, 0] + ltrb[:, 2]) / 2.
    y = (ltrb[:, 1] + ltrb[:, 3]) / 2.
    w = ltrb[:, 2] - ltrb[:, 0]
    h = ltrb[:, 3] - ltrb[:, 1]

    # Make sure consistent shape.
    x = tf.reshape(x, (tf.shape(ltrb)[0], 1))
    y = tf.reshape(y, (tf.shape(ltrb)[0], 1))
    w = tf.reshape(w, (tf.shape(ltrb)[0], 1))
    h = tf.reshape(h, (tf.shape(ltrb)[0], 1))

    with tf.control_dependencies(
        [tf.assert_non_negative(w),
         tf.assert_non_negative(h)]):
        xywh = tf.concat([x, y, w, h], axis=1)
    return xywh


def xywh_to_ltrb(xywh):
    """Convert a box coordinate from xywh to ltrb.

    Parameters
    ----------
    xywh: (num_boxes, (x, y, w, h) location)
        The box location in (x, y, w, h) coordinate.

    Returns
    -------
    ltrb: (num_boxes, (l, t, r, b) location)
        The box location in (l, t, r, b) coordinate.
    """
    l = xywh[:, 0] - xywh[:, 2] / 2.
    t = xywh[:, 1] - xywh[:, 3] / 2.
    r = xywh[:, 0] + xywh[:, 2] / 2.
    b = xywh[:, 1] + xywh[:, 3] / 2.

    # Make sure consistent shape.
    l = tf.reshape(l, (tf.shape(xywh)[0], 1))
    t = tf.reshape(t, (tf.shape(xywh)[0], 1))
    r = tf.reshape(r, (tf.shape(xywh)[0], 1))
    b = tf.reshape(b, (tf.shape(xywh)[0], 1))

    with tf.control_dependencies(
        [tf.assert_non_negative(r - l),
         tf.assert_non_negative(b - t)]):
        ltrb = tf.concat([l, t, r, b], axis=1)
    return ltrb
