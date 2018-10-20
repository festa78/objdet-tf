"""Utility class for bounding box handling.
"""

import numpy as np

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
    anchor_width_scaled = np.repeat(scale, (len(aspect_ratio),)) * np.tile(anchor_width, (len(scale),))
    anchor_height_scaled = np.repeat(scale, (len(aspect_ratio),)) * np.tile(anchor_height, (len(scale),))
    anchor_shapes = np.stack((anchor_width_scaled, anchor_height_scaled), 1)

    # Integrate into anchor_priors.
    anchor_priors[..., 2:] = anchor_shapes.reshape((1, 1, num_anchors, 2))

    return anchor_priors


class AnchorConverter:
    """Manage to generate anchor targets from anchor priors.
    Make sure consistent conversions between absolute bounding box
    coordinates and anchor target coordinates with respect to anchor_priors.

    Parameters
    ----------
    anchor_priors: (grid_height, grid_width, num_anchors, (x, y, w, h) box coordinate)
                   numpy float array
        Anchor priors in (x, y, w ,h) coordinate.
    """

    def __init__(self, anchor_priors):
        self.anchor_priors = anchor_priors

    def generate_anchor_targets(self, gt_bboxes, iou_threshold=.5):
        """Given ground truth bounding boxes and anchor priors,
        compute IoU between them. If IoU is larger than @p iou_threshold,
        assign the corresponding anchor to object and regression.
        """
        pass
