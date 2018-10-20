"""Test set for bounding box handling utility classes.
"""

import numpy as np
import unittest

import project_root

from src.utils.bbox_handler import generate_anchor_priors


class Test(unittest.TestCase):
    def test_generate_anchor_priors(self):
        """Check that it can generate expected anchors.
        """
        IMAGE_SHAPE = (100, 100)

        # Regression.
        gt = np.array([50., 50., 142., 284.])
        grid_size = (1, 1)
        scale = (2,)
        aspect_ratio = (2.,)
        anchor_priors = generate_anchor_priors(IMAGE_SHAPE, grid_size, scale, aspect_ratio)
        np.testing.assert_array_equal(np.squeeze(anchor_priors), gt)

        # Multiple scale and aspect ratio.
        gt = np.array([
            [50., 50., 282., 140.],
            [50., 50., 142., 284.],
            [50., 50., 564., 280.],
            [50., 50., 284., 568.],
            [50., 50., 1128., 560.],
            [50., 50., 568., 1136.],
        ])
        scale = (2, 4, 8)
        aspect_ratio = (.5, 2.)
        anchor_priors = generate_anchor_priors(IMAGE_SHAPE, grid_size, scale, aspect_ratio)
        np.testing.assert_array_equal(np.squeeze(anchor_priors), gt)

        # With grid.
        gt = np.array([[
            [25., 25., 70., 140.],
            [75., 25., 70., 140.]
        ],[
            [25., 75., 70., 140.],
            [75., 75., 70., 140.],
        ]])
        grid_size = (2, 2)
        scale = (2,)
        aspect_ratio = (2.,)
        anchor_priors = generate_anchor_priors(IMAGE_SHAPE, grid_size, scale, aspect_ratio)
        np.testing.assert_array_equal(np.squeeze(anchor_priors), gt)
