import logging
logging.basicConfig(level=logging.INFO)

import tensorflow as tf

import project_root

from src.models.common import Common


class YOLOV2(Common):
    """YOLO v2 implementation.
    cf. https://arxiv.org/abs/1612.08242
    """
    # The output feature map size.
    SCALE = 32
    GRID_H = 10
    GRID_W = 18

    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Necessary for batch normalization.
        self.training = True

        # Prepare layers.
        # DarkNet19.
        self.conv1_1 = self._make_conv2d(
            out_channels=32, kernel_size=3, stride=1, bias=True, name='conv1_1')
        self.conv1_1_bn = self._make_bn2d(
            training=self.training, name='conv1_1/bn')
        self.pool1 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool1')

        self.conv2_1 = self._make_conv2d(
            out_channels=64, kernel_size=3, stride=1, bias=True, name='conv2_1')
        self.conv2_1_bn = self._make_bn2d(
            training=self.training, name='conv2_1/bn')
        self.pool2 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool2')

        self.conv3_1 = self._make_conv2d(
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv3_1')
        self.conv3_1_bn = self._make_bn2d(
            training=self.training, name='conv3_1/bn')
        self.conv3_2 = self._make_conv2d(
            out_channels=64, kernel_size=1, stride=1, bias=True, name='conv3_2')
        self.conv3_2_bn = self._make_bn2d(
            training=self.training, name='conv3_2/bn')
        self.conv3_3 = self._make_conv2d(
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv3_3')
        self.conv3_3_bn = self._make_bn2d(
            training=self.training, name='conv3_3/bn')
        self.pool3 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool3')

        self.conv4_1 = self._make_conv2d(
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv4_1')
        self.conv4_1_bn = self._make_bn2d(
            training=self.training, name='conv4_1/bn')
        self.conv4_2 = self._make_conv2d(
            out_channels=128,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv4_2')
        self.conv4_2_bn = self._make_bn2d(
            training=self.training, name='conv4_2/bn')
        self.conv4_3 = self._make_conv2d(
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv4_3')
        self.conv4_3_bn = self._make_bn2d(
            training=self.training, name='conv4_3/bn')
        self.pool4 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool4')

        self.conv5_1 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5_1')
        self.conv5_1_bn = self._make_bn2d(
            training=self.training, name='conv5_1/bn')
        self.conv5_2 = self._make_conv2d(
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv5_2')
        self.conv5_2_bn = self._make_bn2d(
            training=self.training, name='conv5_2/bn')
        self.conv5_3 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5_3')
        self.conv5_3_bn = self._make_bn2d(
            training=self.training, name='conv5_3/bn')
        self.conv5_4 = self._make_conv2d(
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv5_4')
        self.conv5_4_bn = self._make_bn2d(
            training=self.training, name='conv5_4/bn')
        self.conv5_5 = self._make_conv2d(
            out_channels=512,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv5_5')
        self.conv5_5_bn = self._make_bn2d(
            training=self.training, name='conv5_5/bn')
        self.pool5 = self._make_max_pool2d(
            kernel_size=2, stride=2, name='pool5')

        self.conv6_1 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv6_1')
        self.conv6_1_bn = self._make_bn2d(
            training=self.training, name='conv6_1/bn')
        self.conv6_2 = self._make_conv2d(
            out_channels=512,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv6_2')
        self.conv6_2_bn = self._make_bn2d(
            training=self.training, name='conv6_2/bn')
        self.conv6_3 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv6_3')
        self.conv6_3_bn = self._make_bn2d(
            training=self.training, name='conv6_3/bn')
        self.conv6_4 = self._make_conv2d(
            out_channels=512,
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv6_4')
        self.conv6_4_bn = self._make_bn2d(
            training=self.training, name='conv6_4/bn')
        self.conv6_5 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv6_5')
        self.conv6_5_bn = self._make_bn2d(
            training=self.training, name='conv6_5/bn')
        # DarkNet19 ends.

        # Detection head.
        self.conv7_1 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv7_1')
        self.conv7_1_bn = self._make_bn2d(
            training=self.training, name='conv7_1/bn')
        self.conv7_2 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv7_2')
        self.conv7_2_bn = self._make_bn2d(
            training=self.training, name='conv7_2/bn')

        self.conv8_1 = self._make_conv2d(
            out_channels=64, kernel_size=3, stride=1, bias=True, name='conv8_1')
        self.conv8_1_bn = self._make_bn2d(
            training=self.training, name='conv8_1/bn')
        self.space_to_depth8 = self._make_space_to_depth(size=2, name='space_to_depth8')
        self.concat8 = self._make_concat(axis=3, name='concat8')

        self.conv9_1 = self._make_conv2d(
            out_channels=1024,
            kernel_size=3,
            stride=1,
            bias=True,
            name='conv9_1')
        self.conv9_1_bn = self._make_bn2d(
            training=self.training, name='conv9_1/bn')

        # Per anchor, it predicts regression (tx, ty, tw, th), object-ness,
        # and class logits.
        self.conv_final = self._make_conv2d(
            out_channels=self.num_anchors * (5 + self.num_classes),
            kernel_size=1,
            stride=1,
            bias=True,
            name='conv_final')
        self.conv_final_bn = self._make_bn2d(
            training=self.training, name='conv_final/bn')

    def __call__(self, x):
        """Forward the input tensor through the network.
        Managed by variable_scope to know which model includes
        which variable.
        TODO: make variable_scope shorter but do the same.

        Parameters
        ----------
        x: (N, H, W, C) tf.Tensor
            Input tensor to process.

        Returns
        -------
        out: (N, GRID_H, GRID_W, NUM_ANCHORS, 5 + NUM_CLASSES) tf.Tensor
            The output tensor of the network.
            In the last dimension, the first 4 elements predict
            regression (tx, ty, tw, th), the 5th element predicts objectiveness,
            and the last NUM_CLASSES elements predict class logits.
        """

        with tf.variable_scope('yolo_v2', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('darknet_19', reuse=tf.AUTO_REUSE):
                out = self.conv1_1(x)
                out = self.conv1_1_bn(out)
                out = self.pool1(out)

                out = self.conv2_1(out)
                out = self.conv2_1_bn(out)
                out = self.pool2(out)

                out = self.conv3_1(out)
                out = self.conv3_1_bn(out)
                out = self.conv3_2(out)
                out = self.conv3_2_bn(out)
                out = self.conv3_3(out)
                out = self.conv3_3_bn(out)
                out = self.pool3(out)

                out = self.conv4_1(out)
                out = self.conv4_1_bn(out)
                out = self.conv4_2(out)
                out = self.conv4_2_bn(out)
                out = self.conv4_3(out)
                out = self.conv4_3_bn(out)
                out = self.pool4(out)

                out = self.conv5_1(out)
                out = self.conv5_1_bn(out)
                out = self.conv5_2(out)
                out = self.conv5_2_bn(out)
                out = self.conv5_3(out)
                out = self.conv5_3_bn(out)
                out = self.conv5_4(out)
                out = self.conv5_4_bn(out)
                out = self.conv5_5(out)
                passthrough = self.conv5_5_bn(out)
                out = self.pool5(passthrough)

                out = self.conv6_1(out)
                out = self.conv6_1_bn(out)
                out = self.conv6_2(out)
                out = self.conv6_2_bn(out)
                out = self.conv6_3(out)
                out = self.conv6_3_bn(out)
                out = self.conv6_4(out)
                out = self.conv6_4_bn(out)
                out = self.conv6_5(out)
                out = self.conv6_5_bn(out)

            with tf.variable_scope('detection_head', reuse=tf.AUTO_REUSE):
                out = self.conv7_1(out)
                out = self.conv7_1_bn(out)
                out = self.conv7_2(out)
                out = self.conv7_2_bn(out)

                passthrough = self.conv8_1(passthrough)
                passthrough = self.conv8_1_bn(passthrough)
                passthrough = self.space_to_depth8(passthrough)
                out = self.concat8([out, passthrough])

                out = self.conv9_1(out)
                out = self.conv9_1_bn(out)

                out = self.conv_final(out)
                out = self.conv_final_bn(out)

                out = tf.reshape(
                    out,
                    shape=(-1, self.GRID_H, self.GRID_W, self.num_anchors,
                          5 + self.num_classes))

            return out
