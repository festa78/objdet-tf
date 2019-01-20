import numpy as np
import pytest
import tensorflow as tf

from src.models.yolo_v2 import YOLOV2
from src.pipelines.trainer import Trainer
from src.utils.detection_losses import yolo_detection_loss
from src.utils.bbox_handler import AnchorConverter, generate_anchor_priors


def _setup_trainer(tmpdir):
    """Setup a simple Trainer class instance.
    """
    SAMPLE_SIZE = 8
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 1
    NUM_CLASSES = 3
    SCALE = (8, 16, 32)
    ASPECT_RATIO = (.5, 1., 2.)
    NUM_ANCHORS = 9
    NUM_EPOCHS = 2
    LABEL_TEMPLATE = np.array([0., 0., 0.9, 0.9, 1, 1])
    LEARNING_RATE = .1

    model = YOLOV2
    grid_size = (model.GRID_H, model.GRID_W)
    num_anchors = len(SCALE) * len(ASPECT_RATIO)
    eval_epochs = NUM_CLASSES

    image_height = model.GRID_H * model.SCALE
    image_width = model.GRID_W * model.SCALE

    with tf.device('/cpu:0'):
        anchor_priors = generate_anchor_priors(grid_size, SCALE, ASPECT_RATIO)
        anchor_converter = AnchorConverter(anchor_priors)

        train_batch = {
            'image':
            tf.convert_to_tensor(
                np.ones([SAMPLE_SIZE, image_height, image_width, 3]),
                dtype=tf.float32),
            'label':
            tf.convert_to_tensor(
                np.tile(
                    LABEL_TEMPLATE,
                    [SAMPLE_SIZE, model.GRID_H, model.GRID_W, num_anchors, 1]),
                dtype=tf.float32),
        }
        val_batch = {
            'image':
            tf.convert_to_tensor(
                np.ones([SAMPLE_SIZE, image_height, image_width, 3]),
                dtype=tf.float32),
            'label':
            tf.convert_to_tensor(
                np.tile(
                    LABEL_TEMPLATE,
                    [SAMPLE_SIZE, model.GRID_H, model.GRID_W, num_anchors, 1]),
                dtype=tf.float32),
        }
        train_dataset = tf.data.Dataset.from_tensor_slices(train_batch)
        train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE)
        train_iterator = train_dataset.make_initializable_iterator()
        val_dataset = tf.data.Dataset.from_tensor_slices(val_batch)
        val_dataset = val_dataset.batch(VAL_BATCH_SIZE)
        val_iterator = val_dataset.make_initializable_iterator()
        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model_ins = model(NUM_CLASSES, num_anchors)

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        dut = Trainer(
            model_ins,
            NUM_CLASSES,
            TRAIN_BATCH_SIZE,
            VAL_BATCH_SIZE,
            train_iterator,
            val_iterator,
            anchor_converter,
            yolo_detection_loss,
            optimizer,
            global_step,
            str(tmpdir),
            num_epochs=NUM_EPOCHS,
            evaluate_epochs=eval_epochs)

    return dut


class Test(tf.test.TestCase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir

    def test_trainer_update(self):
        """Test Trainer class surely updates the parameters.
        cf. https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
        """
        with self.test_session() as sess:
            tf.set_random_seed(1234)
            dut = _setup_trainer(self.tmpdir)

            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables(scope='model'))

            dut.train(sess)

            after = sess.run(tf.trainable_variables(scope='model'))

            for b, a in zip(before, after):
                # Make sure something changed.
                self.assertTrue((b != a).any())

    def test_compute_metrics(self):
        """Test Trainer class surely compute mean loss and IoU.
        """
        with self.test_session() as sess:
            tf.set_random_seed(1234)
            dut = _setup_trainer(self.tmpdir)

            sess.run(tf.global_variables_initializer())
            sess.run((dut.train_iterator.initializer,
                      dut.train_metric_reset_op))

            train_mloss = sess.run(dut.train_mean_loss)

            # Without update, it should be zero.
            self.assertEqual(train_mloss, 0.)

            sess.run((dut.train_op, dut.train_mean_loss_update_op))

            train_mloss = sess.run(dut.train_mean_loss)

            # After update.
            self.assertAlmostEqual(train_mloss, 5.2298584)
