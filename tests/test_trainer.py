import numpy as np
import pytest
import tensorflow as tf

from src.models.yolo_v2 import YOLOV2
from src.pipelines.trainer import Trainer
from src.utils.detection_losses import yolo_detection_loss


def _setup_trainer(tmpdir):
    """Setup a simple Trainer class instance.
    """
    BATCH_SIZE = 4
    NUM_CLASSES = 3
    NUM_ANCHORS = 1
    NUM_EPOCHS = 2
    EVALUATE_EPOCHS = 10
    model = YOLOV2

    image_height = model.GRID_H * model.SCALE
    image_width = model.GRID_W * model.SCALE

    with tf.device('/cpu:0'):
        train_batch = {
            'image':
            tf.convert_to_tensor(
                np.ones([BATCH_SIZE, image_height, image_width, 3]),
                dtype=tf.float32),
            'label':
            tf.convert_to_tensor(
                np.ones(
                    [BATCH_SIZE, model.GRID_H, model.GRID_W, NUM_ANCHORS, 6]),
                dtype=tf.int64),
        }
        val_batch = {
            'image':
            tf.convert_to_tensor(
                np.ones([BATCH_SIZE, image_height, image_width, 3]),
                dtype=tf.float32),
            'label':
            tf.convert_to_tensor(
                np.ones(
                    [BATCH_SIZE, model.GRID_H, model.GRID_W, NUM_ANCHORS, 6]),
                dtype=tf.int64),
        }
        train_dataset = tf.data.Dataset.from_tensor_slices(train_batch)
        train_dataset = train_dataset.batch(2)
        train_iterator = train_dataset.make_initializable_iterator()
        val_dataset = tf.data.Dataset.from_tensor_slices(val_batch)
        val_dataset = val_dataset.batch(2)
        val_iterator = val_dataset.make_initializable_iterator()
        global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.device('/gpu:0'):
        model_ins = model(NUM_CLASSES, NUM_ANCHORS)

        optimizer = tf.train.AdamOptimizer()

        dut = Trainer(
            model_ins,
            NUM_CLASSES,
            train_iterator,
            val_iterator,
            yolo_detection_loss,
            optimizer,
            global_step,
            str(tmpdir),
            num_epochs=NUM_EPOCHS,
            evaluate_epochs=EVALUATE_EPOCHS)

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
                self.assertNotAllClose(b, a)

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
            self.assertAlmostEqual(train_mloss, 6.2052207)
