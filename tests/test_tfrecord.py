"""Test set for tfrecord writer and reader.
"""
import os
import xml.etree.ElementTree as ET

from PIL import Image
import pytest
import numpy as np
import tensorflow as tf

import project_root

from src.data.voc import get_file_path, parse_label_files
from src.data.tfrecord import write_tfrecord, read_tfrecord

# Constants.
IMAGE_ROOT = 'JPEGImages'
LABEL_ROOT = 'Annotations'
IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100
TRAIN_FILENAMES = ['train1', 'train2', 'train3']
VAL_FILENAMES = ['val1', 'val2']


def _create_data_list(tmpdir, filenames):
    np.random.seed(1234)

    image_list = []
    label_list = []
    for i, filename in enumerate(filenames):
        # Creates empty files.
        image_path = tmpdir.join(IMAGE_ROOT, filename + '.jpg')
        label_path = tmpdir.join(LABEL_ROOT, filename + '.xml')

        image = np.random.randint(255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        image = Image.fromarray(image.astype(np.uint8))

        label = ET.Element('annotation')

        objects = ET.SubElement(label, 'object')
        ET.SubElement(objects, 'name').text = 'chair'
        ET.SubElement(objects, 'pose').text = 'nan'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(0 + i)
        ET.SubElement(bndbox, 'ymin').text = str(1 + i)
        ET.SubElement(bndbox, 'xmax').text = str(20 + 2 * i)
        ET.SubElement(bndbox, 'ymax').text = str(22 + 2 * i)

        objects = ET.SubElement(label, 'object')
        ET.SubElement(objects, 'name').text = 'car'
        ET.SubElement(objects, 'pose').text = 'nan'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(80 - 2 * i)
        ET.SubElement(bndbox, 'ymin').text = str(78 - 2 * i)
        ET.SubElement(bndbox, 'xmax').text = str(100 - i)
        ET.SubElement(bndbox, 'ymax').text = str(99 - i)

        label = ET.ElementTree(label)

        # Convert path from py.path.local to str.
        image.save(str(image_path))
        label.write(str(label_path))

        image_list.append(image_path)
        label_list.append(label_path)

    return image_list, label_list


def _create_sample_voc_structure(tmpdir):
    """Creates dummy voc like data structure.

    Returns
    -------
    root_dir_path : str
        Root path to the created data structure.
    data_list : dict
        Dummy data dictionary contains 'image_list' and 'label_list'.
    """
    data_list = {}
    tmpdir.mkdir(IMAGE_ROOT)
    tmpdir.mkdir(LABEL_ROOT)

    train_image_list, train_label_list = _create_data_list(
        tmpdir, TRAIN_FILENAMES)
    data_list['train'] = {
        'image_list': train_image_list,
        'label_list': train_label_list
    }

    val_image_list, val_label_list = _create_data_list(tmpdir, VAL_FILENAMES)
    data_list['val'] = {
        'image_list': val_image_list,
        'label_list': val_label_list
    }

    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


class Test(tf.test.TestCase):

    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        self.tmpdir = tmpdir

    def test_write_read_tfrecord(self):
        """Test it can write and read the tfrecord file correctly.
        """
        # Constants.
        DATA_CATEGORY = ['train', 'val']
        GT_LABEL = np.array(
            [[[0., .01, .2, .22, 9.], [.8, .78, 1., .99, 7.]],
             [[.01, .02, .22, .24, 9.], [.78, .76, .99, .98, 7.]],
             [[.02, .03, .24, .26, 9.], [.76, .74, .98, .97, 7.]]])

        # Make a dummy tfrecord file.
        # XXX use more simple structure.
        input_dir, gt_data_list = _create_sample_voc_structure(self.tmpdir)
        output_dir = input_dir

        # Convert from py.path.local to str.
        data_list = get_file_path(input_dir, TRAIN_FILENAMES, VAL_FILENAMES)
        for v in data_list.values():
            v['labels'] = parse_label_files(v['label_list'])

        write_tfrecord(data_list, output_dir, normalize=True)

        # Read the created tfrecord file.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        for category in DATA_CATEGORY:
            dataset = read_tfrecord(
                os.path.join(output_dir, category + '_0000.tfrecord'))
            next_element = dataset.make_one_shot_iterator().get_next()
            with self.test_session() as sess:
                # The op for initializing the variables.
                sess.run(init_op)
                i = 0
                while True:
                    try:
                        sample = sess.run(next_element)
                        gt_image = np.array(
                            Image.open(open(sample['filename'].decode(),
                                            'rb')).convert('RGB'))
                        np.testing.assert_array_equal(sample['image'], gt_image)
                        self.assertEqual(sample['height'], IMAGE_HEIGHT)
                        self.assertEqual(sample['width'], IMAGE_WIDTH)
                        np.testing.assert_array_almost_equal(
                            sample['label'], GT_LABEL[i])
                        i += 1
                    except tf.errors.OutOfRangeError:
                        if category == 'train':
                            assert i == 3
                        else:
                            assert i == 2
                        break
