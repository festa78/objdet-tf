"""Test set for voc classes.
"""
import xml.etree.ElementTree as ET

from PIL import Image
import pytest
import numpy as np
import tensorflow as tf

import project_root

from src.data.voc import get_file_path, parse_label_files

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

    def test_voc_get_file_path(self):
        """Test it can get file paths from cityscapes like data structure.
        """
        input_dir, gt_data_list = _create_sample_voc_structure(self.tmpdir)
        output_dir = input_dir
        data_list = get_file_path(input_dir, TRAIN_FILENAMES, VAL_FILENAMES)
        for cat1, cat2 in zip(data_list.values(), gt_data_list.values()):
            for list1, list2 in zip(cat1.values(), cat2.values()):
                # Do not care about orders.
                self.assertEqual(set(list1), set(list2))

    def test_parse_label_files(self):
        """Test it can properly parse the voc xml label format.
        """
        # Ground truths.
        GT_LABEL = [
            {'l': [0, 80], 't': [1, 78], 'r': [20, 100], 'b': [22, 99], 'id': [9, 7]},
            {'l': [1, 78], 't': [2, 76], 'r': [22, 99], 'b': [24, 98], 'id': [9, 7]},
            {'l': [2, 76], 't': [3, 74], 'r': [24, 98], 'b': [26, 97], 'id': [9, 7]}
        ]  # yapf: disable
        input_dir, gt_data_list = _create_sample_voc_structure(self.tmpdir)
        output_dir = input_dir
        data_list = get_file_path(input_dir, TRAIN_FILENAMES, VAL_FILENAMES)

        train_labels = parse_label_files(data_list['train']['label_list'])
        val_labels = parse_label_files(data_list['val']['label_list'])

        self.assertEqual(GT_LABEL, train_labels)
        self.assertEqual(GT_LABEL[:-1], val_labels)
