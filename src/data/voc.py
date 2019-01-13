"""Dataset utils for VOC2007 dataset.
"""

from collections import namedtuple
import os
import xml.etree.ElementTree as ET

import glob

import project_root

# yapf: disable
# The voc label information following the format from:
# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 0 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   color
    Label(  'background'           ,  0 ,        0 , (  0,  0,  0) ),
    Label(  'aeroplane'            ,  1 ,        1 , (128,  0,  0) ),
    Label(  'bicycle'              ,  2 ,        2 , (  0,128,  0) ),
    Label(  'bird'                 ,  3 ,        3 , (128,128,  0) ),
    Label(  'boat'                 ,  4 ,        4 , (  0,  0,128) ),
    Label(  'bottle'               ,  5 ,        5 , (128,  0,128) ),
    Label(  'bus'                  ,  6 ,        6 , (  0,128,128) ),
    Label(  'car'                  ,  7 ,        7 , (128,128,128) ),
    Label(  'cat'                  ,  8 ,        8 , ( 64,  0,  0) ),
    Label(  'chair'                ,  9 ,        9 , (192,  0,  0) ),
    Label(  'cow'                  , 10 ,       10 , ( 64,128,  0) ),
    Label(  'diningtable'          , 11 ,       11 , (192,128,  0) ),
    Label(  'dog'                  , 12 ,       12 , ( 64,  0,128) ),
    Label(  'horse'                , 13 ,       13 , (192,  0,128) ),
    Label(  'motorbike'            , 14 ,       14 , ( 64,128,128) ),
    Label(  'person'               , 15 ,       15 , (192,128,128) ),
    Label(  'pottedplant'          , 16 ,       16 , (  0, 64,  0) ),
    Label(  'sheep'                , 17 ,       17 , (128, 64,  0) ),
    Label(  'sofa'                 , 18 ,       18 , (  0,192,  0) ),
    Label(  'train'                , 19 ,       19 , (128,192,  0) ),
    Label(  'tvmonitor'            , 20 ,       20 , (  0, 64,128) ),
    Label(  'undefined'            ,255 ,      255 , (  0,  0,  0) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

name2label      = { label.name    : label for label in labels           }
id2label        = { label.id      : label for label in labels           }
trainId2label   = { label.trainId : label for label in reversed(labels) }
id2trainId      = { label.id      : label.trainId for label in labels   }
# yapf: enable


def get_file_path(input_dir, train_list, val_list):
    """Parse data and get file path list.

    Parameters
    ----------
    input_dir: str
        The directory path of Cityscapes data.
        Assume original file structure.
    train_list: str
        The list of train file basenames.
    val_list: str
        The list of val file basenames.

    Returns
    -------
    data_list: dict
        The dictinary which contains a list of image and label pair.
    """
    # Constants.
    IMAGE_ROOT = 'JPEGImages'
    LABEL_ROOT = 'Annotations'

    image_dir = os.path.join(
        os.path.abspath(os.path.expanduser(input_dir)), IMAGE_ROOT)
    label_dir = os.path.join(
        os.path.abspath(os.path.expanduser(input_dir)), LABEL_ROOT)

    train_image_list = [os.path.join(image_dir, f + '.jpg') for f in train_list]
    train_label_list = [os.path.join(label_dir, f + '.xml') for f in train_list]
    val_image_list = [os.path.join(image_dir, f + '.jpg') for f in val_list]
    val_label_list = [os.path.join(label_dir, f + '.xml') for f in val_list]

    data_list = {}
    data_list['train'] = {
        'image_list': train_image_list,
        'label_list': train_label_list
    }
    data_list['val'] = {
        'image_list': val_image_list,
        'label_list': val_label_list
    }
    return data_list


def parse_label_files(label_list):
    """Parse label file and get box coordinate
    (l, t, r, b) and its class id.

    Parameters
    ----------
    label_list: list
        List of .xml label file to parse.

    Returns
    -------
    labels: list
        List of dictionary which represents box coordinates
        and their classes on each single image.
        The dictionary contains follows:
        'l': list of float
            Left position of box on horizontal axis.
        't': list of float
            Top position of box on vertical axis.
        'r': list of float
            Right position of box on horizontal axis.
        'b': list of float
            Bottom position of box on vertical axis.
        'id': list of int
            Class id of object in a box.
    """
    labels = []

    for filename in label_list:
        boxes_and_classes = {'l': [], 't': [], 'r': [], 'b': [], 'id': []}
        tree = ET.parse(filename)
        root = tree.getroot()

        for member in root.findall('object'):
            bbox = member.find('bndbox')
            xmin = int(bbox[0].text)
            ymin = int(bbox[1].text)
            xmax = int(bbox[2].text)
            ymax = int(bbox[3].text)

            boxes_and_classes['l'].append(xmin)
            boxes_and_classes['t'].append(ymin)
            boxes_and_classes['r'].append(xmax)
            boxes_and_classes['b'].append(ymax)
            boxes_and_classes['id'].append(name2label[member[0].text].id)

        labels.append(boxes_and_classes)

    return labels
