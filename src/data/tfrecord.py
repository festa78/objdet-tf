import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import os
import PIL.Image as Image
import tensorflow as tf
import tqdm


def write_tfrecord(data_list, output_dir, batch_size_per_file=100):
    """Write .tfrecord file for semantic segmentation.

    Parameters
    ----------
    data_list: dict
        A dict which contains image and label path list
        under multiple data categories, e.g. 'train', 'val', and 'test'.
    batch_size_per_file: int, default 100
        Each .tfrecord file contains this batch size at maximum.
    """
    for data_category in data_list.keys():
        assert set(('image_list', 'labels')) <= set(data_list[data_category].keys())

        file_basename = os.path.join(output_dir, data_category)
        for i, (image_path, labels) in tqdm.tqdm(
                enumerate(
                    zip(data_list[data_category]['image_list'],
                        data_list[data_category]['labels']))):
            # Create new tfrecord if a current file exceeds @p batch_size_per_file.
            if i % batch_size_per_file == 0:
                if i != 0:
                    writer.close()

                filename = file_basename + '_{:04d}.tfrecord'.format(
                    int(i / batch_size_per_file))
                writer = tf.python_io.TFRecordWriter(filename)

                logging.info('Start writing {} data to {}'.format(
                    data_category, filename))

            filename = image_path

            image = np.array(Image.open(image_path))
            height = image.shape[0]
            width = image.shape[1]

            image_raw = image.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[height])),
                        'width':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[width])),
                        'image_raw':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_raw])),
                        'x':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=labels['x'])),
                        'y':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=labels['y'])),
                        'w':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=labels['w'])),
                        'h':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=labels['h'])),
                        'id':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=labels['id'])),
                        'filename':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[str.encode(filename)]))
                    }))
            writer.write(example.SerializeToString())
        writer.close()


def read_tfrecord(file_path, cycle_length=5, num_parallel_calls=10):
    """Read .tfrecord file for semantic segmentation.

    Parameters
    ----------
    file_path: str
        It can contain regex expression to grab all files match
        to the expression.
    cycle_length: int
        The parameter to the parallel_interleave function.
    num_parallel_calls: int
        The parameter to the map function.

    Returns
    -------
    dataset: tf.data.Dataset
        A parsed dataset.
    """

    files = tf.data.Dataset.list_files(file_path)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=cycle_length))
    dataset = dataset.map(_parse_bytes_sample, num_parallel_calls)
    return dataset


def _parse_bytes_sample(bytedata):
    features = tf.parse_single_example(
    # serialized_example,
        bytedata,
    # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'x': tf.VarLenFeature(tf.float32),
            'y': tf.VarLenFeature(tf.float32),
            'w': tf.VarLenFeature(tf.float32),
            'h': tf.VarLenFeature(tf.float32),
            'id': tf.VarLenFeature(tf.int64),
            'filename': tf.FixedLenFeature([], tf.string)
        })

    filename = features['filename']

    height_org = tf.cast(features['height'], tf.int32)
    width_org = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height_org, width_org, 3])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)

    x = tf.sparse_tensor_to_dense(features['x'])
    y = tf.sparse_tensor_to_dense(features['y'])
    w = tf.sparse_tensor_to_dense(features['w'])
    h = tf.sparse_tensor_to_dense(features['h'])
    class_id = tf.sparse_tensor_to_dense(features['id'])
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    w = tf.cast(w, tf.float64)
    h = tf.cast(h, tf.float64)
    class_id = tf.cast(class_id, tf.int32)

    sample = {
        'height': height_org,
        'width': width_org,
        'image': image,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'id': class_id,
        'filename': filename
    }
    return sample
