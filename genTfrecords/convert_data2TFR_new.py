#!/usr/bin/env python
"""
convert lfw or celebext dataset to TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from random import shuffle
import glob

import cv2
import argparse
import os
import sys
import logging


# convert data to attr
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # load images


def load_image(addr):  # A function to Load image
    img = cv2.imread(addr)
    #   img = img[44:172, 40:136]
    img = img[160:480, 96:416]  # 320x320
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.uint8)
    return img


def main(unused_args):
    # avoid warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # datefmt='%a, %d %b %Y %H:%M:%S',
                        # filename='train.log',
                        filemode='w')
    shuffle_data = FLAGS.shuffle
    dir_path = FLAGS.directory  # work directory

    image_path = ''
    label_path = ''
    train_filename = ''
    dataset_prefix = ''
    # select dataset
    if FLAGS.dataset == 'celebext':
        image_path = os.path.join(dir_path, 'msceleb1m_images.txt')
        label_path = os.path.join(dir_path, 'msceleb1m_labels.txt')
        # train_filename = os.path.join('/data1/face', 'dataset', 'celebext_align.tfrecords')  # tfrecords path
        # dataset_prefix = 'celebext_align'
    elif FLAGS.dataset == 'lfw':
        image_path = os.path.join(dir_path, 'dataset', 'lfw_test_data.txt')
        label_path = os.path.join(dir_path, 'dataset', 'lfw_test_label.txt')
        train_filename = os.path.join('/data1/face/', 'dataset', 'lfw.tfrecords')  # tfrecords path
        dataset_prefix = 'lfw'
    elif FLAGS.dataset == 'megaface':
        image_path = os.path.join(dir_path, 'dataset', 'megaface_train_data_min_10.txt')
        label_path = os.path.join(dir_path, 'dataset', 'megaface_train_label_min_10.txt')
        train_filename = os.path.join('/data/face/', 'dataset', 'megaface_min_10_label_from_msceleb1m.tfrecords')
        # train_filename = os.path.join(dir_path, 'dataset', 'megaface_train.tfrecords')  # tfrecords path
        dataset_prefix = 'megaface'  # several folder
    elif FLAGS.dataset == 'cloud':
        image_path = os.path.join(dir_path, 'dataset', 'cloud_train_data_min_10.txt')
        label_path = os.path.join(dir_path, 'dataset', 'cloud_train_label_min_10.txt')
        train_filename = os.path.join('/data/face/', 'dataset', 'cloud_train_min_10_label_from_msceleb1m.tfrecords')
        # train_filename = os.path.join(dir_path, 'dataset', 'megaface_train.tfrecords')  # tfrecords path
        dataset_prefix = 'jdb'  # several folder
    elif FLAGS.dataset == 'vggface2':
        image_path = os.path.join(dir_path, 'dataset', 'vggface2_train_data.txt')
        label_path = os.path.join(dir_path, 'dataset', 'vggface2_train_label.txt')
        train_filename = os.path.join('/data/face/', 'dataset', 'vggface2_train.tfrecords')  # tfrecords path
        dataset_prefix = 'vggface2'

    else:
        raise Exception('dataset not exists')
    # read paths of images and labels to lists
    f = open(image_path, 'r')
    f.seek(0)
    addrs = f.readlines()
    f.close()
    f = open(label_path, 'r')
    f.seek(0)
    labels = f.readlines()[1:]  # leave the 1st line descriptor
    f.close()

    # shuffle datasets
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)  # unzip

    # allocate train,val,test datasets
    '''
    train_addrs = addrs[0:int(0.7*len(addrs))]
    train_labels = labels[0:int(0.7*len(labels))]

    val_addrs = addrs[int(0.7*len(addrs)):int(0.9*len(addrs))]
    val_labels = labels[int(0.7*len(labels)):int(0.9*len(labels))]

    test_addrs = addrs[int(0.9*len(addrs)):]
    test_labels = labels[int(0.9*len(labels)):]
    '''
    # only have train dataset
    train_addrs = addrs
    train_labels = labels

    # write data to TFRecord
    # create a writer for tfrecords
    for k in range(5):
        train_addrs_ = train_addrs[1000000*k:1000000*k+1000000]
        train_labels_ = train_labels[1000000*k:1000000*k+1000000]
        writer = tf.python_io.TFRecordWriter('/data1/msceleb1m_%d.tfrecords'%k)
        logging.info('convert {} to tfrecords starts...'.format(FLAGS.dataset))
        # convert train set to tfrecords
        for i in range(len(train_addrs_)):
            if not i % 1000:
                logging.info('Train data: {}/{}'.format(i, len(train_addrs_)))
            img_path = os.path.join('', train_addrs_[i].strip())
            # img_path = os.path.join(dir_path, 'dataset/' + dataset_prefix, train_addrs[i].strip())
            '''
                if os.path.exists(img_path):
                    message = 'OK, the %s file exists.'
                else:
                    message = 'Sorry, I cannot find the %s file.'
                logging.info(message % img_path)
                '''
            try:
                # load image
                img = load_image(img_path)
            except:
                logging.info('%s read failed!' % img_path)
                continue
            # get label, symbol
            label = int(train_labels_[i].strip())
            symbol = k*i+i
            # create attr
            feature = {'symbol':  _int64_feature(symbol),
                       'train/label': _int64_feature(label),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

            # create example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # write example protocol buffer to file
            writer.write(example.SerializeToString())
        logging.info('convert {} to tfrecords completed: {} images!'.format(FLAGS.dataset, len(train_addrs_)))
        writer.close()
        sys.stdout.flush()


# .............................................................................

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/data/face/',
        help='Directory to the work directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='celebext',  # celebext
        help='which dataset to convert, celebext,lfw,megaface',
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help='shuffle dataset'
    )

FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

