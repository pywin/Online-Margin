#-*- coding:utf-8 -*-
import os
import tensorflow as tf
from PIL import Image

# for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)
#
#     image = example.features.feature['img_raw'].bytes_list.value
#     label = example.features.feature['label'].int64_list.value
#     symbol = example.features.feature['symbol'].int64_list.value
#     # 可以做一些预处理之类的
#     print(image)
image_size = [112,112]
images_train = tf.placeholder(name='img_inputs', shape=[None, *image_size, 3], dtype=tf.float32)
labels_train = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
tfrecords_f = tf.constant('train.tfrecords')
dataset = tf.data.TFRecordDataset(tfrecords_f)
#dataset = dataset.map(_augment_function)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    images_train, labels_train = sess.run(next_element)
    print(images_train)
    print(labels_train)
