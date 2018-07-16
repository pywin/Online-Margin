#!/usr/bin/env python
"""
extract features from lfw TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import math
sys.path.insert(0, "/data1")

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from nets.L_Resnet_E_IR_test import get_resnet
#sys.path.append("/home/linkface/teng")

# resnet networks from slim

def main(unused_args):
    """Train face classifier for a number of epochs."""

    print('>>>>>>>>>>tensorflow version: %s<<<<<<<<<<<<<<<' % (tf.__version__))

    # avoid warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # tfrecords path
    dir_path = FLAGS.directory
    data_path = os.path.join('/data1', 'lfw.tfrecords')

    # checkpoint path
    checkpoint_dir = os.path.join('/data1/output005_100k', 'ckpt')

    # feature path
    feature_dir = os.path.join('/data1', 'feature', 'lfw_features_005_100k.mat')
    
    #img_inputs = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32) 
    with tf.Session() as sess:
        # create attr feature
        feature = {'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64),
                   'channel': tf.FixedLenFeature([], tf.int64),
                   'train/label': tf.FixedLenFeature([], tf.int64),
                   'train/image': tf.FixedLenFeature([], tf.string)}
        # return a queuerunner, with a FIFOQueue inside
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=FLAGS.num_epochs, shuffle=False)
        # create a reader, read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # parse a record
        features = tf.parse_single_example(serialized_example, features=feature)

        # decode the string to image
        rows = tf.cast(features['height'], tf.int32)
        cols = tf.cast(features['width'], tf.int32)
        chans = tf.cast(features['channel'], tf.int32)
        image = tf.decode_raw(features['train/image'], tf.uint8)
        # print([rows, cols, chans])

        # reshape the image
        image = tf.reshape(image, [224, 224, 3])
        # resize image
        image = tf.image.resize_images(image, (112, 112), 0)
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1.0 / 255) - 0.5
        # cast label to be int32
        label = tf.cast(features['train/label'], tf.int32)

        # create a Queue
        images, labels = tf.train.batch([image, label], batch_size=FLAGS.batch_size,
                                                num_threads=1,
                                                capacity=500 + 3 * FLAGS.batch_size)
        # networks
        # ...................resnet.............................
        # face_net
        NUM_CLASSES = 14294
        NUM_FEATURE = 512
        # networks output
	#with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #    net, end_points = resnet_v2.resnet_v2_50(images,num_classes=512, is_training=False)
        #w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        #tl.layers.set_name_reuse(True)
        #net=get_resnet(img_inputs, 101, type='ir', trainable=False, reuse=tf.AUTO_REUSE)
        net=get_resnet(images, 50, type='ir', trainable=False, reuse=tf.AUTO_REUSE)
        # .....................................................
        # feat = end_points['features'] 
        feat  = net.outputs
        print(feat)

        # initializing
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # defaults to saving all variables
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # deploy networks
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for v in tf.global_variables():
               pass 
               #print(v.name)
        else:
            raise Exception('no checkpoint is found!')

        try:
            batch = 0
            feature_dic = {}
            feature_array = np.empty(shape=[0, 512], dtype=float)
            print('Start extracting features from lfw dataset...')
	    # start training
            while not coord.should_stop():
                batch += 1
                start_time = time.time()
                #images_ = sess.run(images)
                #feed_dict = {img_inputs:images_}
                #feed_dict.update(net.all_drop)
                #lfw_features = sess.run([feat],feed_dict=feed_dict)
                lfw_features = sess.run([feat])
                lfw_features = np.array(lfw_features)
                # lfw_features  = sess.run([end_points])
                duration = time.time() - start_time
		#print(np.shape(lfw_features))
                feature_array = np.concatenate((feature_array,np.squeeze(lfw_features)),axis=0)
		
                if batch % 10 == 0:
                     # print('batch %d: acc = %.4f (%.3f sec/batch)' % (batch, acc, duration)
                     print('Extracting batch %d(total batch %d): (%.3f sec/batch)' % (batch, math.ceil(13233/FLAGS.batch_size), duration))

                if FLAGS.batch_size * batch > 13233 :
                    feature_array = feature_array[0:13233,:]

                    print('Successfully extract 13233 features from lfw dataset!')
                    break
            feature_dic['feature'] = feature_array
            print('Features saved to %s!' % (feature_dir))
            sio.savemat(feature_dir, feature_dic)

        except tf.errors.OutOfRangeError:
	   # print(tf.trainable_variables())
            print('Done running over for %d epochs, %d batches.' % (FLAGS.num_epochs, batch))
        finally:
            # close threads
            coord.request_stop()
            coord.join(threads)
            sess.close()


# .............................................................................
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size.'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='/home/linkface/chenxinhua/face_classification_new/',
        help='Directoryto the work directory.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

