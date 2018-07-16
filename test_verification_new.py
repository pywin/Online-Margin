import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np
import scipy.io as sio
"""
k-fold verification
feature_path:feature.mat
pairs_path:pairs.mat
aver_acc:return accuracy
aver_th:return threshhold
"""

def test_verification(feature_path, pairs_path, num_pairs, k_num=10):
    print('\n-------------------------------------------------------------------------')
    print('==========10 fold cross-validation : %d pairs of faces=========='%(num_pairs))
    print('-------------------------------------------------------------------------\n')

    # read feature
    features = sio.loadmat(feature_path)['feature']
    print('feature shape: ', features.shape)
    # load pairs.mat,  each line is 'id1,id2,target_test'
    pairs = sio.loadmat(pairs_path)
    id1 = np.squeeze(pairs['id1'])
    id2 = np.squeeze(pairs['id2'])
    target_test = np.squeeze(pairs['target_test'])
    print('pairs shape: ', target_test.shape)

    num_pairs = 6000 # number of face pairs
    assert id1.shape[0] == num_pairs
    assert id2.shape[0] == num_pairs
    assert target_test.shape[0] == num_pairs

    # retrieve feature pairs : 6000 pairs
    lfeat = np.squeeze(features[id1])
    rfeat = np.squeeze(features[id2])

    ip = np.sum(lfeat * rfeat, 1) # inner product
    d1 = np.sum(lfeat * lfeat, 1) # norm^2 
    d2 = np.sum(rfeat * rfeat, 1) # norm^2
    cosine = ip/np.sqrt(d1*d2) # consine
    print('cos value shape: ',cosine.shape)

    # 10 fold cross validation.........
    num_per_grp = int(num_pairs / k_num) # number of pairs per group
    accuracy=[] #  store accuracy for each group
    threshold=[]
    for k in range(k_num):
        print('-------------------------------------------------------------------')
        print('group: #{0}'.format(*[k+1]))

        test_ids = [ i for i in range((k+1)*num_per_grp) if i not in range(k*num_per_grp) ]
        train_ids = [ i for i in range(num_pairs) if i not in test_ids ]
        result = {} # store accuracy of each threshold
        
        # select the best threshold for train set
        for th in [(i-1000)/1000.0 for i in range(2001)]:#seq[-1,1,0.001]
            # correct pred nums
            num_correct = np.sum((cosine[train_ids] > th) == target_test[train_ids])
            train_acc = float(num_correct)/len(train_ids) # train groups accuracy
            result[th] = train_acc
        # acquire max acc and th
        thresh, acc = max(result.items(), key=lambda x: x[1])
        print('group #{0}:threshold= {1}, train accuracy= {2}'.format(*[k+1, thresh, acc]))
        # apply the threshold to test set
        # correct pred nums
        num_correct = np.sum((cosine[test_ids] > thresh) == target_test[test_ids])
        # train groups accuracy
        test_acc = float(num_correct) / len(test_ids)
        print('group #{0}:,threshold= {1},test accuracy= {2}'.format(*[k+1, thresh, test_acc]))
        accuracy.append(test_acc) # accuracy for this group
        threshold.append(thresh)
 # average accuracy for 10 fold cross validation
    aver_th = np.mean(threshold)
    aver_accu = np.mean(accuracy)
    print('\n==========================================================')
    print('========average accuracy: {0}========='.format(*[aver_accu,]))
    print('==========================================================\n')
    return aver_accu, aver_th   

if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer("--k_num", 10, "number of cross-validation groups")
    tf.app.flags.DEFINE_integer("num_pairs", 6000, "number of pairs")
    tf.app.flags.DEFINE_string("directory", "/data1", "directory to the work directory.")

    dir_path = FLAGS.directory
    feature_path = os.path.join(dir_path, 'feature', 'lfw_features_005_100k.mat')
    pairs_path = os.path.join(dir_path, 'lfw_test_pairs',  'lfw_test_pairs.mat')
    k_num =  FLAGS.k_num
    num_pairs = FLAGS. num_pairs
    #run test
    test_verification(feature_path,pairs_path,num_pairs,k_num)
