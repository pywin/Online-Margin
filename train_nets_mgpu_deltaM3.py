import tensorflow as tf
import tensorlayer as tl
import argparse
from data.mx2tfrecords_new import _parse_function, _preprocess_function, _augment_function
import os
from nets.L_Resnet_E_IR_MGPU import get_resnet
from losses.face_losses_new import arcface_loss
from tensorflow.core.protobuf import config_pb2
import time
from data.eval_data_reader import load_bin
from verification import ver_test
import numpy as np

from utils.params_count import count_training_params, get_model_size
import math

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--epoch', default=20, help='epoch to train the network')
    parser.add_argument('--batch_size', default=128, help='batch size to train network')
    #parser.add_argument('--lr_steps', default=[40000, 60000, 80000], help='learning rate to train network')
    parser.add_argument('--lr_steps', default=[100000, 140000, 160000], help='learning rate to train network')
    parser.add_argument('--momentum', default=0.9, help='learning alg momentum')
    parser.add_argument('--weight_deacy', default=5e-4, help='learning alg momentum')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=78771, help='the output size')
    parser.add_argument('--num_imgs', default=4970252, help='the images size')
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='/data1/output_deltaM/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='/data1/output_deltaM/ckpt', help='the ckpt file save path')
    parser.add_argument('--log_file_path', default='/data1/output_deltaM/logs', help='the log file save path')
    parser.add_argument('--saver_maxkeep', default=100, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=50000, help='tf dataset api buffer size')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--summary_interval', default=300, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=5000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=1000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to show information')
    parser.add_argument('--num_gpus', default=4, help='the num of gpus')
    parser.add_argument('--tower_name', default='tower', help='tower name')
    parser.add_argument('--from_scratch', default=True, help='switch to train from scratch')
    parser.add_argument('--dropout_rate', default=0.4, help='dropout rate')
    args = parser.parse_args()
    return args


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_deltaM(num_imgs, Budget, firstTrain=True):
    tmp_list = []
    if firstTrain:
        with open('/data1/delta_M.txt', 'w') as f1:
            for i in range(num_imgs):
                f1.write(str(i)+','+str(Budget)+'\n')
        with open('/data1/delta_M.txt', 'r') as f2:
            for line in f2.readlines():
                tmp_list.append(line.split(',')[1].strip())
                tmp_list = np.array(list(map(float,tmp_list)))
    else:
        with open('/data1/delta_M.txt', 'r') as f2:
            for line in f2.readlines():
                tmp_list.append(line.split(',')[1].strip())
                tmp_list = np.array(list(map(float, tmp_list)))
    return tmp_list

def update_deltaM(deltaM):
    with open('/data1/delta_M.txt', 'w') as f3:
        for idx, delta_m in deltaM:
            f3.write(str(idx) + ',' + str(delta_m) + '\n')
    return deltaM



if __name__ == '__main__':
    # 1. define global parameters
    args = get_parser()
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    #inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)
    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    symbol = tf.placeholder(name='img_symbol', dtype=tf.int64)
    # deltaM_in_txt = tf.placeholder(name='img_deltaM', shape=[None, ], dtype=tf.float32)

    # splits input to different gpu
    images_s = tf.split(images, num_or_size_splits=args.num_gpus, axis=0, name='image_split')
    labels_s = tf.split(labels, num_or_size_splits=args.num_gpus, axis=0, name='label_split')

    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    theta = np.zeros(args.num_output)
    post_theta = np.zeros(args.num_output)
    delta_theta = np.zeros(args.num_output)
    gama = np.zeros(args.num_output)
    balance_ratio_lambda = 1
    step_size = 0.1
    minimum_improve_ratio = 0.5
    budget_per_sample = math.pi / 6
    minimum_budget_per_sample = 0
    maximum_budget_per_sample = math.pi / 3
    m = np.zeros(args.num_imgs)
    delta_m = get_deltaM(num_imgs=args.num_imgs, Budget=budget_per_sample, firstTrain=True)
    add_delta_m = 0
    # random flip left right
    tfrecords_f = tf.constant(["/data1/msceleb1m_0.tfrecords",
                               "/data1/msceleb1m_1.tfrecords",
                               "/data1/msceleb1m_2.tfrecords",
                               "/data1/msceleb1m_3.tfrecords",
                               "/data1/msceleb1m_4.tfrecords"])
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_preprocess_function)
    #dataset = dataset.map(_augment_function)
    dataset = dataset.shuffle(buffer_size=args.buffer_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(args.batch_size))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # 2.2 prepare validate datasets

    # 3. define network, loss, optimize method, learning rate schedule, summary writer, saver
    # 3.1 inference phase
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    # 3.2 define the learning rate schedule
    p = int(512.0/args.batch_size)
    lr_steps = [p*val for val in args.lr_steps]
    #print('learning rate steps: ', lr_steps)
    #lr_steps = [500000]
    #lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.02, 0.02, 0.02, 0.02],
                                     name='lr_schedule')
    #dp_rate = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.4, 0.6, 0.8], name='dp_schedule')
    # 3.3 define the optimize method
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum)

    # Calculate the gradients for each model tower.
    tower_grads = []
    tl.layers.set_name_reuse(True)
    loss_dict = {}
    drop_dict = {}
    loss_keys = []
    theta_dict = {}
    theta_keys = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(args.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (args.tower_name, i)) as scope:
            wd_loss_batch = 0
            inference_loss_batch = 0
            logit_list = []
            for j in range(int(args.batch_size/4)):
                wd_loss = 0
                #images_s dimension ---->4*32
                net = get_resnet(images_s[i][j], args.net_depth, type='ir', w_init=w_init_method, trainable=trainable, keep_rate= args.dropout_rate)
                m = m[symbol[j]] + delta_m[symbol[j]]     #symbol dimension---> 32
                logit ,theta = arcface_loss(embedding=net.outputs, labels=labels_s[i][j], w_init=w_init_method, out_num=args.num_output, m=m)
                logit_list.append(logit)
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()
                #Obtain theta_max
                theta_max = tf.nn.top_k(theta[j], k=1, sorted=False, name=None)[0][0]
                theta_dict[('theta_%s_%d_%d' % ('gpu', i, j))] = theta_max
                theta_keys.append(('theta_%s_%d_%d' % ('gpu', i, j)))
                # define the cross entropy
                inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels_s[i][j]))
                inference_loss_batch += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels_s[i][j]))
                # define weight deacy losses
                # wd_loss = 0
                for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
                    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
                for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
                    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
                for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
                    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
                for gamma in tl.layers.get_variables_with_name('gamma', True, True):
                    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
                #for beta in tl.layers.get_variables_with_name('beta', True, True):
                #    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(beta)
                for alphas in tl.layers.get_variables_with_name('alphas', True, True):
                    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
                #for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
                #    wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(bias)
                total_loss = wd_loss + inference_loss
                wd_loss_batch += wd_loss
            total_loss_batch = inference_loss_batch + wd_loss_batch

            loss_dict[('inference_loss_%s_%d' % ('gpu', i))] = inference_loss_batch
            loss_keys.append(('inference_loss_%s_%d' % ('gpu', i)))
            loss_dict[('wd_loss_%s_%d' % ('gpu', i))] = wd_loss_batch
            loss_keys.append(('wd_loss_%s_%d' % ('gpu', i)))
            loss_dict[('total_loss_%s_%d' % ('gpu', i))] = total_loss_batch
            loss_keys.append(('total_loss_%s_%d' % ('gpu', i)))

            grads = opt.compute_gradients(total_loss_batch)
            tower_grads.append(grads)
            drop_dict.update(net.all_drop)
            if i == 0:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                logit_list = tf.stack(tf.constant(logit_list), axis=0)
                pred = tf.nn.softmax(logit_list)
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels_s[i]), dtype=tf.float32))
                embedding_tensor_gpu0 = net.outputs

    grads = average_gradients(tower_grads)
    with tf.control_dependencies(update_ops):
        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # summary writer
    summary = tf.summary.FileWriter(args.summary_path, sess.graph)
    summaries = []
    # add grad histogram op
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    # add trainabel variable gradients
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    # add loss summary
    for keys, val in loss_dict.items():
        summaries.append(tf.summary.scalar(keys, val))
    #add theta summary
    # theta_shape = (tf.shape(theta)[0]).eval(session=tf.Session())
    for theta_keys, theta_val in theta_dict.items():
        summaries.append(tf.summary.scalar(theta_keys, theta_val))
    # add learning rate
    summaries.append(tf.summary.scalar('leraning_rate', lr))
    # add training acc
    summaries.append(tf.summary.scalar('training_acc', acc))
    summary_op = tf.summary.merge(summaries)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    # init all variables
    sess.run(tf.global_variables_initializer())

    _, model_size = get_model_size()
    print('model size: %.3fM' % model_size)
    #drop_dict_test = {keys: 1 for keys in drop_dict.keys()}
    # begin iteration
    count = 0
    if args.from_scratch:
        checkpoint_dir = ckpt = None
        global_step_init = tf.assign(global_step, 0)
        global_step_val = sess.run(global_step_init)
    else:
        checkpoint_dir = '/data1/output_deltaM/ckpt/'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    # saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        count = int(ckpt.model_checkpoint_path.rsplit('_', 1)[1].split('.')[0])
        global_step_init = tf.assign(global_step, count)
        global_step_val = sess.run(global_step_init)

    for i in range(args.epoch):
        sess.run(iterator.initializer)
        if i > 0:
            delta_m = get_deltaM(num_imgs=args.num_imgs, Budget=budget_per_sample, firstTrain=False)
        while True:
            try:
                # save ckpt files
                if global_step_val != count and global_step_val % args.ckpt_interval == 0:
                    filename = 'InsightFace_ResNet_{:d}_MGPU_'.format(args.net_depth) + 'iter_{:d}'.format(
                        global_step_val) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    # saver = tf.train.Saver()
                    saver.save(sess, filename)
                    print('%s saved' % (filename))
                images_train, labels_train, symbol_train = sess.run(next_element)
                feed_dict = {images: images_train, labels: labels_train, trainable: True}
                feed_dict.update(drop_dict)
                start = time.time()
                _, inference_loss_val_gpu_0, wd_loss_val_gpu_0, total_loss_gpu_0, inference_loss_val_gpu_1, \
                wd_loss_val_gpu_1, total_loss_gpu_1, inference_loss_val_gpu_2, wd_loss_val_gpu_2, total_loss_gpu_2, \
                inference_loss_val_gpu_3, wd_loss_val_gpu_3, total_loss_gpu_3, \
                global_step_val, acc_val, theta_all = sess.run([train_op,   loss_dict[loss_keys[0]],
                                                loss_dict[loss_keys[1]],
                                                loss_dict[loss_keys[2]],
                                                loss_dict[loss_keys[3]],
                                                loss_dict[loss_keys[4]],
                                                loss_dict[loss_keys[5]],
                                                loss_dict[loss_keys[6]],
                                                loss_dict[loss_keys[7]],
                                                loss_dict[loss_keys[8]],
                                                loss_dict[loss_keys[9]],
                                                loss_dict[loss_keys[10]],
                                                loss_dict[loss_keys[11]],global_step, acc, theta],
                                                feed_dict=feed_dict)
                # print('=======================theta_all====================',theta_all.shape)
                # print('***********************theta************************',theta_all[0])
                # print('-----------------------theta------------------------', (theta_all[0]).shape)

                labels_idMax_list = []
                for ele in labels_train:
                    id_max = np.argmax(ele, 0)
                    labels_idMax_list.append(id_max)
                for num, labels_maxIdx in enumerate(labels_idMax_list):
                    post_theta[labels_maxIdx] = np.arccos(tf.nn.top_k(abs(theta_all[num]), k=1, sorted=False, name=None)[0][0])
                    delta_theta[labels_maxIdx] = theta[labels_maxIdx] - post_theta[labels_maxIdx]
                    gama[labels_maxIdx] = delta_theta[labels_maxIdx] - delta_m[symbol[num]] * balance_ratio_lambda
                    delta_m[symbol[num]] = delta_m[symbol[num]] + gama[labels_maxIdx] * step_size
                    add_delta_m = np.mean(delta_m) - budget_per_sample
                    delta_m = delta_m - np.ones(args.num_imgs) * add_delta_m
                    for i in range(args.num_imgs):
                        if (delta_m[i] < minimum_budget_per_sample):
                            delta_m[i] = minimum_budget_per_sample
                        elif (delta_m[i] > maximum_budget_per_sample):
                            delta_m[i] = maximum_budget_per_sample
                    if (np.mean(delta_theta) < minimum_improve_ratio * budget_per_sample):
                        delta_m = np.ones(args.num_imgs) * budget_per_sample
                # update_deltaM(delta_m)

                end = time.time()
                pre_sec = args.batch_size/(end - start)
                # print training information
                if global_step_val > 0 and global_step_val % args.show_info_interval == 0:
                    print('epoch %d, total_step %d, total loss gpu 0 is %.2f , inference loss gpu 0 is %.2f, weight deacy '
                          'loss gpu 0 is %.2f, total loss gpu 1 is %.2f , inference loss gpu 1 is %.2f, weight deacy '
                          'loss gpu 1 is %.2f, total loss gpu 2 is %.2f , inference loss gpu 2 is %.2f, weight deacy '
                          'loss gpu 2 is %.2f, total loss gpu 3 is %.2f , inference loss gpu 3 is %.2f, weight deacy '
                          'loss gpu 3 is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                          (i, global_step_val, total_loss_gpu_0, inference_loss_val_gpu_0, wd_loss_val_gpu_0,
                           total_loss_gpu_1, inference_loss_val_gpu_1, wd_loss_val_gpu_1, total_loss_gpu_2,
                           inference_loss_val_gpu_2, wd_loss_val_gpu_2, total_loss_gpu_3, inference_loss_val_gpu_3, wd_loss_val_gpu_3,
                           acc_val, pre_sec))
                #count += 1

                # save summary
                if global_step_val > 0 and global_step_val % args.summary_interval == 0:
                    feed_dict = {images: images_train, labels: labels_train, trainable: True}
                    feed_dict.update(drop_dict)
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, global_step_val)
                """
                # save ckpt files
                if global_step_val > 0 and global_step_val % args.ckpt_interval == 0:
                    filename = 'InsightFace_iter_{:d}'.format(global_step_val) + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)
                """
                # # validate
                """
                if count > 0 and count % args.validate_interval == 0:
                    feed_dict_test ={trainable: False}
                    feed_dict_test.update(drop_dict_test)
                    results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                             embedding_tensor=embedding_tensor_gpu0, batch_size=args.batch_size//args.num_gpus, feed_dict=feed_dict_test,
                             input_placeholder=images_s[0])
                    if max(results) > 0.99:
                        print('best accuracy is %.5f' % max(results))
                        filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)
                """
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % i)
                break
        update_deltaM(delta_m)