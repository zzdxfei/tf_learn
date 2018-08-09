#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import os


BATCH_SIZE = 32


def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    print('+++++++++++++ MNIST DATASET INFO +++++++++++++')
    print('train x shape {}\n test x shape {}'.format(x_train.shape, x_test.shape))
    print('train y shape {}\n test y shape {}'.format(y_train.shape, y_test.shape))
    return (x_train, y_train), (x_test, y_test)


def get_model_bone(input_tensor):
    HIDDEN_UNITS = 512
    with tf.variable_scope('preprocess'):
        input_tensor = input_tensor / 256.0
        tf.summary.image('input', input_tensor)

    with tf.variable_scope('conv1'):
        w = tf.Variable(tf.truncated_normal(
            [5, 5, 1, 32], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.zeros((32,), dtype=tf.float32))
        conv1 = tf.nn.conv2d(input_tensor, w, [1, 1, 1, 1], 'SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b))

    with tf.variable_scope('pooling1'):
        pool1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    with tf.variable_scope('conv2'):
        w = tf.Variable(tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.zeros((64,), dtype=tf.float32))
        conv2 = tf.nn.conv2d(pool1, w, [1, 1, 1, 1], 'SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b))

        padding_shape = tf.constant([2, 2, 2, 2], shape=(2, 2), dtype=tf.int64)

        vis_w = tf.concat([tf.concat(
            [tf.pad(w[:, :, j, i], padding_shape) 
                for i in range(64)], axis=1) for j in range(32)], axis=0)
        vis_w_shape = vis_w.get_shape().as_list()
        vis_w = tf.reshape(vis_w, [1, vis_w_shape[0], vis_w_shape[1], 1])
        tf.summary.image('conv1_w', vis_w)

        """
        for i in range(64):
            tf.summary.image('conv2_' + str(i), conv2[:, :, :, i: i+1])
        """

    with tf.variable_scope('pooling2'):
        pool2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')

    with tf.variable_scope('flatten'):
        flatten = tf.layers.Flatten()(pool2)

    with tf.variable_scope('dense1'):
        flatten_shape = flatten.get_shape().as_list()
        w = tf.Variable(tf.truncated_normal(
            [flatten_shape[1], HIDDEN_UNITS], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.zeros((HIDDEN_UNITS,), dtype=tf.float32))
        dense1 = tf.nn.relu(tf.matmul(flatten, w) + b)
        tf.summary.histogram('dense1_w', w)
        tf.summary.histogram('dense1_b', b)

    with tf.variable_scope('dense2'):
        w = tf.Variable(tf.truncated_normal(
            [HIDDEN_UNITS, HIDDEN_UNITS / 2], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.zeros((HIDDEN_UNITS / 2,), dtype=tf.float32))
        dense2 = tf.nn.relu(tf.matmul(dense1, w) + b)
        tf.summary.histogram('dense2_w', w)
        tf.summary.histogram('dense2_b', b)

    with tf.variable_scope('output'):
        w = tf.Variable(tf.truncated_normal(
            [HIDDEN_UNITS / 2, 10], stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.zeros((10,), dtype=tf.float32))
        output = tf.matmul(dense2, w) + b

    return output


def train_net():
    # load data
    (x_train, y_train), (x_test, y_test) = get_mnist_dataset()

    net_input = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 28, 28, 1))
    net_truth = tf.placeholder(dtype=tf.int64, shape=(BATCH_SIZE,))

    net_output = get_model_bone(net_input)

    # 训练模型
    with tf.variable_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=net_output, labels=net_truth)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 预测模型
    with tf.variable_scope('predict'):
        predict_model = tf.nn.softmax(net_output)

    # accuracy
    with tf.variable_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(predict_model, axis=-1), net_truth), tf.float32))

    init_op = tf.global_variables_initializer()

    # summary
    summaryer = tf.summary.merge_all()

    NUM_EPOCHS = 10
    
    saver = tf.train.Saver()
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs', sess.graph)
        sess.run(init_op)

        global_step = 0
        for e in range(NUM_EPOCHS):
            print('\n\n{0} epoch -> {1} {0}'.format('=' * 10, e))
            running_loss = 0.0
            for i in range(len(x_train) / BATCH_SIZE):
                batch_x_train = x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_y_train = y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_summary, _, train_loss = sess.run(
                    [summaryer, optimizer, loss], feed_dict={
                    net_input: batch_x_train, net_truth: batch_y_train})
                if i == 0:
                    running_loss = train_loss
                else:
                    running_loss = running_loss * .9 + train_loss * .1

                if (i + 1) % 100 == 0:
                    print('step -> {} : running loss -> {}'.format(
                        i, running_loss))
                writer.add_summary(batch_summary, global_step=global_step)
                global_step += 1

            # 测试
            running_accuracy = 0.0
            for i in range(len(x_test) / BATCH_SIZE):
                batch_x_test = x_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_y_test = y_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_accuracy = sess.run(accuracy, feed_dict={
                    net_input: batch_x_test, net_truth: batch_y_test})
                running_accuracy += batch_accuracy * BATCH_SIZE
            running_accuracy /= (len(x_test) / BATCH_SIZE * BATCH_SIZE)
            print(">>>>>> test accuracy -> {} <<<<<<<".format(running_accuracy))

            # save checkpoint
            saver.save(sess, os.path.join(save_dir, 'weights.ckpt'), global_step=e)
            print(">>>>>> MODEL SAVED <<<<<<<")


if __name__ == "__main__":
    train_net()
