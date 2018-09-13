#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np

IMAGE_W = 4
IMAGE_H = 4
IMAGE_D = 3
NUM_SAMPLES = 500


def make_dataset():
    images = np.arange(IMAGE_W * IMAGE_H * IMAGE_D, dtype=np.uint8)
    images = np.reshape(images, (IMAGE_H, IMAGE_W, IMAGE_D))
    images = images[np.newaxis, :, :, :]
    images = np.tile(images, (NUM_SAMPLES, 1, 1, 1))
    labels = np.arange(NUM_SAMPLES, dtype=np.uint64)
    assert len(images) == len(labels)
    print('num samples : {}'.format(len(images)))
    return (images, labels)


def convert_to_tfrecords(dataset, savename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    images, labels = dataset
    with tf.python_io.TFRecordWriter(savename) as writer:
        for i in range(NUM_SAMPLES):
            image_raw = images[i].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(IMAGE_H),
                        'width': _int64_feature(IMAGE_W),
                        'depth': _int64_feature(IMAGE_D),
                        'label': _int64_feature(labels[i]),
                        'image_raw': _bytes_feature(image_raw),
                        }))
            writer.write(example.SerializeToString())
        print('{0} construct finished {0}'.format('~' * 15))


def parse_tfrecords(filename):
    def _decode(example):
        features = tf.parse_single_example(
            example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64)
                })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, (IMAGE_H, IMAGE_W, IMAGE_D))
        label = tf.cast(features['label'], tf.int32)
        height = features['height']
        width = features['width']
        return (image, height, width, label)

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(_decode)
        dataset = dataset.batch(100)
        dataset = dataset.repeat(20)
        iterator = dataset.make_one_shot_iterator()
        batch_data = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                batch_train_data = sess.run(batch_data)
                print(batch_train_data[0].shape, batch_train_data[3].shape)
        except tf.errors.OutOfRangeError:
            print('{0} load finished {0}'.format('~' * 15))


if __name__ == "__main__":
    dataset = make_dataset()
    convert_to_tfrecords(dataset, 'mnist.tfrecords')
    parse_tfrecords('mnist.tfrecords')
