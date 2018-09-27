#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np


def random_rotate(image, angle):
    max_angle = angle / 180.0 * 3.14
    random_number = tf.random_uniform([], dtype=np.float32) - 0.5
    random_number = random_number * 2 * max_angle
    image = tf.contrib.image.rotate(image, random_number)
    return image


def augment_image(image):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = random_rotate(image, 10)
    image = tf.image.random_hue(image, 0.1)
    # image = tf.image.random_jpeg_quality(image, 70, 99)
    wh = tf.random_uniform([2], dtype=tf.float32) * (250 - 224)
    wh = tf.cast(wh, tf.int32)
    image_resized = tf.image.crop_to_bounding_box(image, wh[0], wh[1], 224, 224)
    return image_resized


img = cv2.imread('./images/4.jpg')
img = np.stack((img, img))

dataset = tf.data.Dataset.from_tensor_slices(img)
dataset = dataset.map(augment_image)
dataset = dataset.repeat()
iter = dataset.make_one_shot_iterator()
element = iter.get_next()

with tf.Session() as sess:
    while True:
        current = sess.run(element)
        # current = current[:, :, ::-1]
        cv2.imshow('img', current)
        cv2.waitKey(500)
