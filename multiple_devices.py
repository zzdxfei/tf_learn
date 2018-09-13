import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import tensorflow as tf

with tf.device('/device:GPU:0'):
    X = tf.get_variable('x', shape=(3000, 3000), dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    X = tf.matmul(X, X)
    X = tf.reduce_sum(X)

with tf.device('/device:GPU:1'):
    Y = tf.get_variable('y', shape=(3000, 3000), dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    Y = tf.matmul(Y, Y)
    Y = tf.reduce_sum(Y)


with tf.device('/device:CPU:0'):
    output = X + Y

init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(init_op)
    while True:
        sess.run(output)
