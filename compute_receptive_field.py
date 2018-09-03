import tensorflow as tf


input_tensor = tf.constant(0.0, shape=(1, 832, 416, 3), dtype=tf.float32)


def res_blob(input_tensor, number, channel):
    assert(number >= 1)
    output = tf.layers.Conv2D(
        channel, (3, 3), strides=(2, 2), padding='same')(input_tensor)
    for i in range(0, number):
        conv1 = tf.layers.Conv2D(
            channel / 2, (1, 1), strides=(1, 1), padding='same')(output)
        conv2 = tf.layers.Conv2D(
            channel, (3, 3), strides=(1, 1), padding='same')(conv1)
        output = tf.add(output, conv2)
    return output


def yolo_conv_head(input_tensor, channel, out_channel):
    output = input_tensor
    for _ in range(3):
        output = tf.layers.Conv2D(
            channel, (1, 1), strides=(1, 1), padding='same')(output)
        output = tf.layers.Conv2D(
            channel * 2, (3, 3), strides=(1, 1), padding='same')(output)
    output = tf.layers.Conv2D(
        out_channel, (1, 1), strides=(1, 1), padding='same')(output)
    return output


def get_darknet_model(input_tensor):
    with tf.variable_scope('base_conv'):
        base_conv = tf.layers.Conv2D(
            32, (3, 3), strides=(1, 1), padding='same')(input_tensor)
    with tf.variable_scope('res_blob_1'):
        res_blob_1 = res_blob(base_conv, 1, 64)
    with tf.variable_scope('res_blob_2'):
        res_blob_2 = res_blob(res_blob_1, 2, 128)
    with tf.variable_scope('res_blob_3'):
        res_blob_3 = res_blob(res_blob_2, 8, 256)
    with tf.variable_scope('res_blob_4'):
        res_blob_4 = res_blob(res_blob_3, 8, 512)
    with tf.variable_scope('res_blob_5'):
        res_blob_5 = res_blob(res_blob_4, 4, 1024)
    with tf.variable_scope('yolo_conv_head_1'):
        yolo_conv_head_1 = yolo_conv_head(res_blob_5, 512, 255)
    return res_blob_3


darknet_model = get_darknet_model(input_tensor)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(darknet_model).shape
    # writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)
    result = tf.contrib.receptive_field.compute_receptive_field_from_graph_def(
        sess.graph,
        input_tensor,
        darknet_model,
        input_resolution=[832, 416]
        )
    for item in result:
        print item
    # writer.close()
