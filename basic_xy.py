import tensorflow as tf


def get_net_work():
    x = tf.Variable(initial_value=1.0, name='x', dtype=tf.float32)
    y = tf.multiply(x, x) + tf.multiply(-10.0, x) + 25.0
    return x, y


def main():
    x, y = get_net_work()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(y)
    init_op = tf.global_variables_initializer()
    tf.summary.scalar('x', x)
    tf.summary.scalar('y', y)
    summaryer = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)
        sess.run(init_op)

        for i in range(100):
            _, summary_output = sess.run([optimizer, summaryer])
            writer.add_summary(summary_output, i)
            result = sess.run([x, y])
            print i, ' -> ', result
        writer.close()

if __name__ == "__main__":
    main()
