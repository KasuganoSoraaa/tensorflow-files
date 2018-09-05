# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
TRAIN_STEP = 20000
INPUT_NODE = 784
FILTER_SIZE = 5
CONV1_SIZE = 28
FILTER1_DEEP = 32
CONV2_SIZE = 14
LAST_POOL_SIZE = 7
FILTER2_DEEP = 64
FC1_NODE = 512
FC2_NODE = 10


def initial_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def initial_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(layers, filters):
    return tf.nn.conv2d(layers, filters, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(layers):
    return tf.nn.max_pool(layers, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train(mnist):
    x_input = tf.placeholder(tf.float32, [None, INPUT_NODE])
    x = tf.reshape(x_input, [-1, CONV1_SIZE, CONV1_SIZE, 1])

    weight1 = initial_weight([FILTER_SIZE, FILTER_SIZE, 1, FILTER1_DEEP])
    bias1 = initial_bias([FILTER1_DEEP])
    c1 = tf.nn.relu(conv2d(x, weight1) + bias1)
    p1 = max_pool(c1)

    weight2 = initial_weight([FILTER_SIZE, FILTER_SIZE, FILTER1_DEEP, FILTER2_DEEP])
    bias2 = initial_bias([FILTER2_DEEP])
    c2 = tf.nn.relu(conv2d(p1, weight2) + bias2)
    p2 = max_pool(c2)

    weight_fc1 = initial_weight([LAST_POOL_SIZE*LAST_POOL_SIZE*FILTER2_DEEP, FC1_NODE])
    bias_fc1 = initial_bias([FC1_NODE])
    p2_flat = tf.reshape(p2, [-1, LAST_POOL_SIZE*LAST_POOL_SIZE*FILTER2_DEEP])
    fc1 = tf.nn.relu(tf.matmul(p2_flat, weight_fc1) + bias_fc1)

    dropout_prob = tf.placeholder(tf.float32)
    fc1_drop = tf.nn.dropout(fc1, dropout_prob)

    weight_fc2 = initial_weight([FC1_NODE, FC2_NODE])
    bias_fc2 = initial_bias([FC2_NODE])
    y = tf.nn.softmax(tf.matmul(fc1_drop, weight_fc2) + bias_fc2)

    y_ = tf.placeholder(tf.float32, [None, FC2_NODE])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in range(TRAIN_STEP):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={x_input: batch[0],
                                                      y_: batch[1],
                                                      dropout_prob: 1.0})
            print("after %d step(s) , train_accuracy is %g" % (i, train_accuracy))
        train_step.run(session=sess, feed_dict={x_input: batch[0],
                                                y_: batch[1],
                                                dropout_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(session=sess,
                                             feed_dict={x_input: mnist.test.images,
                                                        y_: mnist.test.labels,
                                                        dropout_prob: 1.0}))


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
