import tensorflow as tf
import numpy as np
import sys

sys.path.append('../')

from loadmnist import load

loadfnc = load('../mnist')

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# LeNet-5
# def set_layer(X):
W1 = tf.Variable(tf.random_normal([5, 5, 1, 6], stddev=0.01))

# input: 28*28 image, padding 2*2, output: 28*28*6
l1a = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
# relu
b1 = tf.Variable(tf.zeros([6]))
_l1a = tf.nn.relu(tf.nn.bias_add(l1a, b1))
# max pooling, out:14*14*6
l1b = tf.layers.max_pooling2d(_l1a, [2, 2], [2, 2])
# conv2d layer, out: 10*10*16
W2 = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=0.01))
l2a = tf.nn.conv2d(l1b, W2, strides=[1, 1, 1, 1], padding="VALID")
# relu
b2 = tf.Variable(tf.zeros([16]))
_l2a = tf.nn.relu(tf.nn.bias_add(l2a, b2))
# max pooling layer, out:5*5*16
l2b = tf.layers.max_pooling2d(_l2a, [2, 2], strides=[2, 2], padding="VALID")
# reshape to one line
l2c = tf.reshape(l2b, [-1, 5*5*16])
# full connection
W3 = tf.Variable(tf.random_normal([5*5*16, 120], stddev=0.01))
b3 = tf.Variable(tf.zeros((1, 120)))
l3 = tf.nn.relu(tf.matmul(l2c, W3) + b3)

W4 = tf.Variable(tf.random_normal([120, 84], stddev=0.01))
b4 = tf.Variable(tf.zeros((1, 84)))
l4 = tf.nn.relu(tf.matmul(l3, W4) + b4)

W5 = tf.Variable(tf.random_normal([84, 10], stddev=0.01))
b5 = tf.Variable(tf.zeros((1, 10)))
l5 = tf.matmul(l4, W5) + b5

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=l5))
    # return l5

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        batch_size = 64
        loss = None
        for n in range(60000//batch_size + 1):
            _X, _Y = loadfnc.get_next_batch_onehot(batch_size=batch_size)
            _X = _X.reshape(_X.shape[0], 28, 28, 1)
            _loss, _train = sess.run([cross_entropy, train_step], feed_dict={X:_X, Y:_Y})
            loss = _loss
        if i % 5 == 0:
            correct_prediction = tf.equal(tf.argmax(l5, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _X, _Y = loadfnc.get_next_batch_onehot(type="test")
            _X = _X.reshape(_X.shape[0], 28, 28, 1)
            acc = sess.run(accuracy, feed_dict={X:_X, Y:_Y})
            print('epoch: %s, loss: %s, acc: %s' % (i, loss, acc))

    correct_prediction = tf.equal(tf.argmax(l5, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    _X, _Y = loadfnc.get_next_batch_onehot(type="test")
    _X = _X.reshape(_X.shape[0], 28, 28, 1)
    acc = sess.run(accuracy, feed_dict={X:_X, Y:_Y})
    print('final acc: %s' % acc)