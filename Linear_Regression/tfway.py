import tensorflow as tf
import sys

sys.path.append('../')

from loadmnist import load

loadfnc = load('../mnist')

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_normal([784, 15]))
W2 = tf.Variable(tf.zeros([15, 10]))

b1 = tf.Variable(tf.random_normal([1, 15]))
b2 = tf.Variable(tf.zeros([1, 10]))

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))

y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

y2 = tf.matmul(y1, W2) + b2
# y2 = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y2, labels = y_))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(100):
        batch_size = 10240
        loss = None

        for n in range(60000//batch_size + 1):
            _X, _Y = loadfnc.get_next_batch_onehot(batch_size=batch_size)
            _t, _loss= sess.run([train_step, cross_entropy], feed_dict={x: _X, y_: _Y})
            loss = _loss
        if i % 5 == 0:
            correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _X, _Y = loadfnc.get_next_batch_onehot(type="test")
            acc = sess.run(accuracy, feed_dict={x: _X, y_: _Y})
            print('epoch: %s, loss: %s, acc: %s' % (i, loss, acc))

