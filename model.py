import tensorflow as tf
import numpy as np
from OCR import generate


class Network:
    def __init__(self, batch_size):
        self.sess = tf.Session()
        self.parameters = []
        self.images = tf.placeholder(tf.float32, shape=[batch_size, 28, 28])
        images = tf.reshape(self.images, [-1, 28, 28, 1])


        # conv1_1
        filters = 32
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 1, filters], dtype=tf.float32,
                                                     stddev=1e-1))
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, filters, filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, filters, filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, filters, filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[filters], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # fc 1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 400],
                                                   dtype=tf.float32,
                                                   stddev=1e-1))
            fc1b = tf.Variable(tf.constant(0.0, shape=[400], dtype=tf.float32))
            pool2_flat = tf.reshape(self.pool2, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc 2 weights
        fc2w = tf.Variable(tf.truncated_normal([400, 64],
                                               dtype=tf.float32,
                                               stddev=1e-1))
        fc2b = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32))
        self.parameters += [fc2w, fc2b]

        # LSTM
        lstm_size = 400
        with tf.variable_scope("LSTM") as scope:
            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
            state1 = tf.zeros([batch_size, self.lstm.state_size[0]])
            state2 = tf.zeros([batch_size, self.lstm.state_size[1]])
        state = (state1, state2)
        max_chars_per_image = 3
        self.loss = 0
        self.labels = tf.placeholder(tf.int64, shape=(batch_size, 3))
        self.preds = []
        self.accuracies = []
        for iterN in range(max_chars_per_image):
            with tf.variable_scope("LSTM") as scope:
                if iterN > 0:
                    scope.reuse_variables()
                output, state = self.lstm(tf.reshape(self.fc1, [batch_size, -1]), state, scope=scope)
                self.fc2 = tf.nn.bias_add(tf.matmul(output, fc2w), fc2b)
                self.loss += tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2, self.labels[:, iterN])
                self.preds.append(self.fc2)
                correct_prediction = tf.equal(tf.argmax(self.fc2, dimension=1), self.labels[:, iterN])
                self.accuracies.append(tf.reduce_mean(tf.cast(correct_prediction, "float")))
        self.loss = tf.reduce_mean(self.loss) / float(max_chars_per_image)
        optimizer = tf.train.AdamOptimizer()
        self.optimize = optimizer.minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(tf.all_variables())
        self.batch_size = batch_size

    def load_weights(self, fname):
        self.saver.restore(self.sess, fname)

    def save_weights(self, fname):
        self.saver.save(self.sess, fname)

    def train(self):
        batch_images, batch_labels = generate(self.batch_size)
        feed_dict = {self.images: batch_images, self.labels: batch_labels}
        _, loss = self.sess.run([self.optimize, self.loss], feed_dict=feed_dict)
        print "Training loss is ", loss
        return loss

    def train_print_accuracy(self):
        batch_images, batch_labels = generate(self.batch_size)
        feed_dict = {self.images: batch_images, self.labels: batch_labels}
        _, loss, ac1, ac2, ac3 = self.sess.run([self.optimize, self.loss] + self.accuracies, feed_dict=feed_dict)
        print "Training loss is ", loss, "accuracies", ac1, ac2, ac3
        return loss

    def test(self, batch_images):
        feed_dict = {self.images: batch_images}
        results = self.sess.run(self.preds, feed_dict=feed_dict)
        return results
