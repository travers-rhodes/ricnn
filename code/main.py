# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from rotationInvariantNetwork import RICNN

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 32
image_size = 28
num_labels = 10

mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size * image_size))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
keep_prob = tf.placeholder(tf.float32)
tf_valid_dataset = tf.constant(mnist.validation.images)
valid_labels = mnist.validation.labels
tf_test_dataset = tf.constant(mnist.test.images)
test_labels = mnist.test.labels

numOutputClasses = 10
fullyConnectedLayerWidth = 1024
nn = RICNN([4, 8, 16], fullyConnectedLayerWidth, numOutputClasses)

logits = nn.setupNodes(tf_train_dataset, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(30000):
    # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    batch_in, batch_out = mnist.train.next_batch(batch_size=batch_size)
    sess.run(optimizer, feed_dict = {X_in: batch_in, Y: batch_out, keep_prob: 0.8})
        
    if not i % 1000:
        ls = sess.run([loss], feed_dict = {tf_train_dataset: batch.images, tf_train_labels: batch.labels, keep_prob: 1.0})
        print(i, ls)
        """ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch.images, Y: batch.labels, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls))"""