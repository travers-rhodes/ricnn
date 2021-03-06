# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from rotationInvariantNetwork2 import RICNN2

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 32
image_size = 28
num_labels = 10

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# I don't think we need this, unless we're calling this code from some other python code
# that might have defined tf nodes? I guess this is just here to be safe? Because I think
# all this does is clear out previously constructed tf nodes, and I don't see any in the very few lines
# defined above this call
tf.reset_default_graph()

tf_train_dataset = tf.placeholder(
    tf.float32, shape=[None, image_size * image_size])
tf_train_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
keep_prob = tf.placeholder(tf.float32)
valid_dataset = mnist.validation.images
valid_labels = mnist.validation.labels
tf_test_dataset = tf.constant(mnist.test.images)
test_labels = mnist.test.labels

numOutputClasses = 10
fullyConnectedLayerWidth = 1024
nn = RICNN2([64, 32, 16, 8, 4], fullyConnectedLayerWidth, numOutputClasses)

logits = nn.setupNodes(tf_train_dataset, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(tf_train_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.00005).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init_op)
  
  for i in range(300000):
      # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
      batch_in, batch_out = mnist.train.next_batch(batch_size=batch_size)
      sess.run(optimizer, feed_dict = {tf_train_dataset: batch_in, tf_train_labels: batch_out, keep_prob: 0.8})
          
      if not i % 1000:
          ls = sess.run(loss, feed_dict = {tf_train_dataset: batch_in, tf_train_labels: batch_out, keep_prob: 1.0})
          acc = sess.run(accuracy, feed_dict={tf_train_dataset: valid_dataset, tf_train_labels: valid_labels, keep_prob: 1.0})
          print(i, ls, acc)
          save_path = saver.save(sess, "/tmp/model.ckpt")
          print("Model saved in file: %s" % save_path)
          """ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch.images, Y: batch.labels, keep_prob: 1.0})
          plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
          plt.show()
          plt.imshow(d[0], cmap='gray')
          plt.show()
          print(i, ls, np.mean(i_ls))"""
