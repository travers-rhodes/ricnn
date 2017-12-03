import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import rotationInvariantNetwork as rin

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape)

numOutputClasses = 10
fullyConnectedLayerWidth = 1024
nn = rin.RICNN([1,1,1,1,1], fullyConnectedLayerWidth, numOutputClasses)
x = tf.placeholder(tf.float32, shape=[None, 784])
keep_prob = tf.placeholder(tf.float32)
preds = nn.setupNodes(x, keep_prob)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  output = sess.run(preds, feed_dict={x: mnist.train.images[0:1,:], keep_prob: 1})
  print(output)
  

