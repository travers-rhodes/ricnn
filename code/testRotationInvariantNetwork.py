import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2

import rotationInvariantNetwork as rin

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape)
print(type(mnist.train.images))

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
  img = np.reshape(mnist.train.images[0:1,:],(28,28))
  img = np.transpose(img)
  img = np.reshape(img, (1,784))
  output = sess.run(preds, feed_dict={x: img, keep_prob: 1})
  print(output)
  rows,cols = img.shape
  M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
  img = cv2.warpAffine(img,M,(cols,rows))
  img = np.reshape(img, (1,784))
  output = sess.run(preds, feed_dict={x: img, keep_prob: 1})
  print(output)
  

