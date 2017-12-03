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
nn = rin.RICNN([1,1,1,1,1,1], fullyConnectedLayerWidth, numOutputClasses)
x = tf.placeholder(tf.float32, shape=[None, 784])
keep_prob = tf.placeholder(tf.float32)
preds = nn.setupNodes(x, keep_prob)

with tf.Session() as sess:
  rawimg = np.reshape(mnist.train.images[0:1,:],(28,28))
  rawimg2 = np.reshape(mnist.train.images[1:2,:],(28,28))
  sess.run(tf.global_variables_initializer())
  output = sess.run(preds, feed_dict={x: np.reshape(rawimg, (1,784)), keep_prob: 1})
  print(output)
  output = sess.run(preds, feed_dict={x: np.reshape(rawimg2, (1,784)), keep_prob: 1})
  print(output)
  cv2.imshow('image',rawimg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  cv2.imshow('image',rawimg2)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  img = np.transpose(rawimg)
  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  img = np.reshape(img, (1,784))
  output = sess.run(preds, feed_dict={x: img, keep_prob: 1})
  print(output)
  rows,cols = rawimg.shape
  M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
  img = cv2.warpAffine(rawimg,M,(cols,rows))
  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  img = np.reshape(img, (1,784))
  output = sess.run(preds, feed_dict={x: img, keep_prob: 1})
  print(output)
  M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
  img = cv2.warpAffine(rawimg,M,(cols,rows))
  cv2.imshow('image',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  img = np.reshape(img, (1,784))
  output = sess.run(preds, feed_dict={x: img, keep_prob: 1})
  print(output)
  

