# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
from rotationInvariantNetwork import RICNN

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

x = tf.placeholder(
    tf.float32, shape=[None, image_size * image_size])
keep_prob = tf.placeholder(tf.float32)

numOutputClasses = 10
fullyConnectedLayerWidth = 1024 
nn = RICNN([10, 50, 100, 200, 500], fullyConnectedLayerWidth, numOutputClasses, 5)

logits = nn.setupNodes(x, keep_prob)

saver = tf.train.Saver()

cols=image_size
rows=image_size
M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)

np.set_printoptions(precision=2)

with tf.Session() as sess:
  saver.restore(sess, "/tmp/model.ckpt")

  numImgsToCheck = 10 
  for i in range(numImgsToCheck):
    rawimg = np.reshape(mnist.validation.images[i:(i+1),:],(28,28))
    output = sess.run(logits, feed_dict={x: np.reshape(rawimg, (1,784)), keep_prob: 1})
    print("Image%d returns %s " % (i,output))
    cv2.imshow('Image%d'%i,rawimg)
    rotimg = cv2.warpAffine(rawimg,M,(cols,rows))
    cv2.imshow('Image%d rotated' % i, rotimg)
    output = sess.run(logits, feed_dict={x: np.reshape(rotimg, (1,784)), keep_prob: 1})
    print("Image%d rotated returns %s " % (i,output))

  
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  """ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch.images, Y: batch.labels, keep_prob: 1.0})
  plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
  plt.show()
  plt.imshow(d[0], cmap='gray')
  plt.show()
  print(i, ls, np.mean(i_ls))"""
