# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2
from rotationInvariantNetwork import RICNN

import imageHelper as ih
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 32
raw_image_size = 28
image_size = 48 
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
nn = RICNN([100,100], fullyConnectedLayerWidth, numOutputClasses, 5)

logits = nn.setupNodes(x, keep_prob, image_size, iterativelyMaxPool=False)

saver = tf.train.Saver()

cols=image_size
rows=image_size
angle = 45
M = cv2.getRotationMatrix2D((cols/2 - 0.5,rows/2 - 0.5),angle,1)
Minv = cv2.getRotationMatrix2D((cols/2 - 0.5,rows/2 - 0.5),-angle,1)

np.set_printoptions(precision=2)

images = ih.padImages(mnist.validation.images, raw_image_size, image_size)

def transpImg(image, side_len):
  rawimg = np.reshape(image,(side_len,side_len))
  rawimg = np.transpose(rawimg, (1,0))
  

with tf.Session() as sess:
  saver.restore(sess, "/tmp/model.ckpt")

  numImgsToCheck = 10 
  for i in range(numImgsToCheck):
    rawimg = np.reshape(images[i:(i+1),:],(image_size,image_size))
    rotimg = cv2.warpAffine(rawimg,M,(rows,cols))
    #rotimg = np.rot90(rawimg)
    output1 = sess.run(logits, feed_dict={x: np.reshape(rawimg, (1,image_size*image_size)), keep_prob: 1})
    output2 = sess.run(logits, feed_dict={x: np.reshape(rotimg, (1,image_size*image_size)), keep_prob: 1})
    sizes = output1[1].shape
    filt11 = output1[1][0,:,:,1]
    filt21 = output2[1][0,:,:,1]

    viewfilt11 = np.reshape(filt11, (sizes[1], sizes[1]))
    viewfilt21 = np.reshape(filt21, (sizes[1], sizes[1]))
    viewfilt21 = cv2.warpAffine(viewfilt21,Minv,(sizes[1],sizes[1]))
    #viewfilt21 = np.rot90(viewfilt21, k=-1)
    

    cv2.imshow('Image%d'%i,viewfilt11)
    cv2.imshow('Image%d rotated and back' % i, viewfilt21)
    print(np.mean(np.mean((abs(viewfilt11 - viewfilt21)))))
    print(np.mean(np.mean(abs(viewfilt11))))

  
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  """ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch.images, Y: batch.labels, keep_prob: 1.0})
  plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
  plt.show()
  plt.imshow(d[0], cmap='gray')
  plt.show()
  print(i, ls, np.mean(i_ls))"""
