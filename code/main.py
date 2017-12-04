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
image_size = 64
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

#pad our valid dataset with zeros
valid_dataset = ih.padImages(mnist.validation.images, raw_image_size, image_size)
valid_labels = mnist.validation.labels
test_labels = mnist.test.labels

numOutputClasses = 10
fullyConnectedLayerWidth = 1024 
nn = RICNN([10, 100], fullyConnectedLayerWidth, numOutputClasses, 6)

logits = nn.setupNodes(tf_train_dataset, keep_prob, image_size, iterativelyMaxPool=False)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(tf_train_labels,1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

cols=image_size
rows=image_size
rawimgs = np.reshape(valid_dataset,(-1,image_size,image_size))
valid_rotimg = np.zeros(rawimgs.shape)
for ind, rawimg in enumerate(rawimgs):
  ang = np.random.randint(1, 360)
  M = cv2.getRotationMatrix2D((cols/2 - 0.5,rows/2 - 0.5),ang,1)
  valid_rotimg[ind] = cv2.warpAffine(rawimg,M,(rows,cols))
valid_rotimg = np.reshape(valid_rotimg, (-1,image_size**2))

with tf.Session() as sess:
  sess.run(init_op)
  
  for i in range(300000):
      # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
      batch_in, batch_out = mnist.train.next_batch(batch_size=batch_size)
      #pad our training dataset with zeros
      batch_in = ih.padImages(batch_in, raw_image_size, image_size)
      sess.run(optimizer, feed_dict = {tf_train_dataset: batch_in, tf_train_labels: batch_out, keep_prob: 0.8})
          
      if not i % 1000:
          print("checking accuracy...")
          numValid = len(valid_labels)
          v_batch_size = 1000
          accsum = 0.0
          rotaccsum = 0.0
          confus = np.zeros((numOutputClasses,numOutputClasses))
          for batchNum in range(numValid/v_batch_size):
            valid_batch = range(batchNum * v_batch_size, min((batchNum + 1) * v_batch_size, numValid))
            accsum += sess.run(accuracy, feed_dict={tf_train_dataset: valid_dataset[valid_batch,:], tf_train_labels: valid_labels[valid_batch], keep_prob: 1.0})
            rotaccsum += sess.run(accuracy, feed_dict={tf_train_dataset: valid_rotimg[valid_batch,:], tf_train_labels: valid_labels[valid_batch], keep_prob: 1.0})
            
            preds = sess.run(tf.argmax(logits,1), feed_dict={tf_train_dataset: valid_rotimg[valid_batch,:], tf_train_labels: valid_labels[valid_batch], keep_prob: 1.0})
            act = np.argmax(valid_labels[valid_batch],1)
            #print(preds)
            #print(act)
            #print(preds.shape)
            #print(act.shape)
            for j in range(len(preds)):
              confus[preds[j], act[j]] = confus[preds[j], act[j]] + 1

          accbat = accsum / numValid
          rotaccbat = rotaccsum / numValid
          print(i, accbat, rotaccbat)
          print(confus)
          save_path = saver.save(sess, "/tmp/model.ckpt")
          print("Model saved in file: %s" % save_path)
          """ls, d, i_ls, mu, sigm = sess.run([loss, dec, img_loss, mn, sd], feed_dict = {X_in: batch.images, Y: batch.labels, keep_prob: 1.0})
          plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
          plt.show()
          plt.imshow(d[0], cmap='gray')
          plt.show()
          print(i, ls, np.mean(i_ls))"""
