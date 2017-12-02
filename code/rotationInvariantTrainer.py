import tensorflow as tf
import numpy as np
import createBasisTensor as cbt

# a matrix of the input MNIST images
x = tf.placeholder(tf.float32, shape=[None, 784])
# a matrix of the output labels (one-hot)
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# the h-value determining the "radius" of our convolutional filters
# the convolution filter will be a square of side length 2h+1
filtRadius = 2

# the basis weights of our rotationally symmetric filters
# this numpy matrix is of shape h+1, 2h+1, 2h+1 and gives
# filters that are rings of various radii around the origin (linearly interpolated)
bweights = tf.constant(cbt.createBasisTensor(filtRadius))


# the parameterization of our rotationally symmetric basis vectors for convolution
# we start with just a single filter and a single bias
p = tf.Variable(tf.ones([filtRadius+1]))
W = tf.tensordot(P, bweights, [0,0])

b = tf.Variable(1)
# copied from https://www.tensorflow.org/get_started/mnist/pros
x_image = tf.reshape(x, [-1, 28, 28, 1])





with tf.Session() as sess:
  pass

