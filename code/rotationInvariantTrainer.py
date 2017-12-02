import tensorflow as tf
import numpy as np
import createBasisTensor as cbt


#copied from https://www.tensorflow.org/get_started/mnist/pros
#To keep our code cleaner, let's also abstract those operations into functions.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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
initializationConstant = 0.01
p_1 = tf.Variable(tf.ones([filtRadius+1])) * initializationConstant
W_conv1 = tf.tensordot(P, bweights, [0,0])
print(W_conv1.get_shape())

b_conv1 = tf.Variable(initializationConstant)
# copied from https://www.tensorflow.org/get_started/mnist/pros
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



with tf.Session() as sess:
  pass

