import tensorflow as tf
import numpy as np
import createBasisWeights as cbw


#copied from https://www.tensorflow.org/get_started/mnist/pros
#To keep our code cleaner, let's also abstract those operations into functions.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_7x7(x):
  return tf.nn.max_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')

# channelSizes is the number of channels in each of the hidden layers
# len(channelSizes) is the number of hidden layers (plus an extra fully-connected layer at the end)
def createRICNN(channelSizes):
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
  bweights = tf.constant(cbw.createBasisWeights(filtRadius), dtype = tf.float32) 
  
  # the parameterization of our rotationally symmetric basis vectors for convolution
  # we start with just a single filter and a single bias
  initializationConstant = 0.01
  # the first layer takes in one input and returns channelSizes[0]
  p_1 = tf.Variable(tf.ones([filtRadius+1,1,channelSizes[0]], dtype = tf.float32)) * initializationConstant
  W_conv1 = tf.tensordot(bweights, p_1, [[0],[0]])

  #print(W_conv1.get_shape())
  
  b_conv1 = tf.Variable(initializationConstant)
  # copied from https://www.tensorflow.org/get_started/mnist/pros
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  
  h_conv = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool = max_pool_2x2(h_conv)

  # add the rest of the layers after the first one in a for loop so we don't have to copy paste and are hopefully less likely to make typos
  numLayers = len(channelSizes) 
  for layerId in range(1,numLayers):
    # the layer takes in something of with channelSizes[layerId-1] number of channels and returns something with channelSizes[layerId] number
    # of channels
    p = tf.Variable(tf.ones([filtRadius+1,channelSizes[layerId-1],channelSizes[layerId]], dtype = tf.float32)) * initializationConstant
    W_conv = tf.tensordot(bweights, p, [[0],[0]])
    b_conv = tf.Variable(initializationConstant)
    h_conv = tf.nn.relu(conv2d(h_pool, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
  
  # add a fully connected layer
  fullyConnectedLayerWidth = 1024
  W_fc1 = tf.Variable(tf.truncated_normal([channelSizes[numLayers-1],fullyConnectedLayerWidth], stddev = 0.1))
  b_fc1 = tf.Variable(tf.constant(0.1,shape=[fullyConnectedLayerWidth]))
  
  h_pool_flat = tf.reshape(h_pool, [-1, channelSizes[numLayers-1]])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

  # add a dropout layer because oh boy are we ready to overfit this puppy
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # add a readout layer
  numOutputClasses = 10
  W_fc1 = tf.Variable(tf.truncated_normal([fullyConnectedLayerWidth, numOutputClasses], stddev = 0.1))
  b_fc1 = tf.Variable(tf.constant(0.1,shape=[numOutputClasses]))

  y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc1) + b_fc1)
  
  return y_conv


if __name__=="__main__":
  nn = createRICNN([1,1,1])
