import tensorflow as tf
import numpy as np
import createBasisWeights as cbw
import math


#copied from https://www.tensorflow.org/get_started/mnist/pros
#To keep our code cleaner, let's also abstract those operations into functions.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

class RICNN2:
  def __init__(self, channelSizes, fullyConnectedLayerWidth, numOutputClasses):
    # the h-value determining the "radius" of our convolutional filters
    # the convolution filter will be a square of side length 2h+1
    filtDiameter = 28
    filtRadius = math.ceil(filtDiameter/2)
    
    # the basis weights of our rotationally symmetric filters
    # this numpy matrix is of shape h+1, 2h+1, 2h+1 and gives
    # filters that are rings of various radii around the origin (linearly interpolated)
    # self.basisFilters = tf.constant(cbw.createBasisWeights(filtRadius), dtype = tf.float32) 
    self.basisFilters = tf.constant(cbw.createBasisWeightsDiameter(filtDiameter), dtype = tf.float32)
    # the parameterization of our rotationally symmetric basis vectors for convolution
    # we start with just a single filter and a single bias
    initializationConstant = 0.05

    # channelSizes is a vector where each index shows the number of channels in each layer of the convolutional hidden layers
    self.channelSizes = channelSizes
    print(self.channelSizes)
    self.channelSizes.append(numOutputClasses)
    print(self.channelSizes)
    self.numLayers = len(channelSizes)
    # add the rest of the layers after the first one in a for loop so we don't have to copy paste and are hopefully less likely to make typos
    # since we allow a variable number of weights, we construct a dictionary layersWeights
    # the projection of our weights onto the basis weight filters (one list of parameters in layersWeights for each number of layers)
    self.layersWeights = {}
    # likewise for the bias terms of each layer
    self.layersBias = {}
    # the first layer is special for our indexing because it takes in one input and returns channelSizes[0] (we don't have channelSizes[-1]
    # self.layersWeights[0] = tf.Variable(tf.truncated_normal([filtRadius+1,1,channelSizes[0]], dtype = tf.float32, stddev = initializationConstant))
    self.layersWeights[0] = tf.Variable(tf.truncated_normal([filtRadius,1,channelSizes[0]], dtype = tf.float32, stddev = initializationConstant))
    self.layersBias[0] = tf.Variable(tf.constant(initializationConstant, shape=[channelSizes[0]]))
    for layerId in range(1,self.numLayers):
      # the layer takes in something of with channelSizes[layerId-1] number of channels and returns something with channelSizes[layerId] number
      # of channels
      self.layersWeights[layerId] = tf.Variable(
        # tf.truncated_normal([filtRadius+1,channelSizes[layerId-1],channelSizes[layerId]], dtype = tf.float32, stddev = initializationConstant))
        tf.truncated_normal([channelSizes[layerId-1],channelSizes[layerId]], dtype = tf.float32, stddev = initializationConstant))
      self.layersBias[layerId] = tf.Variable(tf.constant(initializationConstant, shape=[channelSizes[layerId]]))

  # Set up our network by plugging in the consumer's placeholder nodes to our network, defining the nodes
  # in the network based on the weight values generated in __init__, and then returning the output node.
  # Note that the output node does not (yet) have the softmax applied to it. 
  # we expect the consumer to pass in the placeholder objects of the shape below (though it would be equivalent for us to return them
  # alongside y_conv. It really doesn't matter because these are just the initialization of nodes (we aren't actually
  # running the calculation here))
  # What is important, though, is that the shapes are as expected
  # x = tf.placeholder(tf.float32, shape=[None, 784])
  # keep_prob = tf.placeholder(tf.float32)
  # returns the node for the relu output (doesn't include the softmax node)
  def setupNodes(self, x, keep_prob):
    # copied from https://www.tensorflow.org/get_started/mnist/pros
    # to get nicely into the for loop, we just set h_pool to x_image (as if it were coming from a previous layer)
    h_pool = tf.reshape(x, [-1, 28, 28, 1])
  
    W_conv = tf.tensordot(self.basisFilters, self.layersWeights[0], [[0],[0]])
    b_conv = self.layersBias[0]
    h_conv = tf.nn.relu(conv2d(h_pool, W_conv) + b_conv)
    print(self.channelSizes)
    val = tf.reshape(h_conv, [-1, self.channelSizes[0]])

    for layerId in range(1,self.numLayers):
      val_drop = tf.nn.dropout(val, keep_prob)
      val = tf.nn.relu(tf.matmul(val_drop, self.layersWeights[layerId]) + self.layersBias[layerId])

    return val

if __name__=="__main__":
  numOutputClasses = 10
  fullyConnectedLayerWidth = 1024
  nn = RICNN([1,1,1], fullyConnectedLayerWidth, numOutputClasses)
  x = tf.placeholder(tf.float32, shape=[None, 784])
  keep_prob = tf.placeholder(tf.float32)
  nn.setupNodes(x, keep_prob)
