import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# pad our images with zeros in both dimensions.
# images should be input images of shape raw_image_size by raw_image_size, and output is of size image_size x image_size
# input image tensor should be of shape [-1, raw_image_size^2]. Output is of shape [-1, image_size^2]
def padImages(images, rawside, outside):
  padSize = (outside - rawside)/2.0
  images = [np.reshape(image, (rawside, rawside)) for image in images]
  leftPad = int(np.floor(padSize))
  rightPad = int(np.ceil(padSize))
  padImages = np.lib.pad(images, [[0,0],[leftPad, rightPad], [leftPad, rightPad]], 'constant')
  return np.reshape(padImages, (-1,outside*outside))

if __name__=="__main__":
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  print(padImages([[1,1,1,1],[2,2,2,2]], 2, 10))

