import numpy as np
import math
from scipy.stats import norm
import cv2

# we choose our dimensions in such a way that this can be easily
# viewed when using python's default print statement
# This returns a tensor of the h+1 basis filters
# for rotationally symmetric filters
# TODO: do we want to padd with more zeros? I think we're fine, but maybe worth asking
def createBasisWeights(h):
  w = np.zeros((h+1, 2*h+1, 2*h+1))

  for n in range(h+1):
    for x in range(-h, h+1):
      for y in range(-h, h+1):
        r = np.sqrt(x**2 + y**2) 
        if r > n-1 and r < n:
          w[n,x+h, y+h] = (r - (n-1)) / (n + 1)
        if r >= n and r < n+1:
          w[n,x+h, y+h] = ((n+1) - r) / (n + 1)
  return w

def createBasisWeightsGaussian(h, sigma):
  w = np.zeros((h+1, 2*h+1, 2*h+1))

  for n in range(h+1):
    for x in range(-h, h+1):
      for y in range(-h, h+1):
        r = np.sqrt(x**2 + y**2)
        w[n,x+h, y+h] = norm.pdf(r, n, sigma)
  return w

def createBasisWeightsDiameter(d):
  h = math.floor(d/2)
  w = np.zeros((math.ceil(d/2), d, d))

  for n in range(math.ceil(d/2)):
    for x in range(math.ceil(-d/2), math.ceil(d/2)):
      for y in range(math.ceil(-d/2), math.ceil(d/2)):
        r = np.sqrt(x**2 + y**2) 
        if r > n-1 and r < n:
          w[n,x+h, y+h] = r - (n-1)
        if r >= n and r < n+1:
          w[n,x+h, y+h] = (n+1) - r
  return w

def createBasisWeightsDiameterGaussian(d, sigma):
  w = np.zeros((math.ceil(d/2), d, d))
  c = (d-1)/2

  for n in range(math.ceil(d/2)):
    for x in range(0, d):
      for y in range(0, d):
        r = np.sqrt((x-c)**2 + (y-c)**2) 
        w[n,x, y] = norm.pdf(r, n, sigma)
  return w

if __name__=='__main__':
  # make sure our function looks right
  w = createBasisWeights(3)
  print(w)

  print(w -np.transpose(w,(0,2,1)))

  # make sure we know how to use tensordot to create a linear combination of our filters
  p = np.ones((3,))
  #comb = np.tensordot(w, p, [0,0])
  #print(comb) 
  # looks correct to me!

  basis = np.reshape(createBasisWeights(10)[10,:,:],(21,21))
  angle = 45
  cols = 21
  rows = 21
  M = cv2.getRotationMatrix2D((cols/2 - 0.5,rows/2 - 0.5),angle,1)
  Minv = cv2.getRotationMatrix2D((cols/2 - 0.5,rows/2 - 0.5),-angle,1)
  
  rotimg = cv2.warpAffine(basis,M,(21,21))
  cv2.imshow("basis", basis * 3)
  cv2.imshow("rotbasis",rotimg * 3) 
  cv2.imshow("rotdiff", (basis - rotimg) * 3)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  


  
