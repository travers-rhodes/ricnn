import numpy as np
import math

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

if __name__=='__main__':
  # make sure our function looks right
  w = createBasisTensor(2)
  print(w)

  # make sure we know how to use tensordot to create a linear combination of our filters
  p = np.ones((3,))
  comb = np.tensordot(w, p, [0,0])
  #print(comb) 
  # looks correct to me!
