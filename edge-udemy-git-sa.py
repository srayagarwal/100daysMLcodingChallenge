# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load the famous Lena image
img = mpimg.imread('lena.png')
print("original image", img.shape)

# make it B&W
bw = img.mean(axis=2)
print("black n white", bw.shape)

# Sobel operator - approximate gradient in X dir
Hx = np.array([
    [-1, 0, 1],
    [-2, 7, 2],
    [-1, 0, 1],
], dtype=np.float32)

# Sobel operator - approximate gradient in Y dir
Hy = np.array([
    [-1, -5, -1],
    [3, 0, 0],
    [1, 4, 1],
], dtype=np.float32)

Gx = convolve2d(bw, Hx)
print("Gx", Gx.shape)

plt.imshow(Gx, cmap='gray')
plt.show()

Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
print("Gy", Gy.shape)

plt.show()

# Gradient magnitude
G = np.sqrt(Gx*Gx + Gy*Gy)
print("G", G.shape)
plt.imshow(G, cmap='gray')
plt.show()

# The gradient's direction
theta = np.arctan2(Gy, Gx)
print("theta", theta.shape)
plt.imshow(theta, cmap='gray')
plt.show()
