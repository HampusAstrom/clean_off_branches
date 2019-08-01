import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import random as rng
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import convolve2d
from scipy import ndimage
import imageio
import visvis as vv
import scipy.ndimage.filters as fi
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

base_kernel=np.array(
                [
                [0.1,   0.1,    0.2,    0.3,    0.2,    0.1,    0.1],
                [0.1,   0.2,    0.3,    0.5,    0.3,    0.2,    0.1],
                [0.2,   0.3,    0.6,    0.7,    0.6,    0.3,    0.2],
                [0.3,   0.5,    0.7,    5.0,    0.7,    0.5,    0.3],
                [0.2,   0.3,    0.6,    0.7,    0.6,    0.3,    0.2],
                [0.1,   0.2,    0.3,    0.5,    0.3,    0.2,    0.1],
                [0.1,   0.1,    0.2,    0.3,    0.2,    0.1,    0.1]]
                , dtype='float64')

imageio.imwrite('base_kernel.png', ((1-base_kernel/5.0) * 255).astype(np.uint8))
base_kernel = base_kernel / sum(sum(base_kernel))
print(sum(sum(base_kernel)))

def clean(image, threashold, kernel=base_kernel):
    print(np.shape(kernel))
    print(np.shape(image))

    temp=convolve2d(image,kernel,mode='same', boundary='fill', fillvalue='255')

    temp[temp > threashold] = 255
    temp[temp <= threashold] = 0

    return temp.astype(np.uint8)

def gen_lin_gauss_kernel(tiltk, mean=0, var=0.3):
    grid = np.array([-1., -2/3, -1/3, 0., 1/3, 2/3, 1.])
    tilt = np.array([1., tiltk])

    kernel = np.zeros((7, 7))
    gauss_kernel = np.zeros((7, 7))

    for i in range(7):
        for j in range(7):
            point = [grid[j], grid[6-i]]
            temp = np.dot(point, tilt) / np.dot(tilt, tilt)
            dist = np.linalg.norm(point - temp * tilt)
            kernel[i][j] = dist
            gauss_kernel[i][j] = ((1./np.sqrt(2 * np.pi * var)) *
                np.exp(-(dist - mean)**2 / (2 * var)))
            #print('{:1.2f}'.format(dist), end='\t')
        #print()

    np.set_printoptions(precision=3)
    #print(kernel)
    #print(gauss_kernel)
    kernel = kernel / sum(sum(kernel))
    gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
    return gauss_kernel, kernel

#im = imageio.imread('ask-hi-res-hyffsad.png')
im = imageio.imread('testbild_for_filter.png')

im = im/255

print(im.shape)
print(im[2000][1000:1100])

gauss_kernel, kernel = gen_lin_gauss_kernel(1, var=0.01)
print(kernel)
print(gauss_kernel)

#for i in range(10):
#    ret = clean(im, i/10)
#    imageio.imwrite('ask-hi-res-hyffsad_{}.png'.format(i/10), ret)


ret = clean(im, 0.002, kernel=gauss_kernel)
#ret = clean(im, 0.3)
imageio.imwrite('test_filters_tilt3.png', ret)

#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg1.png', ret)
#ret = clean(ret, 0.5)
#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg2.png', ret)
#ret = clean(ret, 0.5)
#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg3.png', ret)

#vv.imshow(im)
#input("Press Enter to continue...")
