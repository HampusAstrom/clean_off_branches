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

def clean(image, threashold):
    kernel=np.array(
                    [
                    [0.1,   0.1,    0.2,    0.3,    0.2,    0.1,    0.1],
                    [0.1,   0.2,    0.3,    0.5,    0.3,    0.2,    0.1],
                    [0.2,   0.3,    0.6,    0.7,    0.6,    0.3,    0.2],
                    [0.3,   0.5,    0.7,    5.0,    0.7,    0.5,    0.3],
                    [0.2,   0.3,    0.6,    0.7,    0.6,    0.3,    0.2],
                    [0.1,   0.2,    0.3,    0.5,    0.3,    0.2,    0.1],
                    [0.1,   0.1,    0.2,    0.3,    0.2,    0.1,    0.1]]
                    , dtype='float64')

    imageio.imwrite('base_kernel.png', ((1-kernel/5.0) * 255).astype(np.uint8))
    kernel = kernel / sum(sum(kernel))
    print(sum(sum(kernel)))
    temp=convolve2d(image,kernel,mode='same', boundary='fill', fillvalue='255')

    temp[temp > threashold] = 255
    temp[temp <= threashold] = 0

    return temp.astype(np.uint8)

im = imageio.imread('ask-hi-res-hyffsad.png')

im = im/255

print(im.shape)
print(im[2000][1000:1100])

#for i in range(10):
#    ret = clean(im, i/10)
#    imageio.imwrite('ask-hi-res-hyffsad_{}.png'.format(i/10), ret)


ret = clean(im, 0.5)
#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg1.png', ret)
#ret = clean(ret, 0.5)
#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg2.png', ret)
#ret = clean(ret, 0.5)
#imageio.imwrite('ask-hi-res-hyffsad_0.5_steg3.png', ret)

#vv.imshow(im)
#input("Press Enter to continue...")
