# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:34:41 2019

@author: owner
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.


Activity #1.4
1.) The letters VIP was made from photoshop and was used
    and grayscaled.
'''

VIP = plt.imread("VIP.png")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
gVIP = rgb2gray(VIP)

foVIP = np.fft.fft2(gVIP)
fVIP = np.abs(foVIP)
plt.imshow(fVIP, cmap="Greys_r")
plt.title("1.) VIP Image") 
plt.show()

'''
2.) Create pattern matrices... then
'''
Pattern1 = np.array(([-1,-1,-1],[2,2,2],[-1,-1,-1]))
Pattern2 = np.array(([-1,2,-1],[-1,2,-1],[-1,2,-1]))
Pattern3 = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))

'''
use convulution. I used scipy.signal.convolve2d for convenience
'''

fAP1 = convolve2d(gVIP,Pattern1)
fAP2 = convolve2d(gVIP,Pattern2)
fAP3 = convolve2d(gVIP,Pattern3)
plt.imshow(fAP1,cmap="Greys_r")
plt.title("2a) Result of Horizontal Matrix") 
plt.show()
plt.imshow(fAP2,cmap="Greys_r")
plt.title("2b) Result of Vertical Matrix") 
plt.show()
plt.imshow(fAP3,cmap="Greys_r")
plt.title("2c) Result of Dot Matrix") 
plt.show()

'''
By looking at the results, the edges detected are based from the values of
the pattern matrix. Cool. Only the horizontal and vertical edges are seen in 
2a and 2b respectively while all edges were identified in 2c.
'''
print("End of Activity 1.4")