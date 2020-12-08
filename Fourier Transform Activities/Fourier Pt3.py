# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:08:55 2019

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.


Activity #1.3
1.) The letter A and the a text was made from paint and was used
    and grayscaled.
'''

letA2 = plt.imread("letA2.png")
Atext = plt.imread("SPAIN.png")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
gtext = rgb2gray(Atext)
gA    = rgb2gray(letA2)


plt.imshow(gA, cmap="Greys_r")
plt.title("1a) Letter A") 
plt.show()
plt.imshow(gtext, cmap="Greys_r")
plt.title("2a) Text Image") 
plt.show()

'''
2.) Get both of their fourier transforms then...
'''
ftext =  np.fft.fft2(gtext)
fA    =  np.fft.fft2(gA)

'''
Get conj() of the text image then...
'''
cftext = np.conj(ftext)
'''
Multiply fA with cftext and get its ifft
'''

fAT = fA * cftext
ifAT = np.fft.ifft2(fAT)
ifAT2 = np.abs(ifAT)
ifAT3 = np.fft.fftshift(ifAT2)
plt.imshow(ifAT3, cmap="Greys_r")
plt.title("3) Correlated result of the text with ifft")
plt.show()

'''
4.) Comment: When fft it not baliktad but when its ifft its baliktad???
    The images in 3 or 4 show that there is a peaks happening on the places
    where a letter "reside" meaning that this method can be used to detect 
    patterns.
'''

ifAT = np.fft.fft2(fAT)
ifAT2 = np.abs(ifAT)
ifAT3 = np.fft.fftshift(ifAT2)
plt.imshow(ifAT3, cmap="Greys_r")
plt.title("4) Correlated result of the text with fft")
plt.show()


print("End of Activity 1.3")