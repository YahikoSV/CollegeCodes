# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:13:39 2019

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.

Activites #2.4-6
1.) These are the images used for filtering and are converted to grayscale
'''

fing = plt.imread("Finger.png")
luna = plt.imread("luna.jpg")
canv = plt.imread("canvasweave.jpg")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

fing_2, canv_2 = rgb2gray(fing), rgb2gray(canv)

plt.imshow(fing_2, cmap="Greys_r")
plt.title("1a) Fingerprint Image")
plt.show()
plt.imshow(luna, cmap="Greys_r")
plt.title("1b) Moon Image")
plt.show()
plt.imshow(canv_2, cmap="Greys_r")
plt.title("1c) Canvas Image")
plt.show()

'''
2.) For visualization, get its fourier transform abs() it and log() it. 
    In Log scale they will look like these...
'''

def fftimage(Image):
    Image2 = np.fft.fft2(Image)
    Image2 = np.fft.fftshift(Image2)
    Image2 = np.abs(Image2)
    Image2 = np.log(Image2)
    return Image2

fing2, luna2, canv2 = fftimage(fing_2), fftimage(luna), fftimage(canv_2)
plt.imshow(fing2, cmap="viridis_r")
plt.title("2a) Visual for Fingerprint Image FT")
plt.show()
plt.imshow(luna2, cmap="viridis_r")
plt.title("2b) Visual for Moon Imag FTe")
plt.show()
plt.imshow(canv2, cmap="viridis_r")
plt.title("2c) Visual for Canvas Image FT")
plt.show()


'''
3.) Design a filter mask to enhance the appearance of these ridges at the same
time remove the blotches.
My technique is using a threshold parameter on each image in 2 that decides 
the shape of the filter.
'''

#Threshold Parameters
P_Fing = 3.5
P_Luna = 10
P_Canv = 9

fing6, luna6, canv6 = np.copy(fing2), np.copy(luna2), np.copy(canv2)
def filtermaker(image,im_copy,param):
    for i in range (0,len(image)):
        for j in range (0,len(image[0])):
            if image[i,j] >= param:
                im_copy[i,j] = 1
            else:
                im_copy[i,j] = 0
    return im_copy

fing3, luna3, canv3 = filtermaker(fing2, fing6, P_Fing), \
                      filtermaker(luna2, luna6, P_Luna), \
                      filtermaker(canv2, canv6, P_Canv)
plt.imshow(fing6, cmap="viridis_r")
plt.title("3a) Visual for Fingerprint Aperture")
plt.show()
plt.imshow(luna6, cmap="viridis_r")
plt.title("3b) Visual for Moon Aperture")
plt.show()
plt.imshow(canv6, cmap="viridis_r")
plt.title("3c) Visual for Canvas Aperture")
plt.show()

'''
4.) Use what I learn in Activity #1.2 with this function
'''
def fftfilter(image,fill):       
    f1 = np.fft.fftshift(fill) 
    f2 = np.fft.fft2(image)  
    f3 = f1*f2
    f4 = np.abs(np.fft.ifft2(f3))
    return f4 
   
fing5, luna5, canv5 = fftfilter(fing_2,fing3), fftfilter(luna,luna3), \
                      fftfilter(canv_2,canv3)    

plt.imshow(fing5, cmap="Greys_r")
plt.title("4a) Filtered Image for Fingerprint")
plt.show()
plt.imshow(luna5, cmap="Greys_r")
plt.title("4b) Filtered Image for Moon")
plt.show()
plt.imshow(canv5, cmap="Greys_r")
plt.title("4c) Filtered Image for Canvas")
plt.show()
            
'''
The result of the filters were unsatisfactory. The smudges of the fingerprint,
the horizontal lines of the moon and the weave patterns are still present. Thus,
another way is to manually create an aperture using photoshop or paint.

(Run out of time to submit RIP)
'''

"""
Canvas curiosity: if the filter is reversed, it doesn't look like the canvas 
weave since the initial filter is wrong to begin with...
"""

def filtermaker_r(image,im_copy,param):
    for i in range (0,len(image)):
        for j in range (0,len(image[0])):
            if image[i,j] >= param:
                im_copy[i,j] = 0
            else:
                im_copy[i,j] = 1
    return im_copy
canv7 = np.copy(canv2)
canv8 = filtermaker_r(canv2,canv7,P_Canv)            
canv9 = fftfilter(canv_2,canv8)
plt.imshow(canv9, cmap="Greys_r")
plt.title("## Filtered Image for Canvas (reverse filter)")
plt.show()
            
print("Not really the end of Activities #2.4-6")
            
            