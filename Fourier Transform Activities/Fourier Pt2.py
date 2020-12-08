# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:40:29 2019

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.


Activity #1.2
1.) The letters VIP and the a circle aperture was made from photoshop was used
    and grayscaled.
'''

VIP = plt.imread("VIP.png")
APE = plt.imread("CirApe.png")
plt.imshow(VIP, cmap="Greys_r")
plt.imshow(APE, cmap="Greys_r")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
gVIP = rgb2gray(VIP)
gAPE = rgb2gray(APE)

plt.imshow(VIP, cmap="Greys_r")
plt.title("1a) VIP Image") 
plt.show()
plt.imshow(APE, cmap="Greys_r")
plt.title("2a) Circle Aperture") 
plt.show()


'''
2.) FT the VIP image while aperture only needs to be shifted since its in the 
    fourier plane. We get the product of the 2 and get its inverse
'''
fAPE = np.fft.fftshift(gAPE)
foVIP = np.fft.fft2(gVIP)
fVIP = np.abs(foVIP)

fVA = foVIP * fAPE
iVA = np.abs(np.fft.ifft2(fVA))
plt.imshow(iVA, cmap="Greys_r")
plt.title("2.) Convoluted VIP with Circle Aperture") 
plt.show()
'''
Although the aperture radius is fixed ill use the cirlce in 1 then
'''


### Parameters ###
n = 128                #Aperture Length
R_out = 15              #Radius of Circle Outside (R_out > R_in)
T_out = 1               #Transmittance ranges from 0 to 1

### Aperture Function Code ###

I = np.arange(0,n,1)    #Counts from 0 to 1023
x = I-(n/2)             #Counts from -512 to 511
y = (n/2)-I             #Making the y-axis upright when isng imshow

xx,yy = np.meshgrid(x,y)   #meshgrid: an array with an x,y value embedded
A = xx**2 + yy**2       #fucntion of a circle


#Making the circles 
for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if A[i,j] <= R_out**2:
            A[i,j] = 1
        else:
            A[i,j] = 0

plt.imshow(A, cmap="Greys_r")
plt.title("3.) A Smaller Circle Aperture yields...") 
plt.show()


'''
4. Same as No.2 (Note: next time do not reuse variables)
'''
fAPE = np.fft.fftshift(A)
fVA = foVIP * fAPE
iVA = np.abs(np.fft.ifft2(fVA))
plt.imshow(iVA, cmap="Greys_r")
plt.title("4.) ...a more blurred but smoother image with some abberations") 
plt.show()


print("End of Activity 1.2")