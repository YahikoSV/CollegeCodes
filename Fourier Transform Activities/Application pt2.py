# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:49:10 2019

@author: owner
"""
import numpy as np
import matplotlib.pyplot as plt

### Parameters ###
n = 256                 #Aperture Length

### Aperture Function Code ###

I = np.arange(0,n,1)    #Counts from 0 to 1023
x = I-(n/2)             #Counts from -512 to 511
y = (n/2)-I             #Making the y-axis upright when isng imshow

xx,yy = np.meshgrid(x,y)   #meshgrid: an array with an x,y value embedded


'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.

Activity #2.1
1.) When changing the frequency, the ridges become more dense as f increases
    and vice versa
'''
#Multitude parameter
f =  .8          #Note: Indirectly proportional to size
f2 = .2

B,H = np.sin(f*xx),np.sin(f2*xx) #fucntion of a roof
plt.imshow(B, cmap="Greys_r")
plt.title("1a) Sine fxn")
plt.show()
plt.imshow(H, cmap="Greys_r")
plt.title("1b) Sine fxn with lesser f")
plt.show()

'''
2) When getting their fourier transform, the spacing of the two dots from
   each other increases as f increases.
'''
B2, H2 = np.fft.fft2(B), np.fft.fft2(H)
B2, H2 = np.fft.fftshift(B2), np.fft.fftshift(H2)
B2, H2 = np.abs(B2), np.abs(H2)
plt.imshow(B2, cmap="Greys_r")
plt.title("2a) FT ofSine fxn")
plt.show()
plt.imshow(H2, cmap="Greys_r")
plt.title("2b) Sine fxn with lesser f")
plt.show()

'''
Digital images have no negative values. Simulate a real image by adding a
constant bias to the sinusoid you generated in Step 1. Take the FT. What
do you observe? Suppose you took an image of an interferogram in a
Young's Double Slit experiment. What can you do to find the actual
frequencies? Suppose a non-constant bias is added to the sinusoid (e.g.
very low frequency sinusoids), how can you get the frequencies of the
interferogram?
'''

'''
3) When the sinusoid is rotated the FT also rotates.

'''
theta = np.pi/6
C = np.sin(f*(yy*np.sin(theta)+xx*np.cos(theta)))
plt.imshow(C, cmap="Greys_r")
plt.title("3) Sine fxn rotated by pi/6")
plt.show()

C2 = np.fft.fft2(C)
C2 = np.fft.fftshift(C2)
C2 = np.abs(C2)
plt.imshow(C2, cmap="Greys_r")
plt.title("3b) FT of Sine fxn rotated by pi/6")
plt.show()


'''
When adding more sinusoids in the equation, the result is that the get their
respective pairs of dots in fourier space. Seems that you can get superposition
of waves easily with the fourier transform
'''
theta2 = np.pi/12
theta3 = np.pi/2

#D = np.sin(f*xx)*np.sin(f*yy) + np.sin(f*xx) + np.sin(f*(yy*np.sin(theta)+xx*np.cos(theta)))    #try adding and multiplying
D = np.sin(f*(yy*np.sin(theta)+xx*np.cos(theta)))   +\
    np.sin(f*(yy*np.sin(theta2)+xx*np.cos(theta2))) +\
    np.sin(f*(yy*np.sin(theta3)+xx*np.cos(theta3)))
    
plt.imshow(D, cmap="Greys_r")
plt.title("4a) Superposition of rotated sine fxns")
plt.show()

D2 = np.fft.fft2(D)
D2 = np.fft.fftshift(D2)
D2 = np.abs(D2)
plt.imshow(D2, cmap="Greys_r")
plt.title("4b) FT of Superposition of rotated sine fxns")
plt.show()

print("End of Activity 2.2")