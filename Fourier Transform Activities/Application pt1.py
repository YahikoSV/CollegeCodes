# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 03:21:20 2019

@author: owner
"""


import numpy as np
import matplotlib.pyplot as plt

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.


Activity #2.1
'''

### Parameters ###
n = 128                #Aperture Length               #Transmittance ranges from 0 to 1

### Aperture Function Code ###

I = np.arange(0,n,1)    #Counts from 0 to 1023
x = I-(n/2)             #Counts from -512 to 511
y = (n/2)-I             #Making the y-axis upright when isng imshow

xx,yy = np.meshgrid(x,y)   #meshgrid: an array with an x,y value embedded

'''
I Created these patterns and get its fourier transform...
a) Tall rectangle aperture
b) Wide rectangle aperture
c) Two dots along x-axis symmetric about the center
d) Same as (c) with different spacing between the dots.
'''

'''
a) Tall rectangle aperture
'''
#Width parameter
Wmax = 5  #Note: Width is their difference
Wmin = -5

a = np.copy(xx)

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if a[i,j] >= Wmin and a[i,j]<= Wmax:
            a[i,j] = 1
        else:
            a[i,j] = 0
            
plt.imshow(a, cmap="Greys_r")

a2 = np.fft.fft2(a)
a2 = np.fft.fftshift(a2)
a2 = np.abs(a2)
plt.imshow(a2, cmap="Greys_r")
plt.title("a) FT of Tall Rectangle Aperture") 
plt.show()

'''
b) Wide rectangle aperture
'''
#Width parameter
W2max = 5  #Note: Width is their difference
W2min = -5
b = np.copy(yy)

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if b[i,j] >= W2min and b[i,j]<= W2max:
            b[i,j] = 1
        else:
            b[i,j] = 0
            
plt.imshow(a, cmap="Greys_r")

b2 = np.fft.fft2(b)
b2 = np.fft.fftshift(b2)
b2 = np.abs(b2)
plt.imshow(b2, cmap="Greys_r")
plt.title("b) FT of Wide Rectangle Aperture") 
plt.show()


'''
c) Two dots along x-axis symmetric about the center
'''
#Width parameter
Dot1 = 5  #Note: Width is their difference
Dot2 = -Dot1
c = np.copy(yy)
cx, cy = xx, yy

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if np.abs(cx[i,j]) == Dot1 and cy[i,j]== 0:
            c[i,j] = 1
        else:
            c[i,j] = 0
            
plt.imshow(c, cmap="Greys_r")

c2 = np.fft.fft2(c)
c2 = np.fft.fftshift(c2)
c2 = np.abs(c2)
plt.imshow(c2, cmap="Greys_r")
plt.title("c) FT of Two Dots") 
plt.show()


'''
d) Same as (c) but diff width distance
'''
#Width parameter
Dot1 = 10  #Note: Width is their difference
Dot2 = -Dot1
d = np.copy(yy)
dx, dy = xx, yy

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if np.abs(cx[i,j]) == Dot1 and cy[i,j]== 0:
            d[i,j] = 1
        else:
            d[i,j] = 0
            
plt.imshow(d, cmap="Greys_r")

d2 = np.fft.fft2(d)
d2 = np.fft.fftshift(d2)
d2 = np.abs(d2)
plt.imshow(d2, cmap="Greys_r")
plt.title("d) FT of Two different distanced dots") 
plt.show()

print("End of Activity 2.1")
