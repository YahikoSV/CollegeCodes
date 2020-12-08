# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:16:47 2019
@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt


'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.


Activity #1.1
1.) Create a 128x128 image of a white circle, centered and with the
    background black.
'''

### Parameters ###
n = 128                #Aperture Length
R_out = 7              #Radius of Circle Outside (R_out > R_in)
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
            A[i,j] = T_out
        else:
            A[i,j] = 0
            
plt.imshow(A, cmap="Greys_r")
plt.title("1.) 128x128 Circle Image")
plt.show()

'''
2.) Since it is creating here via array, there is no need of grayscale 
    conversion yet. Apply fft2() and its abs()
'''
DP2 = np.fft.fft2(A)
DP = np.abs(DP2)
plt.imshow(DP, cmap="Greys_r")
plt.title("2.) Fourier Transform of Circle Image")
plt.show()

'''
3.) Apply fftshift() and you will see that it isconsistent with the 
    analytical fourier transform of a circle
'''
CP = np.fft.fftshift(DP)
plt.imshow(CP, cmap="Greys_r")
plt.title("3.) Shifted Circle Image")
plt.show()

'''
4.) When you apply fft2() again to 2. It looks like a the circle again due to
    symmetry.
'''          
BP2 = np.fft.fft2(DP2)
BP = np.abs(BP2)
plt.imshow(BP, cmap="Greys_r")
plt.title("4.) Double fft Circle Image")
plt.show()


'''
5.) I made a letter A via photoshop and this time it needs to be grayscaled
'''
F = plt.imread("letA.png")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])
F1 = rgb2gray(F)
plt.imshow(F1, cmap="Greys_r")
plt.title("5.) Grayscaled letter A")
plt.show()

'''
6.) Apply fft2() and its abs() and we get faintly sparky edges
'''
F2 = np.fft.fft2(F1)
F2 = np.abs(F2)
plt.imshow(F2, cmap="Greys_r")
plt.title("6.) Fourier transform of letter A")
plt.show()

'''
7.) When you fftshift() it is like a spark
'''
F3 = np.fft.fftshift(F2)
plt.imshow(F3, cmap="Greys_r")
plt.title("7.) Shifted letter A")
plt.show()

'''
8.) When you apply fft2() again to 6. It does not look like A because it is
    not symmetric at all angles. I guess...
'''          
BP2 = np.fft.fft2(F2)
BP = np.fft.fftshift(np.abs(BP2))
plt.imshow(BP, cmap="Greys_r")
plt.title("8.) Double fft letter A")
plt.show()


'''
9.) Convince yourself it is part real and imaginary by getting a real, imag part
    of A. Comparing these parts to 7, the real part has a radial contribution
    at the edges while the imaginary part has a spiky contribution at the edge.
'''
F4 = np.fft.fft2(F1)
F4R = np.abs(np.real(F4))
F4I = np.abs(np.imag(F4))

plt.imshow(F4R, cmap="Greys_r")
plt.title("9a.) Real part of fft letter A")
plt.show()
plt.imshow(F4I, cmap="Greys_r")
plt.title("9b.) Imag part of fft letter A")
plt.show()
       
'''
I shall now find the fft of a...
10.) Corrugated roof
'''
#Multitude parameter
M = 1/2  #Note: Indirectly proportional to size
xxc = M * xx

B = np.sin(xxc)                 #fucntion of a roof
plt.imshow(B, cmap="Greys_r")
#plt.show()

B2 = np.fft.fft2(B)
B2 = np.fft.fftshift(B2)
B2 = np.abs(B2)
plt.imshow(B2, cmap="Greys_r")
plt.title("10.) FT of Corrugated Roof")
plt.show()

'''
11.) Double SLit
'''
#Width parameter
Wmax = 15  #Note: Width is their difference
Wmin = 10

C = np.copy(xx)

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if np.abs(C[i,j]) >= Wmin and np.abs(C[i,j])<= Wmax:
            C[i,j] = 1
        else:
            C[i,j] = 0
            
plt.imshow(C, cmap="Greys_r")

C2 = np.fft.fft2(C)
C2 = np.fft.fftshift(C2)
C2 = np.abs(C2)
plt.imshow(C2, cmap="Greys_r")
plt.title("11.) FT of Double Slit")
plt.show()
'''
12.) Square fxn
'''

#Area parameter
L1Max = 10  #Note: Range = (-32,32)
L1Min = -10
L2Max = 10
L2Min = -10

D = np.copy(xx)
Dx, Dy = xx,yy

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if Dx[i,j] <= L1Max and Dx[i,j] >= L1Min and  \
           Dy[i,j] <= L2Max and Dy[i,j] >= L2Min:
        
            D[i,j] = 1
        else:
            D[i,j] = 0

D2 = np.fft.fft2(D)
D2 = np.fft.fftshift(D2)
D2 = np.abs(D2)
plt.imshow(D2, cmap="Greys_r")
plt.title("12.) FT of Square Fxn")
plt.show()

'''
13.) and Gaussian Hill
'''
#Gaussain Parameters
mu = 0
sig = 20

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

E = np.sqrt(xx**2 + yy**2)
E = gaussian(E, mu, sig)


E2 = np.fft.fft2(E)
E2 = np.fft.fftshift(E2)
E2 = np.abs(E2)
plt.imshow(E2, cmap="Greys_r")
plt.title("13.) FT of Gaussian Hill")
plt.show()

print("End of Activity 1.1")
