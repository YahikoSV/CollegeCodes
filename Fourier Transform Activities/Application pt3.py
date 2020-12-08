# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:26:50 2019

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt

### Definitions ###

def fftimage(Image):
    Image2 = np.fft.fft2(Image)
    Image2 = np.fft.fftshift(Image2)
    Image2 = np.abs(Image2)
    return Image2

### Parameters ###
n = 128                #Aperture Length

### Aperture Function Code ###

I = np.arange(0,n,1)    #Counts from 0 to 1023
x = I-(n/2)             #Counts from -512 to 511
y = (n/2)-I             #Making the y-axis upright when isng imshow

xx,yy = np.meshgrid(x,y)   #meshgrid: an array with an x,y value embedded

'''
IMPORTANT: Before running clear all variables. All the plots and images will
appear in order.

Acitvity #2.3
1.) Create a binary image of two dots (one pixel each) along the x-axis
symmetric about center. Take the FT and display the modulus. The parameter
"Dot1" can be used to vary the width. Observing it, the FT of it is a sine 
wave and as the distance between them increases, the density of the since
wave increases (higher frequency).
'''
#Width parameter
Dot1 = 8  #Note: Width is their difference
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
plt.title("1a) Two dots")
plt.show()

c2 = np.fft.fft2(c)
c2 = np.fft.fftshift(c2)
c2 = np.abs(c2)
plt.imshow(c2, cmap="Greys_r")
plt.title("1b) FT of Two dots")
plt.show()

'''
2.) Replace the dots with circles of some radius. You can vary the width with
    parameter "Rc". As the width increases its fourier transform becoming more
    complex while there are sine wave patterns retained the looks the same
    as 1b.
'''
Dc = 8 #distance from center
Rc = 3  #radius
xxc = np.abs(xx)-Dc
yyc = np.abs(yy)
A = xxc**2 + yyc**2       #fucntion of a circle


#Making the circles 

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if A[i,j] <= Rc**2:
            A[i,j] = 1
        else:
            A[i,j] = 0
            
plt.imshow(A, cmap="Greys_r")
plt.title("2a) FT of Two Circles")
plt.show()

A2 = np.fft.fft2(A)
A2 = np.fft.fftshift(A2)
A2 = np.abs(A2)
plt.imshow(A2, cmap="Greys_r")
plt.title("2b) FT of Two Circles")
plt.show()

'''
3.) Replace the dots with squares of some width. You can vary their lengths and
    width with "L1Max" and "L2Max" respectively. By increasing those parameters,
    more black lines appear (more complex).
'''

#Area parameter
X_dist = 8
L1Max = 2  #Note: Range = (-32,32)
L2Max = 2

D = np.copy(xx)
Dx, Dy = np.copy(xx),np.copy(yy)
Dx = np.abs(Dx)-X_dist
Dy = np.abs(Dy)

for i in range (0,len(I)):
    for j in range (0,len(I)):
        
        if np.abs(Dx[i,j]) <= np.abs(L1Max) and \
           np.abs(Dy[i,j]) <= np.abs(L2Max):
        
            D[i,j] = 1
        else:
            D[i,j] = 0
            
plt.imshow(D, cmap="Greys_r")
plt.title("3a) Two Squares")
plt.show()
D2 = fftimage(D)
plt.imshow(D2, cmap="Greys_r")
plt.title("3b) FT of Two Squares")
plt.show()

'''
4.) Replace the dots with Gaussians. You can change sigma by using the 
    parameter "sig". As sigma increases, the fourier transform shrinks.
'''

Dg = 8 #distance from center
xxg = np.abs(xx)-Dg
yyg = np.abs(yy)
E = xxg**2 + yyg**2       #fucntion of a radius

#Gaussain Parameters
mu = 0
sig = 5

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

E = gaussian(E, mu, sig)
plt.imshow(E, cmap="Greys_r")
plt.title("4a) Two Gaussians")
plt.show()
E2 = fftimage(E)
plt.imshow(E2, cmap="Greys_r")
plt.title("4b) FT of Two Gaussians")
plt.show()


'''
5.) Create a 200×200 array of zeros. Put 10 1's in random locations in the
array. These ones will approximate dirac deltas.
'''
No = 200
Lattice = np.zeros((No,No))
Chosen = []
np.random.seed(32)

for i in range (0,10):
    X, Y = np.random.randint(No),np.random.randint(No)
    if Lattice[X,Y] == 0:
        Lattice[X,Y] = 1 
        Chosen.append([X,Y])
    else:
        i = i - 1
plt.imshow(Lattice, cmap="Greys_r")
plt.title("5a) Some random points")
plt.show()       
'''
Create an arbitrary 9×9 pattern, call it d. 
'''   
Pattern1 = np.array(([-1,-1,-1],[2,2,2],[-1,-1,-1]))
Pattern2 = np.array(([-1,2,-1],[-1,2,-1],[-1,2,-1]))
Pattern3 = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))

'''
Convolve A and d. What do you observe?  I observed that the resulting FTs
have dirac deltas that look like the patterns.
'''
from scipy.signal import convolve2d

fAP1 = convolve2d(Lattice,Pattern1)
fAP2 = convolve2d(Lattice,Pattern2)
fAP3 = convolve2d(Lattice,Pattern3)
plt.imshow(fAP1, cmap="Greys_r")
plt.title("5b) Convolving lattice with horizontal pattern")
plt.show()  
plt.imshow(fAP2, cmap="Greys_r")
plt.title("5c) Convolving lattice with vertical pattern")
plt.show()    
plt.imshow(fAP3, cmap="Greys_r")
plt.title("5d) Convolving lattice with dot pattern")
plt.show()  
'''
6.) Create another 200×200 array of zeros but this time put equally spaced 1's
along the x- and y-axis in the array. Get the FT and display the modulus.
Change the spacing the 1's and repeat. You can change the spacing by changing
the parameter "Space." The FT of this forms a grid and as the spacing increases,
the grid lines become more dense or compact to each other.
'''

Space = 4
Lattice2 = np.zeros((No,No))
for i in range (0,No):
    for j in range (0,No):
        if i%Space == 0 and j == 100: 
            Lattice2[i,j] = 1 
        elif j%Space == 0 and i == 100: 
            Lattice2[i,j] = 1         
        else:
            Lattice2[i,j] = 0 
plt.imshow(Lattice2,cmap="Greys_r")
plt.title("6a) Equally Spaced 1s")
plt.show()

flattice2 = fftimage(Lattice2)
plt.imshow(flattice2, cmap="Greys_r")
plt.title("6b) FT of 6a")
plt.show()

print("End of Activity 2.3")