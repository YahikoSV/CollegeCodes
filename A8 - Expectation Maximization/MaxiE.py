# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:38:02 2019

@author: Potatsu
"""


import matplotlib.pyplot as plt
import numpy as np

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 


# r_norm, g_norm, eccentricity, convexity
sample = np.copy(MANGA).T

cov_sam = np.cov(sample)
cov_mean = np.mean(sample, axis=1)

cov_inv = np.linalg.inv(cov_sam)
x_mew = np.zeros(4)
x_mew[:] = sample[0,0]-cov_mean[0]
a = np.matmul(np.matmul(x_mew.T,cov_inv),x_mew)


#########   STARTO!!!   #################
#Step 0: Parameters
starting_p = 1/4
no_classes = 3
no_features = 4   #how many items

manga = 0
orang = 1
banan = 2  #text to num

d = 4 #dimensions 
N = len(MANGA) # no of manga
# Step 1: Make Matrices (Note: Manga0 -> Orang1 -> Banan2)
P_big = np.zeros([no_classes,no_features])
P_big[:,:] = starting_p     #Prior probabilities

#p_small = np.zeros([no_classes,no_features])   #Component Probabilities


muse = np.zeros([no_classes,no_features])      #mean collection
muse[manga,:] = np.mean(MANGA.T, axis=1)
muse[orang,:] = np.mean(ORANG.T, axis=1)
muse[banan,:] = np.mean(BANAN.T, axis=1)


cov_mat = np.zeros([no_features,no_features,no_classes])  #covariance mat collection
cov_mat[:,:,manga] = np.cov(MANGA.T)
cov_mat[:,:,orang] = np.cov(ORANG.T)
cov_mat[:,:,banan] = np.cov(BANAN.T)


x_mew = np.zeros(4)  #Xi - Mew

#Step 2: Create a matrix for the samples
# r,g,eccen,convex,id no.,p_small value
sample = np.zeros([len(ORANG)+len(MANGA)+len(BANAN),6])
sample[0:len(MANGA),0:4]  = MANGA
sample[0:len(MANGA),4] = manga
sample[len(MANGA):len(MANGA)+len(ORANG),0:4]  = ORANG
sample[len(MANGA):len(MANGA)+len(ORANG),4] = orang
sample[len(MANGA)+len(ORANG):len(ORANG)+len(MANGA)+len(BANAN),0:4]  = BANAN
sample[len(MANGA)+len(ORANG):len(ORANG)+len(MANGA)+len(BANAN),4] = banan

sample2 = np.copy(sample[0:len(MANGA),:])  #test


p_small = np.zeros([len(sample2),4])

for i in range (0,len(sample2)):
    for j in range (0,4):
        deter = np.linalg.det(cov_mat[:,:,manga])  #determinant
        x_mew[:] = sample2[i,j]-muse[manga,j]
        cov_inv = np.linalg.inv(cov_mat[:,:,manga])
        expot = (np.matmul(np.matmul(x_mew.T,cov_inv),x_mew))*-1/2
        
        p_small[i,j] = np.e**(expot) / ((2*np.pi)**(d/2)*(deter)**(1/2))

#Step 3 Combining step
        
p_med  = np.zeros([len(sample2),4])
p_msum  = np.zeros([len(sample2)])

       
for i in range (0,len(sample2)):
    for j in range (0,4):
        p_med[i,j] = P_big[0,j]*p_small[i,j]   #P*p
        
for i in range (0,len(sample2)):
    p_msum[i] = sum(p_med[i,:])

for i in range (0,len(sample2)):
    for j in range (0,4):
        p_med[i,j] = p_med[i,j]/p_msum[i] 

#Step 4 Updating step
        
x_med = np.zeros([len(sample2),4])
for i in range (0,len(sample2)):
    for j in range (0,4):
        x_med[i,j] = sample2[i,j]*p_med[i,j]          
        
for i in range (0,4):
    P_big[manga,i] = sum(p_med[:,i])*1/11  
    muse[manga,i:] = sum(x_med[:,i])/sum(p_med[:,i])





