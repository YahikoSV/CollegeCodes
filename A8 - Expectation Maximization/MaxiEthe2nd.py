# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:27:34 2019

@author: Potatsu
"""

import matplotlib.pyplot as plt
import numpy as np

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 


# r_norm, g_norm, eccentricity, convexity
sample = np.copy(MANGA)

#########   STARTO!!!   #################
#Step 0: Parameters
starting_p = 1/4
no_features = len(sample.T)   #how many items
d = 4 #dimensions 
N = len(sample) # no of manga


# Step 1: Make Matrices 
P_big = np.zeros([len(sample.T)])
P_big[:] = starting_p                #Prior probabilities be init 1/M

cov_mat = np.ones([no_features,no_features,no_features])  #let all covariance matrices be equal
cov_mat[:,:,0] = np.cov(sample.T)
cov_mat[:,:,1] = np.cov(sample.T)
cov_mat[:,:,2] = np.cov(sample.T)
cov_mat[:,:,3] = np.cov(sample.T)

cov_med = np.zeros([len(sample.T),len(sample.T)])

muse = np.zeros(no_features) 
muse_new = np.zeros(no_features) 
muse_med = np.zeros([len(sample),len(sample.T)])
muse_real = np.mean(sample.T, axis=1)
#muse[:] = .5
muse = np.copy(muse_real)

p_med   = np.zeros([len(sample),len(sample.T)])
p_small = np.zeros([len(sample),len(sample.T)])
p_msum  = np.zeros([len(sample)])
P_old   = np.zeros([len(sample.T)])
P_new   = np.zeros([len(sample.T)])


#Step 2: Solve the pdf per sample per component

x_mew = np.zeros([no_features])
for i in range (0,len(sample)):
    for j in range (0,no_features):
        deter = np.linalg.det(cov_mat[:,:,j])  #determinant
        x_mew[:] = sample[i,j]-muse[j]
        cov_inv = np.linalg.inv(cov_mat[:,:,j])
        expot = (np.matmul(np.matmul(x_mew.T,cov_inv),x_mew))*-1/2
        
        p_small[i,j] = np.e**(expot) / ((2*np.pi)**(d/2)*(deter)**(1/2))

#Step 3 Combining step        
for i in range (0,len(sample)):
    for j in range (0,4):
        p_med[i,j] = P_big[j]*p_small[i,j]   #P*p
        
for i in range (0,len(sample)):
    p_msum[i] = sum(p_med[i,:])              #sum of P*p

for i in range (0,len(sample)):
    for j in range (0,4):
        p_med[i,j] = p_med[i,j]/p_msum[i]    #fraction of P*p
        
#Step 4 Updating step
        
for i in range (0,len(sample)):
    for j in range (0,4):
        muse_med[i,j] = sample[i,j]*p_med[i,j]          
        
for i in range (0,4):
    P_new[i] = sum(p_med[:,i])*1/len(sample)
    muse_new[i] = sum(muse_med[:,i])/sum(p_med[:,i])
'''
x_mew = np.zeros([no_features])
for j in range (0,4):
    cov_sum = np.zeros([len(sample)])
    for i in range (0,len(sample)):
        x_mew[:] = sample[i,j]-muse_new[j]
        cov_med  = p_med[i,j]*(np.matmul(np.matmul(x_mew,cov_mat),x_mew.T))
        #cov_sum  = cov_sum + cov_med
        
a = np.matmul(x_mew,cov_mat)
b = np.matmul(a,x_mew.T)
'''