# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 06:23:09 2019

@author: Potatsu
"""


import matplotlib.pyplot as plt
import numpy as np

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 

#Let's try 2-d (eccentricity,convexity)
manga = 1
orang = 2
banan = 3  #text to num
feat_no = 2   #dimensions
class_no = 2
 

sample = np.zeros([len(MANGA)+len(BANAN),feat_no+1])
sample[0:len(MANGA),0] = MANGA[:,2]
sample[0:len(MANGA),1] = MANGA[:,3]
sample[0:len(MANGA),2] = manga
sample[len(MANGA):len(sample),0] = BANAN[:,2]
sample[len(MANGA):len(sample),1] = BANAN[:,3]
sample[len(MANGA):len(sample),2] = banan


#Step 1 Give a random mean and variance per sample and prior prob that is equal to 1
mean_list     = np.array(([.2,.8],
                          [.3,.6]))
variance_list = np.ones([2,2,2])
variance_list[:,0,0] = 3
variance_list[:,1,1] = 2
prior_list = np.ones([class_no])
prior_list[:] = 1/2

#Step 2 How likely the sample comes from the first cluster or the second cluster
p_list = np.zeros([len(sample),feat_no])
for i in range(0,len(sample)):
    for j in range(0,class_no):
        
        x_used        = sample[i,0:2]
        mean_used     = mean_list[j,:]
        variance_used = variance_list[j,:,:]
        
        deter         = np.linalg.det(variance_used)       
        x_mew = x_used - mean_used
        cov_inv = np.linalg.inv(variance_used)
        
        expo = (np.matmul(np.matmul(x_mew.T,cov_inv),x_mew))*-1/2
        p_list[i,j] = np.e**(expo) / ((2*np.pi)**(feat_no/2)*(deter)**(1/2))

#Step 3 record old mean, variance, prior
mean_old = np.copy(mean_list)
variance_old = np.copy(variance_list)
prior_old = np.copy(prior_list)

#Step 4 we get a list of prior x gaussian
priorxp_list = np.zeros([len(sample),feat_no*2])
priorxp_list[:,0:2] = prior_list*p_list
priorxp_list[:,2] = np.sum(priorxp_list[:,0:2],axis=1)
priorxp_list[:,3] = np.sum(priorxp_list[:,0:2],axis=1)
priorxp_list[:,0:2] = priorxp_list[:,0:2] / priorxp_list[:,2:4] 

#Step 5 update prior_list and mean_list and variance_list
prior_list = np.sum(priorxp_list[:,0:2],axis=0)/len(sample) #updated priors


samplexpriorxp_list = np.zeros([len(sample),feat_no,len(prior_list)])
for i in range(0, feat_no):
        for j in range(0, len(prior_list)):
            samplexpriorxp_list[:,i,j] = sample[:,i]*priorxp_list[:,j]
    #samplexpriorxp_list[:,0] = sample[:,0]*priorxp_list[:,0]
    #samplexpriorxp_list[:,1] = sample[:,0]*priorxp_list[:,1]
#mean_list = np.sum(samplexpriorxp_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means

#priorxpxvarsq_list = np.copy(sample)
#priorxpxvarsq_list[:,0] = (sample[:,0]-mean_list[0])**2*priorxp_list[:,0]
#priorxpxvarsq_list[:,1] = (sample[:,0]-mean_list[1])**2*priorxp_list[:,1]
#variance_list = np.sum(priorxpxvarsq_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        