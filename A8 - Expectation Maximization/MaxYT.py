# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:43:55 2019

@author: Potatsu
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 

#Let's try 1-d (convexity)
manga = 1
orang = 2
banan = 3  #text to num

sample = np.zeros([len(MANGA)+len(BANAN),2])
sample[0:len(MANGA),0] = MANGA[:,3]
sample[0:len(MANGA),1] = manga
sample[len(MANGA):len(sample),0] = BANAN[:,3]
sample[len(MANGA):len(sample),1] = banan


#Step 1 Give a random mean and variance per sample and prior prob that is equal to 1
mean_list     = np.array([.2,.8])
variance_list = np.array([1,1])
prior_list = np.array([1/2,1/2])

#Step 2 How likely the sample comes from the first cluster or the second cluster

p_list = np.zeros([len(sample),len(sample.T)])
for i in range(0,len(sample)):
    for j in range(0,len(sample.T)):
        expo = -(sample[i,0]-mean_list[j])**2 / (2*variance_list[j])
        p_list[i,j] = np.e**(expo) / (((2*np.pi)**(1/2))*((variance_list[j])**(1/2)))
        
#Step 3 record old mean, variance, prior
mean_old = np.copy(mean_list)
variance_old = np.copy(variance_list)
prior_old = np.copy(prior_list)



#Step 4 we try to guess their prior porbabilities
priorxp_list = np.zeros([len(sample),len(sample.T)*2])
priorxp_list[:,0:2] = prior_list*p_list
priorxp_list[:,2] = np.sum(priorxp_list[:,0:2],axis=1)
priorxp_list[:,3] = np.sum(priorxp_list[:,0:2],axis=1)
priorxp_list[:,0:2] = priorxp_list[:,0:2] / priorxp_list[:,2:4] 


#Step 5 update prior_list and mean_list
prior_list = np.sum(priorxp_list[:,0:2],axis=0)/len(sample) #updated priors

samplexpriorxp_list = np.copy(sample)
samplexpriorxp_list[:,0] = sample[:,0]*priorxp_list[:,0]
samplexpriorxp_list[:,1] = sample[:,0]*priorxp_list[:,1]
mean_list = np.sum(samplexpriorxp_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means

priorxpxvarsq_list = np.copy(sample)
priorxpxvarsq_list[:,0] = (sample[:,0]-mean_list[0])**2*priorxp_list[:,0]
priorxpxvarsq_list[:,1] = (sample[:,0]-mean_list[1])**2*priorxp_list[:,1]
variance_list = np.sum(priorxpxvarsq_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means

#Step 6
diff = np.array([abs(mean_list-mean_old),abs(variance_list-variance_old),abs(prior_list-prior_old)])
diff = diff.max()
print(diff)

#Step 6 Iterate!!
while diff > 10**-4:

    p_list = np.zeros([len(sample),len(sample.T)])
    for i in range(0,len(sample)):
        for j in range(0,len(sample.T)):
            expo = -(sample[i,0]-mean_list[j])**2 / (2*variance_list[j])
            p_list[i,j] = np.e**(expo) / (((2*np.pi)**(1/2))*((variance_list[j])**(1/2)))
            
    mean_old = np.copy(mean_list)
    variance_old = np.copy(variance_list)
    prior_old = np.copy(prior_list)
    
    priorxp_list = np.zeros([len(sample),len(sample.T)*2])
    priorxp_list[:,0:2] = prior_list*p_list
    priorxp_list[:,2] = np.sum(priorxp_list[:,0:2],axis=1)
    priorxp_list[:,3] = np.sum(priorxp_list[:,0:2],axis=1)
    priorxp_list[:,0:2] = priorxp_list[:,0:2] / priorxp_list[:,2:4] 
    
    
    prior_list = np.sum(priorxp_list[:,0:2],axis=0)/len(sample) #updated priors
    
    samplexpriorxp_list = np.copy(sample)
    samplexpriorxp_list[:,0] = sample[:,0]*priorxp_list[:,0]
    samplexpriorxp_list[:,1] = sample[:,0]*priorxp_list[:,1]
    mean_list = np.sum(samplexpriorxp_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means
    
    priorxpxvarsq_list = np.copy(sample)
    priorxpxvarsq_list[:,0] = (sample[:,0]-mean_list[0])**2*priorxp_list[:,0]
    priorxpxvarsq_list[:,1] = (sample[:,0]-mean_list[1])**2*priorxp_list[:,1]
    variance_list = np.sum(priorxpxvarsq_list,axis=0) / np.sum(priorxp_list[:,0:2],axis=0) #updated means
    
    diff = np.array([abs(mean_list-mean_old),abs(variance_list-variance_old),abs(prior_list-prior_old)])
    diff = diff.max()
    print(diff)

#Step 7 Plot

#for i in range (0,len(sample)):
#    if i < len(MANGA):
#        plt.plot(sample[i,0],0, c='g', marker='o')
#    else:
#        plt.plot(sample[i,0],0, c='y', marker='o')
y0 = np.zeros(len(sample))  
plt.plot(sample[0:len(MANGA),0], y0[0:len(MANGA)],'go')
plt.plot(sample[len(MANGA):len(sample),0], y0[len(MANGA):len(sample)],'yo')
    
x1 = np.linspace(.45,.9,50)
x2 = np.linspace(.9,1.2,50)
x3 = np.linspace(-1.5,1.9,50)
x4 = np.linspace(-1.1,2.5,50)
#y1 = stats.norm.pdf(x1, mean_list[0], np.sqrt(variance_list[0]))
#y2 = stats.norm.pdf(x2, mean_list[1], np.sqrt(variance_list[1]))
y3 = stats.norm.pdf(x3, .2, 1)
y4 = stats.norm.pdf(x4, .8, 1)
#plt.plot(x1, y1/max(y1),c='y')
#plt.plot(x2, y2/max(y2),c='g')
plt.plot(x3, y3/max(y3),c='r')
plt.plot(x4, y4/max(y4),c='r')

plt.title("Samples and their respective convexity values + pdfs before EM")
plt.xlabel("Convexity value")
plt.ylabel("Relative pdf value")
plt.legend(["Mango"," Banana"," pdf 1","pdf 2"])
plt.show()



