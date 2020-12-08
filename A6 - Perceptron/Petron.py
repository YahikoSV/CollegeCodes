# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:30:57 2019

@author: Potatsu
"""

import numpy as np;
import matplotlib.pyplot as plt

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 

#Recall colums 1.) r-cor 2. g-cor 3. eccentricity 4. convexity
#I will use Oranges and Bananas 

#Variables
Length1 = len(ORANG)+len(MANGA)
Length2 = len(ORANG)
learn = .01
iterations = 100

#INITIALIZING WEIGHTS
np.random.seed(32)
count = 0
error_f = np.ones([4])
w1 = .85  #this is w0
w2 = .37
w3 = .55
'''
#Sample Matrices
sample_orange = np.ones([len(ORANG),3])
sample_banana = np.ones([len(BANAN),3])
sample_orange[:,1:3] = ORANG[:,2:4]
sample_banana[:,1:3] = BANAN[:,2:4]
'''
#ONE SAMPLE
sample = np.ones([Length1,4])
sample[Length2:Length1,0] = -1
sample[0:Length2,2:4] = ORANG[:,2:4]
sample[Length2:Length1,2:4] = MANGA[:,2:4]
'''
#CALCULATE THE WEIGHT SUM
a = np.zeros([len(sample)])
for i in range (0,len(sample)):
    a[i] = w1*sample[i,1] + w2*sample[i,2] + w3*sample[i,3]
    
#CALCULATE THE PERCEPTRON FUNCTION (STEP-FXN)
z = np.zeros([len(sample)])
for i in range (0,len(sample)):
    if a[i] >= 0:
        z[i] = 1
    else:
        z[i] = -1


#Weight Change
delta_w = np.zeros([4])
for i in range(1,4):
    for j in range(0,len(sample)):
        delta_w[i] += learn*(sample[j,0]-z[j])*sample[j,i]
'''        
#NOW WE PUT THEM ALL TOGETHER NOW!!!
        
while sum(error_f) > .01:
    
    count += 1
    a = np.zeros([len(sample)])
    z = np.zeros([len(sample)])
    delta_w = np.zeros([4])
    error, error_f = np.zeros([4]), np.zeros([4])
    
    for i in range (0,len(sample)):
        a[i] = w1*sample[i,1] + w2*sample[i,2] + w3*sample[i,3] 
        if a[i] >= 0:
            z[i] = 1
        else:
            z[i] = -1
    for k in range(1,4):
        for j in range(0,len(sample)):
            delta_w[k] += learn*(sample[j,0]-z[j])*sample[j,k]
            error[k] += (sample[j,0]-z[j])**2
            
    error_f = np.sqrt(error)
    #print(error_f)
    
    w1, w2, w3 = round(w1+delta_w[1],4), round(w2+delta_w[2],4), round(w3+delta_w[3],4)
    
#MAKE A FREAKING LINE GRAPH
A = w2
B = w3
C = -w1
x = np.linspace(0,10,100)
y = (C/B) - (A/B)*x

#4. Eccentricity vs Convexity
plt.figure(figsize=(5,5))
plt.plot(MANGA[:,2],MANGA[:,3], 'go')
plt.plot(ORANG[:,2],ORANG[:,3], 'o', color='#FFA500' )
#plt.plot(BANAN[:,2],BANAN[:,3], 'yo')
#plt.plot(x,y_OB)
#plt.plot(x,y_OM, 'r')
plt.plot(x,y,'k')

#plt.legend(['Mango','Orange','Banana','O-B Line','O-M Line','B-M Line'])
plt.legend(['mango','orange','m-o Line'])
plt.xlabel('eccentricity')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
   

