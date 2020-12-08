# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:55:07 2019

@author: Potatsu
"""

import numpy as np;
import matplotlib.pyplot as plt
import math
np.random.seed(34)


#1. create a test variable
N = 13
x = np.linspace(0,2*np.pi,N)
y = np.sin(x)


#2 sampe weight vector and x-matrix  
w_no = 20
w_set = np.zeros([w_no])
w_set[:] = 0.01


x_set = np.zeros([N,w_no])
for i in range (0, N):
    for j in range(0, w_no):
        x_set[i,j] = x[i]**j
        
#3 Normalize the x-matrix (3 options sigmoid,tanh,rel max)
        
nx_set = np.zeros([N,w_no])
for i in range (0, N):
    for j in range(0, w_no):
        nx_set[i,j] = 1/(1 + np.exp(-x_set[i,j]))

nx2_set = np.zeros([N,w_no])
for i in range (0, N):
    for j in range(0, w_no):
        nx2_set[i,j] = np.tanh(x_set[i,j])

nx3_set = np.zeros([N,w_no])
for i in range (0, N):
    for j in range(0, w_no):
        nx3_set[i,j] = x_set[i,j]/np.max(x_set[:,j])
        
#4 desired value
y_taylor     = np.zeros([N])
taylor_mat   = np.zeros([N,w_no])
for i in range(0,N):
    for j in range(0,w_no):
        if j % 4 == 1:
            y_taylor[i]     += x[i]**j/(math.factorial(j))
            taylor_mat[i,j] = x[i]**j/(math.factorial(j)) 
        if j % 4 == 3:
            y_taylor[i]     -= x[i]**j/(math.factorial(j))
            taylor_mat[i,j] = x[i]**j/(math.factorial(j)) 
        
taylor_t_mat = np.tanh(taylor_mat)  
d2 = taylor_t_mat.sum(axis=1)  

#5 Weight sum
a = np.zeros([N])
for i in range (0,N):
        for j in range(0, w_no):
            a[i] += w_set[j]*nx3_set[i,j]

z = np.tanh(a)
d = np.tanh(d2)
        
#6 Calculate error
learn = .05
delta_w = np.zeros([w_no])
for i in range(0,w_no):
    for j in range(0,N):
        delta_w[i] += learn*(d[j]-z[j])*nx3_set[j,i]

#7 Update w
w_new = w_set + delta_w

#8 Calculate error 
error1 = d-z
error = (d-z)**2
error_tot = np.sum(error)/2
print(error_tot)
                
#Redo-all steps (5-8)
iterations = 10000
max_error =  0.00001
count = 1 

while error_tot > max_error and count < iterations:
    a = np.zeros([N])
    for i in range (0,N):
        for j in range(0, w_no):
            a[i] += w_new[j]*nx3_set[i,j]
    
    z = np.tanh(a)

    delta_w = np.zeros([w_no])
    for i in range(0,w_no):
        for j in range(0,N):
            delta_w[i] += learn*(d[j]-z[j])*nx3_set[j,i]

    w_new = w_new + delta_w

    error1 = d-z
    error = (d-z)**2
    error_tot = np.sum(error)/2
    print(error_tot,count)      
    count += 1
    
    
# Correction/Verify
y_res = np.zeros([N])   
for i in range (0,N):
    for j in range(0, w_no):
        y_res[i] += x[i]*w_new[j]

#Taylor series

y_taylor     = np.zeros([N])
taylor_mat   = np.zeros([N,w_no])
for i in range(0,N):
    for j in range(0,w_no):
        if j % 4 == 1:
            y_taylor[i]     += x[i]**j/(math.factorial(j))
            taylor_mat[i,j] = x[i]**j/(math.factorial(j)) 
        if j % 4 == 3:
            y_taylor[i] -= x[i]**j/(math.factorial(j))
            taylor_mat[i,j] = x[i]**j/(math.factorial(j)) 
            
        
plt.plot(x,y,'bo')
plt.plot(x,y,'b')
plt.plot(x,y_res,'ro')
plt.plot(x,y_res,'r')
#plt.plot(x,y_taylor,'ko')
#plt.plot(x,y_taylor,'k')        
        