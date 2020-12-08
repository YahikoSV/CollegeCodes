# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:36:06 2019

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

#2 Add random noise ony
var = .2
y_n = np.copy(y)
for i in range (0,N):
    rand_num = 2*np.random.rand()-1
    y_n[i] = y[i] + rand_num*var

#3 sampe matrix  
w_no = 20
w_set = np.zeros([w_no])
w_set[:] = 0.0000


x_set = np.zeros([N,w_no])
for i in range (0, N):
    for j in range(0, w_no):
        x_set[i,j] = x[i]**j
        
        
#4 Weight sum
a = np.zeros([N])
for i in range (0,N):
        for j in range(0, w_no):
            a[i] += w_set[j]*x_set[i,j]
z = np.tanh(a)
            
#5 Desired value
d2 = np.copy(y)
d = np.tanh(d2)

#6 Calculate error
learn = .1
delta_w = np.zeros([w_no])
for i in range(0,w_no):
    for j in range(0,N):
        delta_w[i] += learn*(d[j]-z[j])*x[j]
        
#7 Update w
w_new = w_set + delta_w

#8 Calculate error 
error1 = d-z
error = (d-z)**2
error_tot = np.sum(error)/2
print(error_tot)
        

#Redo-all steps (4-8)
iterations = 000
max_error =  0.1
count = 0 

while error_tot > max_error and count < iterations:
    a = np.zeros([N])
    for i in range (0,N):
            for j in range(0, w_no):
                a[i] += w_new[j]*x_set[i,j]
    z = np.tanh(a)              
    delta_w = np.zeros([w_no])
    for i in range(0,w_no):
        for j in range(0,N):
            delta_w[i] += learn*(d[j]-z[j])*x[j]
            
    w_new = w_new + delta_w
    
    error = (d-z)**2
    error_tot = np.sum(error)/2
    count += 1
    print(error_tot,count)

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
            
        
#plt.plot(x,y,'bo-')
#plt.plot(x,y_res,'ro-')
plt.plot(x,y_taylor,'ko-')
plt.legend(['20 deg taylor'])