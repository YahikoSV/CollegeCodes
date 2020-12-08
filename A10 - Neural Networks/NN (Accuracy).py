# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:58:35 2019

@author: Potatsu
"""

import numpy as np;
import matplotlib.pyplot as plt

#UPLOAD THE FFING MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
BANAN=np.loadtxt("BANAN.txt") 

#Variables
Length1 = len(BANAN)+len(MANGA)
Length2 = len(BANAN)
learn = .01
iterations = 100

#ONE SAMPLE
sample = np.ones([14,4])
sample[0:8,2:4] = BANAN[0:8,2:4]
sample[8:14,2:4] = MANGA[0:6,2:4]
sample[8:14,0] = -1

test = np.ones([12,4])
test[0:7,2:4] = BANAN[8:15,2:4]
test[7:12,2:4] = MANGA[6:11,2:4]
test[7:12,0] = -1
#STEP 1 of SVM# getting the H vector
G = np.matmul(sample[:,2:4],sample[:,2:4].T)
Z = np.array([[1,1,1,1,1,1,1,1,
               -1,-1,-1,-1,-1,-1,]]).T 
Z1 = np.matmul(Z,Z.T)
H_m = G*Z1

#Step 2 set up f, a, b, A and B
f_v = (np.ones([len(sample)])*-1)
A_m = np.zeros([len(sample),len(sample)])
np.fill_diagonal(A_m,-1)
a_v = np.zeros([len(sample)])
B_m = np.zeros([len(sample),len(sample)])
B_m[0,:] = Z.T
b_v = np.zeros([len(sample)])
H_ms = H_m - A_m*0.001 

#Step 3 Quadprog
from qpsolvers import solve_qp
sol_a = solve_qp(H_ms, f_v, A_m, a_v, B_m, b_v)


#Find 4 w daw
for i in range(0,len(sample)):
    if sol_a[i] < 0.0001:
        sol_a[i] = 0
        
w_total = np.zeros([2])
for i in range (0,2):
    for j in range (0,len(sample)):
        w_total[i] += sol_a[j]*Z[j]*sample[j,i+2]
        
w0_total = 1/Z[3] - np.matmul(w_total.T,sample[3,2:4])


w0_dis = np.zeros([len(sample)])
w0_avg = 0
for i in range(0,len(sample)):
    w0_dis[i] = 1/Z[i] - np.matmul(w_total.T,sample[i,2:4])
    w0_avg += w0_dis[i]
w0_avg = w0_avg/len(sample)

#Step 5 Redo the part
A = w_total[0]
B = w_total[1]
C = -w0_total
x = np.linspace(0,10,100)
y = (C/B) - (A/B)*x

#4. Eccentricity vs Convexity
plt.figure(figsize=(5,5))
plt.plot(BANAN[0:8,2],BANAN[0:8,3], 'yo')
plt.plot(MANGA[0:6,2],MANGA[0:6,3], 'go')
plt.plot(x,y)

plt.plot(sample[3,2],sample[3,3], 'ro')
plt.plot(sample[9,2],sample[9,3], 'ro')
plt.plot(sample[11,2],sample[11,3], 'ro')
#plt.plot(sample[25,2],sample[25,3], 'ro')

plt.legend(['banana','mango','b-m Line'])
plt.xlabel('eccentricity')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


p_w0 = C
p_w1 = A
p_w2 = B

a = np.zeros([len(sample)])
for i in range (0,len(sample)):
    a[i] = p_w0*sample[i,1] + p_w1*sample[i,2] + p_w2*sample[i,3]
b = np.zeros([len(test)])
for i in range (0,len(test)):
    b[i] = p_w0*test[i,1] + p_w1*test[i,2] + p_w2*test[i,3]
    

    

#INITIALIZING WEIGHTS
learn = .001
iterations = 100
np.random.seed(32)
count = 0
error_f = np.ones([4])
p_w0 = .85  #this is w0
p_w1 = .37
p_w2 = .55

#CALCULATE THE WEIGHT SUM
a = np.zeros([len(sample)])
for i in range (0,len(sample)):
    a[i] = p_w0*sample[i,1] + p_w1*sample[i,2] + p_w2*sample[i,3]

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

while sum(error_f) > .01:
    
    count += 1
    a = np.zeros([len(sample)])
    z = np.zeros([len(sample)])
    delta_w = np.zeros([4])
    error, error_f = np.zeros([4]), np.zeros([4])
    
    for i in range (0,len(sample)):
        a[i] = p_w0*sample[i,1] + p_w1*sample[i,2] + p_w2*sample[i,3]
        if a[i] >= 0:
            z[i] = 1
        else:
            z[i] = -1
    for k in range(1,4):
        for j in range(0,len(sample)):
            delta_w[k] += learn*(sample[j,0]-z[j])*sample[j,k]
            error[k] += (sample[j,0]-z[j])**2
            
    error_f = np.sqrt(error)
    print(error_f)
    
    p_w0, p_w1, p_w2 = round(p_w0+delta_w[1],4), round(p_w1+delta_w[2],4), round(p_w2+delta_w[3],4)


a = np.zeros([len(sample)])
for i in range (0,len(sample)):
    a[i] = p_w0*sample[i,1] + p_w1*sample[i,2] + p_w2*sample[i,3]
b = np.zeros([len(test)])
for i in range (0,len(test)):
    b[i] = p_w0*test[i,1] + p_w1*test[i,2] + p_w2*test[i,3]