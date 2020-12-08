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
Length1 = len(BANAN)+len(MANGA)
Length2 = len(BANAN)
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
sample[0:Length2,2:4] = BANAN[:,2:4]
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

plt.plot(sample[3,2],sample[3,3], 'bo')
plt.plot(sample[14,2],sample[14,3], 'bo')

#plt.legend(['Mango','Orange','Banana','O-B Line','O-M Line','B-M Line'])
plt.legend(['mango','orange','m-o Line'])
plt.xlabel('eccentricity')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
   

#STEP 1 of SVM# getting the H vector
G = np.matmul(sample[:,2:4],sample[:,2:4].T)
#Z = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,
#               -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]).T  #for some reason i cant matmul properly
#Z = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,
#               -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,]]).T 
Z = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
               -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,]]).T 
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


'''
#Step 3 Quadprog
from cvxopt import matrix, solvers
H_ms = matrix(H_ms)
f_v = matrix(f_v)
A_m = matrix(A_m)
a_v = matrix(a_v)
B_d = matrix(B_m)
b_v = matrix(b_v)
alpha = solvers.qp(H_ms,f_v,A_m,a_v)
sol = np.array(alpha['x'])
'''

#Find 4 w daw
for i in range(0,len(sample)):
    if sol_a[i] < 0.0001:
        sol_a[i] = 0

w_total = np.zeros([2])
for i in range (0,2):
    for j in range (0,len(sample)):
        w_total[i] += sol_a[j]*Z[j]*sample[j,i+2]
        
w0_total = 1/Z[12] - np.matmul(w_total.T,sample[12,2:4])


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
#plt.plot(MANGA[:,2],MANGA[:,3], 'go')
#plt.plot(ORANG[:,2],ORANG[:,3], 'o', color='#FFA500' )
plt.plot(BANAN[:,2],BANAN[:,3], 'yo')
plt.plot(MANGA[:,2],MANGA[:,3], 'go')
plt.plot(x,y)
#plt.plot(x,y_OM, 'r')




#plt.plot(sample[3,2],sample[3,3], 'bo')
plt.plot(sample[12,2],sample[12,3], 'bo')
#plt.plot(sample[22,2],sample[22,3], 'bo')
plt.plot(sample[25,2],sample[25,3], 'bo')


#plt.legend(['Mango','Orange','Banana','O-B Line','O-M Line','B-M Line'])
plt.legend(['banana','mango','b-m Line'])
plt.xlabel('eccentricity')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
  
