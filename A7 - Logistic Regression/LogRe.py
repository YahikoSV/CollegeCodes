# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:56:17 2019

@author: Potatsu
"""

import cv2
import os
import glob
import time


import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb


t = time.time()
'''
img_dir = "" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
'''

#Upload Mango Collabs

Mango_y=cv2.imread("Mangocollab1.png")
Mango_g=cv2.imread("Mangocollab2.png")
Mango_t=cv2.imread("Mangocollab3.png")



#Matrix Used    
M_collab = Mango_t
plt.imshow(M_collab[...,::-1])
plt.show()
'''
############Getting the region (SCIKIT) #####################################
# apply threshold
Data_gray = cv2.cvtColor(M_collab, cv2.COLOR_BGR2GRAY)
threshold = 240
image = Data_gray
bw = image < threshold

#remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay[...,::-1])
plt.show()

props = regionprops(label_image)  #array of region properties

# Filter some wrong bounding boxes
props_filtered = []
center = []

for i in range (0,len(props)):
    if props[i].filled_area > 500:
        props_filtered.append(props[i])       
        dott = props[i].centroid
        center.append(dott)
        plt.plot(dott[1], dott[0], "bo")

# Mean color
Red_Ave = np.zeros([len(props_filtered)])  
Blue_Ave = np.zeros([len(props_filtered)])  
Green_Ave = np.zeros([len(props_filtered)])  
for i in range (0,len(props_filtered)):
    yx = props_filtered[i].coords
    Redness  = np.zeros([len(yx)])  
    Greeness = np.zeros([len(yx)])  
    Blueness = np.zeros([len(yx)])  
    for j in range(0,len(yx)):
        Redness[j]  = M_collab[yx[j,0],yx[j,1],0]
        Greeness[j] = M_collab[yx[j,0],yx[j,1],1]
        Blueness[j] = M_collab[yx[j,0],yx[j,1],2]
    
    Red_Ave[i] = np.mean(Redness)/255
    Blue_Ave[i] = np.mean(Blueness)/255
    Green_Ave[i] = np.mean(Greeness)/255

Result = np.ones([len(props_filtered),4])
Result[:,1], Result[:,2], Result[:,3] = Red_Ave, Green_Ave, Blue_Ave,

#np.savetxt('MANGA_TRANSITION.txt',Result,fmt='%.2f')   
'''


#LOAD FFING MATRIXES
yellow     = np.loadtxt("MANGA_YELLOW.txt") 
green      = np.loadtxt("MANGA_GREEN.txt") 
transition = np.loadtxt("MANGA_TRANSITION.txt")  
print('time to calc everything:', time.time()-t)

#ONE SAMPLE
length1 = len(yellow)
length2 = len(yellow)+len(green)
length3 = len(yellow)+len(green)+len(transition)
sample = np.ones([length2,5])   #1. x0,x1,x2,x3,d

sample[0:length1,0:4] = yellow[:,0:4]  #d is already 1

sample[length1:length2,0:4] = green[:,0:4]
sample[length1:length2,4] = 0  #d is 0

#sample[length2:length3,0:4] = transition[:,0:4]
#sample[length2:length2+6,5] = 1
#sample[length2+7,5] = .8
#sample[length2+8,5] = .6
#sample[length2+9,5] = .4
#sample[length2+10,5] = 0
#sample[length2+11,5] = .6
#sample[length2+12,5] = 
#sample[length2+13,5] = .6
#sample[length2+14,5] = .4

#Variables
learn = 1.5
iterations = 20

#INITIALIZING WEIGHTS
count = 0
error_f = np.ones([4])
w0 = 0.1
w1 = 0.1  
w2 = 0.1
w3 = 0.1

while sum(error_f) > .01:
#for m in range(0,iterations):
    count += 1
    a = np.zeros([len(sample)])
    z = np.zeros([len(sample)])
    delta_w = np.zeros([4])
    error, error_f = np.zeros([4]), np.zeros([4])
    
    for i in range (0,len(sample)):
        a[i] = w0*sample[i,0] + w1*sample[i,1] + w2*sample[i,2] + w3*sample[i,3] 
        z[i] = 1 / (1 + np.e**(-a[i]))
        
    for k in range(0,4):
        for j in range(0,len(sample)):
            delta_w[k] += learn*(sample[j,4]-z[j])*sample[j,k]
            error[k] += (sample[j,4]-z[j])**2
            
    error_f = np.sqrt(error)
    #print(error_f)
    
    w0, w1, w2, w3 = round(w0+delta_w[0],4), round(w1+delta_w[1],4), round(w2+delta_w[2],4), round(w3+delta_w[3],4)

#test SAMPLES
test = transition
a_t = np.zeros([len(test)])
z_t = np.zeros([len(test)])
for i in range (0,len(test)):
    a_t[i] = w0*test[i,0] + w1*test[i,1] + w2*test[i,2] + w3*test[i,3] 
    z_t[i] = 1 / (1 + np.e**(-a_t[i]))
    
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(green[:,1],green[:,2],green[:,3], c='g', marker='o')
ax.scatter(yellow[:,1],yellow[:,2],yellow[:,3], c='y', marker='o')

ax.legend(['Unripe','Ripe'])
ax.set_xlabel('Red pixel value')
ax.set_ylabel('Green pixel value')
ax.set_zlabel('Blue pixel value')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
plt.show()

        