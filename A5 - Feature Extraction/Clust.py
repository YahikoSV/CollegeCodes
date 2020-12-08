# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:30:23 2019

@author: Potatsu
"""
import cv2
import os
import glob
import numpy as np;
import matplotlib.pyplot as plt
import skimage.measure as ski
import copy

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb


#Import all image collabs in an array
img_dir = "" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data1 = []
for f1 in files:
    img = cv2.imread(f1)
    data1.append(img)
    
data_array = np.array(data1)


#Separate them
Bananas = data_array[0,:,:,:]
Mangos = data_array[1,:,:,:]
Oranges = data_array[2,:,:,:]

#Data Set
Data_Used = Mangos
#Color Class
threshold = 240

Data_gray = cv2.cvtColor(Data_Used, cv2.COLOR_BGR2GRAY)
Data_thresh = np.uint8(Data_gray<threshold)
#plt.imshow(Data_gray<threshold, cmap='gray')
#plt.show()


############Getting the region (SCIKIT) #####################################
# apply threshold
image = Data_gray
bw = image < threshold

#remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay[...,::-1])

props = regionprops(label_image)  #array of region properties

# Filter some wrong bounding boxes
props_filtered = []

for i in range (0,len(props)):
    if props[i].filled_area > 500:
        props_filtered.append(props[i])
        dott = props[i].centroid
        plt.plot(dott[1], dott[0], "bo")



#ECCENTRICITY FIRST THEN B!!!
#ALSO CONVEXITY C5!!!!
a = np.zeros([len(props_filtered),2])
b = np.zeros([len(props_filtered)])
c1 = np.zeros([len(props_filtered)])
c4 = np.zeros([len(props_filtered)])
c5 = np.zeros([len(props_filtered)])
for i in range (0,len(props_filtered)):
       a[i,:] = props_filtered[i].centroid
       b[i] =  props_filtered[i].eccentricity
       c1[i] = props_filtered[i].filled_area
       c4[i] = props_filtered[i].convex_area
       c5[i] = c1[i]/c4[i]


#Finding the color
r_norm = np.zeros([len(props_filtered)]) 
g_norm = np.zeros([len(props_filtered)])
for i in range (0,len(props_filtered)):
    p = int(a[i,1]) 
    q = int(a[i,0]) 
    r_norm[i] = int(Data_Used[q,p,0])/(int(Data_Used[q,p,0]) + int(Data_Used[q,p,1]) + int(Data_Used[q,p,2]))
    g_norm[i] = int(Data_Used[q,p,1])/(int(Data_Used[q,p,0]) + int(Data_Used[q,p,1]) + int(Data_Used[q,p,2]))
    
#ax.set_axis_off()
plt.tight_layout()
plt.show()


#GET ALL THE DATA IN A TABLE
mat = np.zeros([len(props_filtered),4])
mat[:,0] = r_norm
mat[:,1] = g_norm
mat[:,2] = b
mat[:,3] = c5
#np.savetxt('MANGA.txt',mat,fmt='%.2f')


#UPLOAD THE MATRIXES
MANGA=np.loadtxt("MANGA.txt") 
ORANG=np.loadtxt("ORANG.txt") 
BANAN=np.loadtxt("BANAN.txt") 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(MANGA[:,1],MANGA[:,2],MANGA[:,3], c='g', marker='o')
ax.scatter(ORANG[:,1],ORANG[:,2],ORANG[:,3], c='#FFA500', marker='o')
ax.scatter(BANAN[:,1],BANAN[:,2],BANAN[:,3], c='y', marker='o')

ax.set_xlabel('r-chroma')
ax.set_ylabel('eccentricity')
ax.set_zlabel('convexity')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
plt.show()

'''
#MAKE FFING PLOTS
#1. R-Cor vs G-Cor
plt.figure(figsize=(5,5))
plt.plot(MANGA[:,0],MANGA[:,1], 'go')
plt.plot(ORANG[:,0],ORANG[:,1], 'o', color='#FFA500' )
plt.plot(BANAN[:,0],BANAN[:,1], 'yo')

plt.legend(['Mango','Orange','Banana'])
plt.xlabel('r chroma')
plt.ylabel('g chroma')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#2. R-Cor vs Eccentricity
plt.figure(figsize=(5,5))
plt.plot(MANGA[:,0],MANGA[:,2], 'go')
plt.plot(ORANG[:,0],ORANG[:,2], 'o', color='#FFA500' )
plt.plot(BANAN[:,0],BANAN[:,2], 'yo')


plt.legend(['Mango','Orange','Banana'])
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('r chroma')
plt.ylabel('eccentricity')
plt.show()

#3. R-Cor vs Convexity
plt.figure(figsize=(5,5))
plt.plot(MANGA[:,0],MANGA[:,3], 'go')
plt.plot(ORANG[:,0],ORANG[:,3], 'o', color='#FFA500' )
plt.plot(BANAN[:,0],BANAN[:,3], 'yo')

plt.legend(['Mango','Orange','Banana'])
plt.xlabel('r chroma')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#4. Eccentricity vs Convexity
plt.figure(figsize=(5,5))
plt.plot(MANGA[:,2],MANGA[:,3], 'go')
plt.plot(ORANG[:,2],ORANG[:,3], 'o', color='#FFA500' )
plt.plot(BANAN[:,2],BANAN[:,3], 'yo')

plt.legend(['Mango','Orange','Banana'])
plt.xlabel('eccentricity')
plt.ylabel('convexity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
'''


#RESULT AS OF NOW
#I CAN ONLY GET THE COLOR OF MANGOES AND ORAGES AUTOMATICALLY NOT BANANAS YET
