# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:00:55 2019

@author: Potatsu
"""

# Standard imports
import cv2
import numpy as np;
import matplotlib.pyplot as plt

# Read image
im = cv2.imread("3468.jpg", cv2.IMREAD_GRAYSCALE)
test = cv2.imread("3468.jpg")

# Show blobs
#plt.imshow(im, cmap='gray', vmin=0, vmax=255)
#plt.show()
thresh_max = 219
thresh_min = -1

im2 = np.zeros([len(im),len(im)])
im3 = np.zeros([len(im),len(im)])

for i in range(0,len(im)): #maximum threshold
    for j in range(0,len(im)):
        if im[i,j] > thresh_max:
            im2[i,j] = 0
        elif im[i,j] > thresh_min:
            im2[i,j] = 1
        else:
            im2[i,j] = 0


            
#plt.imshow(im, cmap='gray')
#plt.show()
#plt.imshow(im2, cmap='gray')
#plt.show()




kernel = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(im2, cv2.MORPH_CLOSE, kernel, iterations = 2)
#plt.imshow(closing, cmap='gray')
#plt.show()

kernel4 = np.zeros((12,12),np.uint8)
kernel4[0,5:7], kernel4[11,5:7]  = 1,1
kernel4[1,4:8], kernel4[10,4:8]  = 1,1
kernel4[2,3:9], kernel4[9,3:9]  = 1,1
kernel4[3,2:10], kernel4[8,2:10]  = 1,1
kernel4[4,1:11], kernel4[7,1:11]  = 1,1
kernel4[5:7,0:12] = 1

kernel3 = np.array(([0,1,1,0],
                    [1,1,1,1],
                    [1,1,1,1],
                    [0,1,1,0]), dtype=np.uint8)

kernel5 = np.array(([0,1,0],
                    [1,1,1],
                    [0,1,0]), dtype=np.uint8)



many = 4
# noise removal
opening = (cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3, iterations = many))
opening2 = np.uint8(opening)
# sure background area
kernel2 = np.ones((3,3),np.uint8)
sure_bg = cv2.dilate(opening2,kernel2,iterations=1)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening2,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,.01*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

#
#plt.imshow(opening2, cmap='gray')
#plt.show()
plt.imshow(sure_fg, cmap='gray')
plt.show()

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
plt.imshow(markers)
plt.show()
markers[unknown==255] = 0
#
#markers = cv2.watershed(im,markers)
#im[markers == -1] = [255,0,0]

plt.imshow(markers)
plt.show()

unique, counts = np.unique(markers, return_counts=True)
a = (np.asarray((unique, counts)).T)
b = np.copy(a)

countduku = 0
for i in range(0,len(a)):
    if a[i,1] > 230:
        b = np.delete(b, i-countduku, 0)
        countduku += 1

mean = np.mean(b[:,1])
std = np.std(b[:,1])

#Next part?
inverted_pic = np.invert(sure_fg)
im_used = inverted_pic
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
#params.minThreshold = 250
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = .95
    
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im_used)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im_used, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
#plt.imshow(im, cmap='gray', vmin=0, vmax=255)
#plt.show()


x, y = np.ones((len(keypoints))), np.ones((len(keypoints)))
for k in range (0,len(keypoints)):
    x[k] = keypoints[k].pt[0] #k is the index of the blob you want to get the position
    y[k] = keypoints[k].pt[1]

plt.imshow(im_with_keypoints, cmap='gray', vmin=0, vmax=255)
plt.plot(x,y,'yo',markersize = 2)
plt.show()


x,y,w,h = cv2.boundingRect(keypoints)
heh = cv2.rectangle(im_used,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(heh, cmap='gray', vmin=0, vmax=255)