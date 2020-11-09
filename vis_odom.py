#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('dataset/sequences/06/image_0/000000.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('dataset/sequences/06/image_0/000010.png',cv.IMREAD_GRAYSCALE)          # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Selecting inlier points -> need to research some more
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# make opencv happy by turning our points into 32 bit floats
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

w = []
for p1, p2 in zip(pts1, pts2):
    u, v = p1
    u_prime, v_prime = p2
    temp = [u*u_prime, v*v_prime, u_prime, u*v_prime, v*v_prime, v_prime, u, v, 1] # collating necessary values to perform 8-point algorithm
    w.append(temp)

w_np = np.array(w) # need to turn into numpy array for opencv to be happy

# perform SVD on 'w' to try and get Fundamental Matrix f (subject to f >= 1 to avoid degenerate scenarios)

u, s, vh = np.linalg.svd(w_np)
print("shape of u: " + str(u.shape))
print("u: " + str(u))
print("shape of s: " + str(s.shape))
print("s: " + str(s))
print("vh: " + str(vh))
print("shape of vh: " + str(vh.shape))

# 9th column is fundamental matrix
fund_matrix = vh[:,9]
print("fund: " + str(fund_matrix))

u, s, vh = np.linalg.svd(fund_matrix)
print("New Fund: " + str(s))



# make opencv happy by turning our points into 32 bit floats
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

# Selecting inlier points -> need to research some more
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# function to find epilines
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# actually finds epilines 
# print("shape of pts1: " + str(np.shape(pts1)))
# print("shape of pts2: " + str(np.shape(pts2)))

# print("shape of pts1 reshaped: " + str(np.shape(pts1.reshape(-1,1,2))))
# print("shape of pts2 reshaped:  " + str(np.shape(pts2.reshape(-1,1,2))))
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()