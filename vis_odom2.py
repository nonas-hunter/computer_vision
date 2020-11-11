#!/usr/bin/env python3
"""
An Implementation of the Eight Point Algorithm
Luke Nonas-Hunter
Sam Kaplan

RESOURCES
OpenCV -- Epipolar Geometry:
    https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# get image data
img1 = cv.imread('dataset/sequences/06/image_0/000000.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('dataset/sequences/06/image_0/000010.png',cv.IMREAD_GRAYSCALE)          # trainImage

# initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# match descriptors
matches = bf.match(des1,des2)
# sort them in the order of their distanceimage is
pts2 = []
for i, m in enumerate(matches):
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append(kp1[m.queryIdx].pt)

# make opencv happy by turning our points into 32 bit floats
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# find fundamental matrix using OpenCV
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
print("opencv Fund: "+ str(F))

# we select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


###########################
## Eight-Point Algorithm ##
###########################

# function to normalize points (translate by image centroid and scale by 2/mean distance)
def normalize_points(arr):
    centroid = np.mean(arr, axis=0)
    scaling_factor = 2 / np.mean(np.square(arr), axis=0)
    transformation_matrix = np.matrix([[scaling_factor[0],0,-centroid[0]*scaling_factor[0]],
                                        [0,scaling_factor[1],-centroid[1]*scaling_factor[1]],
                                        [0,0,1]])
    array = np.array([np.append(row, 1) for row in arr])
    return ((arr - centroid) * scaling_factor, transformation_matrix)

# perform normalization
pts1_norm, T = normalize_points(pts1)
pts2_norm, T_prime = normalize_points(pts2)

# collating our 8-point vector
w = []
for p1, p2 in zip(pts1_norm, pts2_norm):
    u, v = p1
    u_prime, v_prime = p2
    temp = [u*u_prime, v*u_prime, u_prime, u*v_prime, v*v_prime, v_prime, u, v, 1] # collating necessary values to perform 8-point algorithm
    w.append(temp)

# need to turn into numpy array for opencv to be happy
w_np = np.array(w)

# perform SVD on 'w' to try and get Fundamental Matrix f (subject to f >= 1 to avoid degenerate scenarios)
u, s, vh = np.linalg.svd(w_np)
print("shape of u: " + str(u.shape))
print("u: " + str(u))
print("shape of s: " + str(s.shape))
print("s: " + str(s))
print("vh: " + str(vh))
print("shape of vh: " + str(vh.shape))

# 9th column is fundamental matrix (could probably automate this for differnent sized arrays in the future)
fund_matrix = vh[:,8].reshape((3,3))

# need to get it to rank 2, so SVD again!
u, s, vh = np.linalg.svd(fund_matrix)
identity = np.identity(3) # identity matrix to add ones to things for correct matrix multplication
sigma = identity*s
sigma[2,2] = 0 # rank 2 Fmatrix

# need to undo normalization so we can get back into a convenient pixel cooridinate system
F1 = np.matmul(np.matmul(u, sigma), vh)
F1 = np.matmul(T_prime.T, np.matmul(F1, T))


# sanity check to see if our Fmatrix is close to 0
a = np.matrix(np.append(pts2[0],1)).T
a_prime = np.matrix(np.append(pts1[0],1)).T
sanity = np.matmul(a_prime.T, np.matmul(F1, a))
print("***Sanity values should be very close to 0***")
for p1, p2 in zip(pts1, pts2):
    a = np.matrix(np.append(p2,1)).T
    a_prime = np.matrix(np.append(p1,1)).T
    sanity = np.matmul(a_prime.T, np.matmul(F1, a))
    print("SANITY: " + str(sanity))

###############################
## Eight-Point Algorithm END ##
###############################



###############################
## TESTING/DRAWING EPILINES  ##
###############################

# find and draw epilines (from OpenCV)
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


# find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# find epilines corresponding to points in left image (first i
# mage) and drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
