import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# files = os.listdir("camera_cal/")
# print(files)

def calibration(image):
    nx = 9
    ny = 6
    imagepoints = []
    objectpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    files = os.listdir("camera_cal/")

    for file in files:
        file_name = 'camera_cal/' + file
        gray = cv2.cvtColor(file_name, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imagepoints.append(corners)
            objectpoints.append(objp)

            img = cv2.drawChessboardCorners(file_name, (9,6), corners, ret)

    return img, objectpoints, imagepoints 

def cal_undistort(img, objectpoints,imagepoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagepoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def 



