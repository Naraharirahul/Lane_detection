import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

dist_pickle = pickle.load(open("wide_dist_pickle.p","rb"))
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

img = cv2.imread('test_image.png')

def cal_undistort(img,objpoints,imgpoints):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,corners = cv2/findChessBoardCorners(gray,(8,6),None)
    # img = cv2.drawChessboardCorners(img,(8,6),corners,ret)
    
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img.shape[1:],None,None)
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist
Undistorted = cal_undistort(img,objpoints,imgpoints)     
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(Undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()