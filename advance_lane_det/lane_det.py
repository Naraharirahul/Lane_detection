import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def calibration(files):
    nx = 9
    ny = 6
    imagepoints = []
    objectpoints = []
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    for file in files:
        file_name = 'camera_cal/' + file
        image = cv2.imread(file_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            imagepoints.append(corners)
            objectpoints.append(objp)

            img = cv2.drawChessboardCorners(image, (9,6), corners, ret)

    # return img, objectpoints, imagepoints
    return objectpoints, imagepoints

def cal_undist(objectpoints, imagepoints):
    test_images_dir = os.listdir("test_images/")
    for image_dir in test_images_dir:
        image = 'test_images/' + image_dir
        img = mpimg.imread(image)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, img.shape[1:], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)    
        return undist

def perspective(undist):
    offset = 200
    img_size = (undist.shape[1], undist.shape[0])
    src = np.float32([(185,720), (593, 446), (673, 446), (1094, 720)] )
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0]- offset, 0], [img_size[0] - offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    return warped, M_inv, M

    
def threshold(warped):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # grad_thesh = np.sqrt(sobelx**2, sobely**2)
    grad_thesh = np.absolute(sobelx)
    scale_fac = np.max(grad_thesh)/255
    grad_thesh = (grad_thesh/scale_fac).astype(np.uint8)

    binary_output = np.zeros_like(grad_thesh)
    binary_output[(grad_thesh >= 30) & (grad_thesh <= 255)] = 1

    hls_image = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:,:,2]
    hls_output = np.zeros_like(s_channel)
    hls_output[(s_channel > 50) & (s_channel <=255)] = 1

    binary = cv2.bitwise_or(binary_output, hls_output)
    return binary

def histogram(img):
    bottom_half = img[img.shape[0] //2 :,:]
    sum_hist = np.sum(bottom_half,axis=0)
    return sum_hist


files = os.listdir("camera_cal/")
objectpoints, imagepoints = calibration(files)
undist = cal_undist(objectpoints, imagepoints)
warped, M_inv, M = perspective(undist)
binary = threshold(warped)
sum = histogram(binary)
# plt.imshow(binary, cmap='gray')
# plt.show()





