import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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

            # img = cv2.drawChessboardCorners(image, (9,6), corners, ret)

    # return img, objectpoints, imagepoints
    return objectpoints, imagepoints

def cal_undist(image, objectpoints, imagepoints):
    img = mpimg.imread(image)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    return undist

def perspective(undist):
    offset = 200
    img_size = (undist.shape[1], undist.shape[0])
    # src = np.float32([(185,720), (593, 446), (673, 446), (1094, 720)] )
    src = np.float32([(580,460), (203,720), (1207,720), (705,460)])
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0]- offset, 0], [img_size[0] - offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size, undist.shape[1::-1], flags=cv2.INTER_LINEAR)

    return warped, M_inv, M

    
def threshold(warped):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # grad_thesh = np.sqrt(sobelx**2, sobely**2)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_thesh = np.absolute(sobelx)
    # scale_fac = np.max(grad_thesh)/255
    grad_thesh = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gray)
    binary_output[(grad_thesh >= 30) & (grad_thesh <= 255)] = 1

    hls_image = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:,:,2]
    l_channel = hls_image[:,:,1]
    h_channel = hls_image[:,:,0]
    hls_output = np.zeros_like(s_channel)
    hls_output[(s_channel > 60) & (s_channel <=255) ] = 1

    hls_output_1 = np.zeros_like(s_channel)
    hls_output_1[(s_channel > 60) & (s_channel <=255) & (l_channel > 0) & (l_channel > 140) & (h_channel > 20) & (h_channel > 60)] = 1


    # plt.imshow(hls_output, cmap='gray')
    # plt.show()
    rgb_image = warped
    r_channel = rgb_image[:,:,0]
    g_channel = rgb_image[:,:,1]
    b_channel = rgb_image[:,:,2]

    rgb_output = np.zeros_like(r_channel)
    rgb_output[(30 < r_channel) & (30 < g_channel) & (30 < b_channel)] = 1

    rgb_output_1 = np.zeros_like(r_channel)
    rgb_output_1[(r_channel > 200) & (g_channel > 200) & (b_channel > 200)] = 1

    hls_rgb_binary = cv2.bitwise_or(hls_output,rgb_output) #, rgb_output_1)
    # plt.imshow(hls_rgb_binary, cmap='gray')
    # plt.show()
    binary = cv2.bitwise_or(binary_output, hls_output,rgb_output_1,rgb_output)
    # binary = binary_output | hls_output | rgb_output_1 | rgb_output | hls_output_1
    plt.imshow(binary, cmap='gray')
    plt.show()
    return binary

def histogram(img):
    bottom_half = img[img.shape[0] //2 :,:]
    sum_hist = np.sum(bottom_half,axis=0)
    out_img = np.dstack((img, img, img))
    window_img = np.zeros_like(out_img)
    midpoint = np.int(sum_hist.shape[0]//2)
    leftx_base = np.argmax(sum_hist[:midpoint])
    rightx_base = np.argmax(sum_hist[midpoint:]) + midpoint
    
    nwindows = 9
    margin = 100
    minipix = 50

    window_height = np.int(img.shape[0]//nwindows)
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - (window)*window_height
        win_x_left_low = leftx_current - margin
        win_x_left_high = leftx_current + margin
        win_x_right_low = rightx_current - margin
        win_x_right_high = rightx_current + margin

        # cv2.rectangle(out_img,(win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0,255,0), 2)
        # cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_left_low) & (nonzerox <= win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_right_low) & (nonzerox <= win_x_right_high)).nonzero()[0]


        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minipix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if(len(good_right_inds)) > minipix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass    

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Converting from pixel data to real-world data

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # print(left_fit)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Measuring the curvature of the road

    y_eval = np.max(ploty)
    left_curve = (( 1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])

    right_curve = (( 1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    print(" Left and right radius of curvature", left_curve, right_curve)

    return result

# def process_image(image):
#     imshape = image.shape
#     # print("shape",imshape)
#     copy_image = np.copy(image)
#     gray_image = grayscale(copy_image)
#     blur_image = gaussian_blur(gray_image,5)
#     canny_image = canny(blur_image,50,150)
#     vertices = np.array([[(200,660),(600, 450), (750, 450), (1200,660)]], dtype=np.int32)
#     image_roi = region_of_interest(canny_image,vertices)
#     line_image = hough_lines(image_roi,1,np.pi/180,10,40,30)
#     result = weighted_img(line_image,copy_image) 
#     # plt.imshow(line_image)
#     # plt.show()
#     # NOTE: The output you return should be a color image (3 channel) for processing video below
#     # TODO: put your pipeline here,
#     # you should return the final output (image where lines are drawn on lanes)

#     return result

# challenge_output = 'test_videos_output/challenge.mp4'
# # clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,0.25)
# clip3 = VideoFileClip('test_videos/challenge.mp4')
# challenge_clip = clip3.fl_image(process_image)
# %time challenge_clip.write_videofile(challenge_output, audio=False)


files = os.listdir("camera_cal/")
objectpoints, imagepoints = calibration(files)


test_images_dir = os.listdir("test_images/")
for image_dir in test_images_dir:
    image = 'test_images/' + image_dir
    undist = cal_undist(image, objectpoints, imagepoints)
    warped, M_inv, M = perspective(undist)
    binary = threshold(warped)
    # out_img = histogram(binary)
    # plt.imshow(binary, cmap='gray')
    # plt.show()





