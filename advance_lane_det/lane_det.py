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
    # img = mpimg.imread(image)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, image.shape[1:], None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)    
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
    warped_1 = cv2.warpPerspective(undist, M_inv, img_size, undist.shape[1::-1], flags=cv2.INTER_LINEAR)
    return warped, warped_1, M_inv

    
def threshold(warped):
    gray = cv2.cvtColor(warped,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # grad_thesh = np.sqrt(sobelx**2, sobely**2)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_thesh = np.absolute(sobelx)
    scale_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_thesh = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gray)
    binary_output[(grad_thesh >= 30) & (grad_thesh <= 255)] = 1

    x_binary = np.zeros_like(scale_sobel)
    x_binary[(scale_sobel >=20) & (scale_sobel <=100)] = 1


    hls_image = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:,:,2]
    l_channel = hls_image[:,:,1]
    h_channel = hls_image[:,:,0]
    hls_output = np.zeros_like(s_channel)
    hls_output[(s_channel > 170) & (s_channel <=255) ] = 1

    # hls_output_1 = np.zeros_like(s_channel)
    # hls_output_1[(s_channel > 60) & (s_channel <=255) & (l_channel > 0) & (l_channel > 140) & (h_channel > 20) & (h_channel > 60)] = 1

    gray_binary = np.zeros_like(gray)
    gray_binary[(gray > 180) & (gray <=255)] = 1

    rgb_image = warped
    r_channel = rgb_image[:,:,0]
    g_channel = rgb_image[:,:,1]
    b_channel = rgb_image[:,:,2]

    rgb_output = np.zeros_like(r_channel)
    rgb_output[( 230 < r_channel) & (255 >= r_channel) ] = 1

    binary = cv2.bitwise_or(x_binary,hls_output, rgb_output) #, rgb_output_1)
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

    # ym_per_pix = 30/720
    # xm_per_pix = 3.7/700
    ym_per_pix = 1
    xm_per_pix = 1

    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    # ploty = np.linspace(0,719,num=720)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

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

    y_eval = np.max(ploty)
    left_curve = (( 1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])

    right_curve = (( 1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return result,left_fitx, right_fitx, ploty

def lane_vis(image, warped_1, M_inv, left_fitx,right_fitx, ploty):

    # img = mpimg.imread(image)
    warp_zero = np.zeros_like(warped_1).astype(np.uint8)
    # warp_zero = np.zeros_like(params['binary_warped']).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # pts = np.hstack((pts_left, pts_right))

    left_point = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_point = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    points = np.hstack((left_point,right_point))
    cv2.fillPoly(color_warp, np.int_([points]), (0,255, 0))
    # Draw the lane onto the warped blank image
    # cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


    # cv2.fillPoly(color_warp, np.int_([points]), (0,255,0))
    newwarp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0])) 
    
    out_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return out_img

files = os.listdir("camera_cal/")
objectpoints, imagepoints = calibration(files)

def process_image(image):
    # imshape = image.shape
    # print("shape",imshape)
    undist = cal_undist(image, objectpoints, imagepoints)
    warped,warped_1, M_inv = perspective(undist)
    binary = threshold(warped)
    result,left_fitx, right_fitx, ploty = histogram(binary)
    out_img = lane_vis(image, binary, M_inv, left_fitx,right_fitx, ploty)
    plt.imshow(out_img)
    plt.show()
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

#     return result

# challenge_output = 'test_videos_output/challenge.mp4'

challenge_output =  'project_video.mp4'
clip3 = VideoFileClip('project_video.mp4').subclip(0,0.25)
# clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
# %time challenge_clip.write_videofile(challenge_output, audio=False)


# files = os.listdir("camera_cal/")
# objectpoints, imagepoints = calibration(files)


# test_images_dir = os.listdir("test_images/")
# for image_dir in test_images_dir:
#     image = 'test_images/' + image_dir
#     undist = cal_undist(image, objectpoints, imagepoints)
#     warped,warped_1, M_inv = perspective(undist)
#     binary = threshold(warped)
#     result,left_fitx, right_fitx, ploty = histogram(binary)
#     out_img = lane_vis(image, binary, M_inv, left_fitx,right_fitx, ploty)
    # plt.imshow(out_img, cmap='gray')
    # plt.show()





