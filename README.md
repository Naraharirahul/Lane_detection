# Lane Detection

The goal of this project to develop a software pipeline to identify the lane boundaries in a video from a front facing camera on a car. 

### Software Pipeline

1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2) Apply a distortion correction to raw images.
3) Use color transforms, gradients, etc., to create a thresholded binary image.
4) Apply a perspective transform to rectify binary image ("birds-eye view").
5) Detect lane pixels and fit to find the lane boundary.
5) Determine the curvature of the lane and vehicle position with respect to center.
6) Warp the detected lane boundaries back onto the original image.
7) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Result

https://user-images.githubusercontent.com/60045406/117581083-00025500-b0c9-11eb-8bcf-7610a66712c1.mp4


### Challenging conditions


https://user-images.githubusercontent.com/60045406/117581089-0abcea00-b0c9-11eb-97d2-2ea51e1eb523.mp4

