import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()

# Read in an image
img = plt.imread(images[-1])
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def cam_calibration(img,objpts,imgpts):
    return cv2.calibrateCamera(objpts, imgpts, np.shape(img)[0:2], None, None)

def cal_undistort(img, mtx,dist):
    # Use cv2.calibrateCamera and cv2.undistort()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    return undist

def corners_unwarp(img, nx, ny,mtx,dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist_img = cal_undistort(img, mtx, dist)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray,cmap="gray")
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if(ret):
        # a) draw corners
        margin = 75.0
        undist_img_draw = cv2.drawChessboardCorners(undist_img, (nx, ny), corners, ret)
        #Generic corner identification
        src = np.float32( [
                [corners[nx-1][0][0]     , corners[nx-1][0][1]     ],
                [corners[nx*ny-1][0][0]  , corners[nx*ny-1][0][1]  ],
                [corners[nx*ny-nx][0][0] , corners[nx*ny-nx][0][1] ],
                [corners[0][0][0]        , corners[0][0][1]        ]
                ]  )
        width = undist_img_draw.shape[1]
        height = undist_img_draw.shape[0]
        dst = np.float32([
            [width-margin,    margin        ],
            [width-margin ,   height-margin ],
            [margin       ,   height-margin ],
            [margin       ,   margin        ]


                 ])
        M = cv2.getPerspectiveTransform(src,dst)
        img_size = np.shape(undist_img_draw)[0:2]
        im_s = (img_size[1],img_size[0])
#         print(img_size)
#         print(im_s)
        warped = cv2.warpPerspective(undist_img_draw,M,im_s, flags=cv2.INTER_LINEAR)
    print(np.shape(warped))
    return warped, M
    

def pipeline(img, s_thresh=(20, 100), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    l_channel = hsv[:,:,0]
    s_channel = hsv[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_img = hls_img[:,:,2]
    binary_output = np.zeros_like(s_img)
    # 3) Return a binary image of threshold result
    binary_output[(s_img>thresh[0]) & (s_img<=thresh[1])]=1
    #binary_output = np.copy(img) # placeholder line
    return binary_output

def get_hist(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    return histogram




prev_leftx = []
prev_rightx = []
prev_lefty = []
prev_righty = []
def poly_fit(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
    #     cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
    #     cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]


    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    #     print(lefty)
    global prev_rightx
    global prev_leftx
    global prev_righty
    global prev_lefty
    if(len(leftx)) == 0:
        leftx = prev_leftx
        lefty = prev_lefty
    else:
        prev_leftx = leftx
        prev_lefty = lefty
    if(len(rightx)) == 0:
        rightx = prev_rightx
        righty = prev_righty
    else:
        prev_rightx = rightx
        prev_righty = righty
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]





    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return out_img,left_fit,right_fit




def highlight(img):
    hsvimg = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 120, 120])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 255, 255])

    #0, 0, 200
    #180, 255, 255
    # Threshold the HSV image to get only yellow colors
    maskhsv = cv2.inRange(hsvimg, lower_yellow, upper_yellow)
    maskwhite_hsv = cv2.inRange(hsvimg,lower_white,upper_white)

    # Bitwise-AND mask and original image
    res_yellow = cv2.bitwise_and(img,img, mask= maskhsv)
    res_white = cv2.bitwise_and(img,img, mask= maskwhite_hsv)
    res = cv2.bitwise_or(res_yellow,res_white)
    return res


def process_image_final(img):
#     print(np.shape(img))

    img = cal_undistort(img,mtx,dist)
    img_size = np.shape(img)[0:2]
    M = cv2.getPerspectiveTransform(src,dst)
    im_s = (img_size[1],img_size[0])
    for_hsv_img = cv2.warpPerspective(img,M,im_s, flags=cv2.INTER_LINEAR)


    M_inverse = cv2.getPerspectiveTransform(dst,src)

    #hls_img = #hls_select(highlight(img), thresh=(10, 255))
    hls_img = hls_select(highlight(img),thresh=(80,255))

    warped = cv2.warpPerspective(hls_img,M,im_s, flags=cv2.INTER_LINEAR)
    histogram = get_hist(warped)

    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    out_img2,left_fit,right_fit = poly_fit(warped)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    window_img = np.zeros_like(out_img2)

     # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx -margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
#     plt.imshow(cv2.warpPerspective(for_hsv_img,M_inverse,im_s, flags=cv2.INTER_LINEAR))
#     plt.show()

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    window_img_warp = cv2.warpPerspective(window_img,M_inverse,im_s, flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(img, 1, window_img_warp, 0.3, 0)

    #result =

    return result


src = np.float32([
    [585.0, 460.0],
    [203.0, 720.0],
    [1127, 720],
    [695, 460]
])

dst = np.float32([
    [320, 0],
    [320, 720 ],
    [960, 720 ],
    [960, 0 ]
])
