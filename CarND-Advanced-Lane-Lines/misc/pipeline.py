import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt




class Pipeline():
    def __init__(self,src,dst,mxt,dist,alpha=0.9,debug=False):
        self._left_fit = None
        self._right_fit = None
        self.src = src
        self.dst = dst
        self.mtx = mtx
        self.dist = dist
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.inverse_M = cv2.getPerspectiveTransform(dst,src)
        self.alpha = alpha
        self.s_thresh=(130, 255)
        self.sobel_x_thresh=(45, 250)
        self.dir_thresh=(0.8, 1.2)
        self.debug = debug
        self.debug_array = []


    def _abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        if(len(img.shape)>2):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if(orient == 'x'):
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8( 255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
                # is > thresh_min and < thresh_max
        # 6) Return this mask as your binary_output image
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def _dir_threshold(self,image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately


        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients

        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        direction = np.arctan2(abs_sobel_y, abs_sobel_x)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output



    def _color_gradient_threshold(self,image):
        img = np.copy(image)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]

        sobel_x_binary = self._abs_sobel_thresh(l_channel,'x',thresh=self.sobel_x_thresh)

        dir_binary = self._dir_threshold(img, sobel_kernel=15, thresh=self.dir_thresh)

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) & (dir_binary == 1)] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sobel_x_binary), sobel_x_binary, s_binary))
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sobel_x_binary)
        combined_binary[((s_binary == 1) | (sobel_x_binary == 1))] = 1
        return combined_binary, color_binary

    def _warp(self,image):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_NEAREST)

    # with exponential smoothing Exponential smoothing
    # https://en.wikipedia.org/wiki/Exponential_smoothing
    # alpha = 0.1
    def _fit(self,leftx,lefty,rightx,righty):

        # fit the polynomial to the points ..
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # for the first image no moving average
        if((self._left_fit == None) and (self._right_fit == None)):
            self._left_fit = left_fit
            self._right_fit = right_fit
        else:
            self._left_fit = self._left_fit*self.alpha+left_fit*(1-self.alpha)
            self._right_fit = self._right_fit*self.alpha + right_fit*(1-self.alpha)


        return self._left_fit,self._right_fit
    def _curvature(self,A,B,y):
        return (1+(2*A*y+B)**2)**(1.5)/(np.absolute(2*A))

    def process_image(self,img):
        orig = np.copy(img)

        self.debug_array = []
        self.debug_array.append(orig)
        ## Undistort the image
        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        self.debug_array.append(np.copy(img))

        gradient_image,_ = self._color_gradient_threshold(img)
        binary_warped = self._warp(gradient_image)

        self.debug_array.append(cv2.cvtColor(np.copy(gradient_image),cv2.COLOR_GRAY2RGB))

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
        window_size = np.int(binary_warped.shape[0]/nwindows)
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
            win_y_low = binary_warped.shape[0] - (window+1)*window_size
            win_y_high = binary_warped.shape[0] - window*window_size
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
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
        # Fit a second order polynomial to each

        ### fit the polynomial and moving average
        left_fit,right_fit = self._fit(leftx,lefty,rightx,righty)

        fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
        fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
        yvals = fity
        image  = np.copy(img)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([fit_leftx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, yvals])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        inverse_warp = cv2.warpPerspective(color_warp, self.inverse_M, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(orig, 1, inverse_warp, 0.3, 0)
        y_eval = np.max(yvals)

        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(fity * ym_per_pix, fit_leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(fity * ym_per_pix, fit_rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        # 1+(2*Ay+B)/()
        left_curverad = self._curvature(left_fit_cr[0],left_fit_cr[1],y_eval*ym_per_pix )
        right_curverad = self._curvature(right_fit_cr[0],right_fit_cr[1],y_eval*ym_per_pix ) 
        # radius of curvature is in meters


        screen_middel_pixel = img.shape[1]/2
        left_lane_pixel = fit_leftx[-1]    # x position for left lane
        right_lane_pixel = fit_rightx[-1]   # x position for right lane
        car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
        screen_off_center = screen_middel_pixel-car_middle_pixel
        meters_off_center = np.absolute(xm_per_pix * screen_off_center)

        result_mask_image = np.zeros_like(result)



        mrg = 100
        cv2.rectangle(result, (mrg, 150), (img.shape[1]- mrg, 0),(52,73,94), -1)

        cv2.putText(result_mask_image, "Radius of Curvature left %d (m) , right %d (m)" %(int(left_curverad),int(right_curverad))
                    , (mrg+120,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (61, 35, 71), 3)
        cv2.putText(result_mask_image, "Vehicle is %.2f left of center" % (meters_off_center)
                    , (mrg+280,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (61, 35, 71), 3)


        result = cv2.addWeighted(result, 1, result_mask_image, 0.8, 0)
        if(self.debug):
            self.debug_array.append(result)
            result = np.hstack([cv2.resize(x,None,fx=0.2,fy=0.2) for x in  self.debug_array])
        else:
            result
        return result
