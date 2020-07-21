import numpy as np
import cv2
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import glob
import os
import pickle
import matplotlib
from tqdm import tqdm
matplotlib.use('QT5Agg')

def undistort(image):
    calibration_file = open("calibration_params.p",'rb')
    params = pickle.load(calibration_file)
    CMtx = params["CameraMatrix"]
    distC = params["DistortionCoeff"]
    undistorted_image = cv2.undistort(image, CMtx, distC, None, CMtx)
    return undistorted_image

def gaussian_blur(img, kernel_size=3):
    blurred_image = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
    return blurred_image

def canny_thresh(img, thresh_low=0, thresh_high=255):
    edges = cv2.Canny(img, thresh_low, thresh_high)
    return edges

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    
    dx=(orient=='x')
    dy=(orient=='y')
    sobel = cv2.Sobel(img,cv2.CV_64F,dx,dy,sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel<=thresh_max)] = 1
    
    return binary_output

def gradient_thresh(image,sobel_kernel=3,x_thresh=(0,255),y_thresh=(0,255)):
    gradx = abs_sobel_thresh(image, 'x', sobel_kernel, thresh=x_thresh)
    grady = abs_sobel_thresh(image, 'y', sobel_kernel, thresh=y_thresh)
    binary_output = np.zeros_like(image)
    binary_output[(gradx==1) & (grady==1)]=1
    return binary_output

def polar_thresh(image,sobel_kernel,mag_thresh=(0,255),ang_thresh=(0,np.pi/2)):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,None,sobel_kernel)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,None,sobel_kernel)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    scaled_mag = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(image)
    binary_output[(scaled_mag >= mag_thresh[0]) & (scaled_mag<=mag_thresh[1]) & (ang >= ang_thresh[0]) & (ang<=ang_thresh[1])] = 1
    binary_output_debug_mag = np.zeros_like(image)
    binary_output_debug_mag[(scaled_mag >= mag_thresh[0]) & (scaled_mag<=mag_thresh[1])]=1
    binary_output_debug_ang = np.zeros_like(image)
    binary_output_debug_ang[(ang >= ang_thresh[0]) & (ang<=ang_thresh[1])]=1

    # plt.figure()
    # plt.imshow(scaled_mag, cmap='gray')
    # plt.figure()
    # plt.imshow(binary_output_debug_mag, cmap='gray')
    # plt.figure()
    # plt.imshow(binary_output_debug_ang, cmap='gray')

    return binary_output

def color_thresh(image,ch=1,thresh=(0,255)):
    #Assume a BGR image is being passed in
    channel = image[:,:,ch] # First Channel    
    binary_output = np.zeros_like(channel)
    binary_output[(channel>thresh[0]) & (channel<=thresh[1])]=1
    return binary_output

def S_thresh(image, thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s>=thresh[0]) & (s<=thresh[1])]=1
    return binary_output

def warp(image):
    transform_file = open("pTransformData.p",'rb')
    params = pickle.load(transform_file)
    M = params["M"]
    warped = cv2.warpPerspective(image,M,image.shape[1::-1])   
    return warped

def unwarp(image):
    transform_file = open("pTransformData.p",'rb')
    params = pickle.load(transform_file)
    Minv = params["Minv"]
    unwarped = cv2.warpPerspective(image,Minv,image.shape[1::-1])    
    return unwarped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin   # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))       
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
     

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad+right_curverad)/2

    #Calculation of deviation from center of lane
    y_eval_cr=y_eval*ym_per_pix
    left_limit = left_fit_cr[0]*(y_eval_cr**2)+left_fit_cr[1]*y_eval_cr+left_fit_cr[2]
    right_limit = right_fit_cr[0]*(y_eval_cr**2)+right_fit_cr[1]*y_eval_cr+right_fit_cr[2] 
    lane_center=(right_limit-left_limit)/2+left_limit
    image_center = (binary_warped.shape[1]/2)*xm_per_pix
    disp = lane_center-image_center


    return out_img,left_fitx,right_fitx,ploty,curvature,disp

def search_around_poly(binary_warped,left_fit,right_fit):
        # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    


    # Fit new polynomials    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad+right_curverad)/2

    #Calculation of deviation from center of lane
    y_eval_cr=y_eval*ym_per_pix
    left_limit = left_fit_cr[0]*(y_eval_cr**2)+left_fit_cr[1]*y_eval_cr+left_fit_cr[2]
    right_limit = right_fit_cr[0]*(y_eval_cr**2)+right_fit_cr[1]*y_eval_cr+right_fit_cr[2] 
    lane_center=(right_limit-left_limit)/2+left_limit
    image_center = (binary_warped.shape[1]/2)*xm_per_pix
    disp = lane_center-image_center
        
    return out_img,left_fitx,right_fitx,ploty,curvature,disp

def process_frame(frame):
    image = frame
    undistorted_image = undistort(image)
    gray = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
    filtered = gaussian_blur(gray,kernel_size=5)
    # polarthresh = polar_thresh(filtered, sobel_kernel=3, mag_thresh=(100, 255), ang_thresh=(np.pi/4,60*np.pi/180))
    # colorthresh = color_thresh(undistorted_image,ch=2,thresh=(200,255))
    gradthresh = abs_sobel_thresh(filtered,orient='x', sobel_kernel=5,thresh=(50,200))
    sthresh = S_thresh(undistorted_image, thresh=(200, 255))
    bin_output = np.zeros_like(gray)
    bin_output[(gradthresh==1) | (sthresh == 1)] = 1
    binary_warped = warp(bin_output)
    out_img,left_fitx,right_fitx,ploty,curvature,disp = fit_polynomial(binary_warped)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))     

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2RGB), 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(unwarp(out_img), 1, result ,.7,0)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (10,30)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(result,"Radius of Curvature = "+str(int(curvature))+ "(m)",
        topLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)   
    topLeftCornerOfText    = (10,60)
    dir = 'left' if disp>0 else 'right'
    cv2.putText(result,"Vehicle is "+str("{:.3f}".format(np.abs(disp)))+ "m "+dir+" of center",
        topLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    return result

def process_video(inputvideo, outputvideo):
    cap = cv2.VideoCapture(inputvideo)
    if (cap.isOpened()== False): 
        assert("Error opening video stream or file")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    out = cv2.VideoWriter(outputvideo, 0x7634706d , fps, (width,height))
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        pbar.update(1)
        ret, frame = cap.read()
        if ret == True:
            result = process_frame(frame)
            out.write(result)
        else: 
            break


    cap.release()
    out.release()
    pbar.close()


plt.close('all')
os.chdir('D:\\Courses\\Udacity\\Workbooks\\ND013\\SelfDrivingCarEngineer\\CarND-Advanced-Lane-Lines')
process_video('project_video.mp4','processed_project_video.mp4') 
process_video('challenge_video.mp4','processed_challenge_video.mp4') 
process_video('harder_challenge_video.mp4','processed_harder_challenge_video.mp4') 



# print(os.listdir())
# test_images = glob.glob('.\\test_images\\*.jpg')  
# subset=[test_images[0]]
# for fname in subset:
#     image = cv2.imread(fname)
#     print(image.shape)
#     undistorted_image = undistort(image)
#     gray = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
#     filtered = gaussian_blur(gray,kernel_size=5)
#     # polarthresh = polar_thresh(filtered, sobel_kernel=3, mag_thresh=(100, 255), ang_thresh=(np.pi/4,60*np.pi/180))
#     # colorthresh = color_thresh(undistorted_image,ch=2,thresh=(200,255))
#     gradthresh = abs_sobel_thresh(filtered,orient='x', sobel_kernel=5,thresh=(50,200))
#     sthresh = S_thresh(undistorted_image, thresh=(200, 255))
#     bin_output = np.zeros_like(gray)
#     bin_output[(gradthresh==1) | (sthresh == 1)] = 1
#     binary_warped = warp(bin_output)
#     out_img,left_fitx,right_fitx,ploty,curvature,disp = fit_polynomial(binary_warped)
#     # Create an image to draw the lines on
#     warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
#     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#     pts = np.hstack((pts_left, pts_right))


     

#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    

#     # Warp the blank back to original image space using inverse perspective matrix (Minv)
#     newwarp = unwarp(color_warp) 
    
#     # Combine the result with the original image
#     result = cv2.addWeighted(cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2RGB), 1, newwarp, 0.3, 0)
#     result = cv2.addWeighted(unwarp(out_img), 1, result ,.7,0)
    


#     f,((ax1, ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4, 2, figsize=(20,20))
#     ax1.imshow(image)
#     ax1.set_title('Original Image', fontsize=10)
#     ax2.imshow(undistorted_image)
#     ax2.set_title('Undistorted Image', fontsize=10)
#     ax3.imshow(filtered, cmap = 'gray')
#     ax3.set_title('Filtered Image', fontsize=10)
#     ax4.imshow(sthresh, cmap='gray')
#     ax4.set_title('Saturation Thresholded Image', fontsize=10)
#     ax5.imshow(gradthresh, cmap='gray')
#     ax5.set_title('Polar Thresholded Image', fontsize=10)    
#     ax6.imshow(gradthresh, cmap = 'gray')
#     ax6.set_title('Color Thresholded Image', fontsize=10)
#     ax7.imshow(bin_output,cmap='gray')
#     ax7.set_title('Binary Image', fontsize=10)
#     ax8.imshow(result)
#     # ax8.imshow(out_img)
#     # ax8.plot(left_fitx,ploty,color='yellow')
#     # ax8.plot(right_fitx,ploty,color='violet')    
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#     f.tight_layout()

#     plt.figure()
#     font                   = cv2.FONT_HERSHEY_SIMPLEX
#     topLeftCornerOfText    = (10,30)
#     fontScale              = 1
#     fontColor              = (255,255,255)
#     lineType               = 2
#     cv2.putText(result,"Radius of Curvature = "+str(int(curvature))+ "(m)",
#         topLeftCornerOfText,
#         font,
#         fontScale,
#         fontColor,
#         lineType)   
#     topLeftCornerOfText    = (10,60)
#     dir = 'left' if disp>0 else 'right'
#     cv2.putText(result,"Vehicle is "+str("{:.3f}".format(np.abs(disp)))+ "m "+dir+" of center",
#         topLeftCornerOfText,
#         font,
#         fontScale,
#         fontColor,
#         lineType)
#     plt.imshow(result)




# plt.show()

