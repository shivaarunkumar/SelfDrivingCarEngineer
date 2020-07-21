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

class Lines():
    n = 50    
    def __init__(self):        
        # was the line detected in the last iteration?
        self.detected = (False,False) #(left,right)           
        # Number of fits missed
        self.recent_misses = [0,0] 
        #average x values of the fitted line over the last n iterations
        self.best_fitx = [np.array([]),np.array([])]
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [np.array([]),np.array([])]  
        # x values of the last n fits of the line
        self.recent_xfits = [[],[]] 
        # x values of the last n fits of the line
        self.recent_xfitvals = [[],[]]
        #polynomial coefficients for the most recent fit
        self.current_fit = [None,None]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0 
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [[],[]]   
        #y values for detected line pixels
        self.ally = [[],[]]  
        # Quality of currnt fit 
        self.max_deviation = [0,0]
    def updatebestfit(self):
        lastnleftfits = self.recent_xfits[0][-self.n:]
        lastnleftfitvals = self.recent_xfitvals[0][-self.n:]
        #Valid Values
        vlastnleftfits=[val for val in lastnleftfits if val.any()]
        vlastnleftfitvals = [val for val in lastnleftfitvals if val.any()]
        self.best_fit[0]=np.mean(vlastnleftfits,axis=0) if len(vlastnleftfitvals)!=0 else np.array([])
        self.best_fitx[0]=np.mean(vlastnleftfitvals,axis=0) if len(vlastnleftfitvals)!=0 else np.array([])

        lastnrightfits = self.recent_xfits[1][-self.n:]
        lastnrightfitvals = self.recent_xfitvals[1][-self.n:]
        vlastnrightfits = [val for val in lastnrightfits if val.any()]
        vlastnrightfitvals = [val for val in lastnrightfitvals if val.any()]
        self.best_fit[1]=np.mean(vlastnrightfits,axis=0) if len(vlastnrightfits)!=0 else np.array([])
        self.best_fitx[1]=np.mean(vlastnrightfitvals,axis=0) if len(vlastnrightfitvals)!=0 else np.array([])
        

def converRGBto(image, type='gray'):
    if type=='gray':
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    elif type == 'hsv':
        return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    elif type == 'hls':
        return cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    else:
        return None

def displayAllComponents(image,title='components'):
    ndim = np.ndim(image)
    if(ndim>2):
        f,axs=plt.subplots(1,ndim,figsize=(20,20))
        for dim in range(ndim):
            axs[dim].imshow(image[:,:,dim],cmap='gray')
        f.suptitle(title)
    else:
        return

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

def component_threshold(image,thresh=(0,255),exposethresh = .25):
    binary_image=np.zeros_like(image)
    binary_image[(image>=thresh[0]) & (image<=thresh[1])]=1
    overexposed = (np.count_nonzero(binary_image[binary_image.shape[0]//2:]) / (binary_image.shape[0]//2*binary_image.shape[1]) > exposethresh)
    return binary_image,overexposed

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
    binary_output[(gradx==1) | (grady==1)]=1
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

def white_thresh(image, thresh_low=220):
    binary_output = np.zeros_like(image[:,:,0])
    binary_output[(image[:,:,0]>=thresh_low) & (image[:,:,1]>=thresh_low) & (image[:,:,2]>=thresh_low)]=1
    return binary_output

def S_thresh(image, thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s>=thresh[0]) & (s<=thresh[1])]=1
    return binary_output

def warp(image, trdata = "pTransformData.p"):
    transform_file = open(trdata,'rb')
    params = pickle.load(transform_file)
    M = params["M"]
    src = params["src"]
    dst = params["dst"]
    warped = cv2.warpPerspective(image,M,image.shape[1::-1])  
    scaled_bin = image.copy()*255

    image_wpoly = cv2.polylines(np.dstack((scaled_bin,scaled_bin,scaled_bin)),np.int32([src]),True,(255,0,0),3) 
    warped_wpoly = cv2.polylines(warped.copy(),np.int32([dst]),True,(255,0,0),3) 
    return warped,image_wpoly,warped_wpoly

def unwarp(image, trdata = "pTransformData.p"):
    transform_file = open(trdata,'rb')
    params = pickle.load(transform_file)
    Minv = params["Minv"]
    unwarped = cv2.warpPerspective(image,Minv,image.shape[1::-1])    
    return unwarped

def computeCurvature(fit,value):
    curvature = ((1 + (2*fit[0]*value + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return curvature

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    padding = 100
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[padding:midpoint])+padding
    rightx_base = np.argmax(histogram[midpoint:(histogram.shape[0]-padding)]) + midpoint

    # if(histogram[leftx_base]>histogram[rightx_base]):
    #     rightx_base = np.argmax(histogram[midpoint:leftx_base+600]) + midpoint
    # else:
    #     leftx_base = np.argmax(histogram[:rightx_base-600])


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
        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin   # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
                
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

    return leftx, lefty, rightx, righty

def find_lane_pixels_fromprev(binary_warped,lines):
    
    if((lines.detected[0] | lines.best_fit[0].any()) & (lines.detected[1] | lines.best_fit[1].any())):        # HYPERPARAMETER
    #if(False):
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        left_fit = lines.current_fit[0] if lines.detected[0] else lines.best_fit[0]
        right_fit = lines.current_fit[1] if lines.detected[1] else lines.best_fit[1]
            
        margin = 100
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
    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    return leftx, lefty, rightx, righty

def fit_lane(binary_warped, lines = Lines(),ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    y_eval = binary_warped.shape[0]-1 #Compute Curvature at the base
    
    #leftx, lefty, rightx, righty = find_lane_pixels(binary_warped) 
    leftx, lefty, rightx, righty = find_lane_pixels_fromprev(binary_warped,lines) 
    left_fit,lcov = np.polyfit(lefty, leftx, 2,cov=True)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_fit,rcov = np.polyfit(righty, rightx, 2,cov=True)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    ploty = np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])
    
    # Quality of Fit
    left_fitx = np.polyval(left_fit,ploty)
    right_fitx = np.polyval(right_fit,ploty)
    n = 2 #Degree of fit
    TT = np.vstack([ploty**(n-i) for i in range(n+1)]).T     
    C_left_fitx = np.dot(TT, np.dot(lcov, TT.T)) # C_y = TT*cov*TT.T
    sig_left_fitx = np.sqrt(np.diag(C_left_fitx))    
    C_right_fitx = np.dot(TT, np.dot(rcov, TT.T)) # C_y = TT*cov*TT.T
    sig_right_fitx = np.sqrt(np.diag(C_right_fitx))
    lfit_good = True if np.max(sig_left_fitx)<15 else False
    rfit_good = True if np.max(sig_right_fitx)<15 else False

    # Calculation of R_curve (radius of curvature)
    curvature = (left_curverad*lfit_good+right_curverad*rfit_good)/(lfit_good+rfit_good) if (lfit_good | rfit_good) else 0

    # Calculation of deviation from center of lane
    y_eval_cr=y_eval*ym_per_pix
    left_limit = np.polyval(left_fit_cr,y_eval_cr)
    right_limit = np.polyval(right_fit_cr,y_eval_cr)
    lane_center=(right_limit-left_limit)/2+left_limit
    image_center = (binary_warped.shape[1]/2)*xm_per_pix
    disp = lane_center-image_center

    # Record Keeping
    lines.detected = [lfit_good,rfit_good]
    lines.recent_misses[0] = 0 if lfit_good else lines.recent_misses[0]+1
    lines.recent_misses[1] = 0 if rfit_good else lines.recent_misses[1]+1
    lines.current_fit[0] = left_fit if (lfit_good | (lines.best_fit[0].any()==False)) else lines.best_fit[0]
    lines.current_fit[1] = right_fit if (rfit_good | (lines.best_fit[1].any()==False))else lines.best_fit[1]
    lines.allx[0].append(leftx)
    lines.ally[0].append(lefty) 
    lines.allx[1].append(rightx)
    lines.ally[1].append(righty)
    if lines.recent_xfits[0]==None:
        lines.recent_xfits[0] = [left_fit] if lfit_good else [np.array([])]
        lines.recent_xfitvals[0] = [left_fitx] if lfit_good else [np.array([])]
        lines.recent_xfits[1] = [right_fit] if rfit_good else [np.array([])]
        lines.recent_xfitvals[1] = [right_fitx] if rfit_good else [np.array([])]
    else:
        lines.recent_xfits[0].append(left_fit) if lfit_good else lines.recent_xfits[0].append(np.array([]))
        lines.recent_xfitvals[0].append(left_fitx) if lfit_good else lines.recent_xfitvals[0].append(np.array([]))
        lines.recent_xfits[1].append(right_fit) if rfit_good else lines.recent_xfits[1].append(np.array([]))
        lines.recent_xfitvals[1].append(right_fitx) if rfit_good else lines.recent_xfitvals[1].append(np.array([]))  
    lines.radius_of_curvature = curvature if (lfit_good | rfit_good) else lines.radius_of_curvature
    lines.line_base_pos = disp if (lfit_good & rfit_good) else lines.line_base_pos
    lines.max_deviation = [np.max(sig_left_fitx),np.max(sig_right_fitx)]    
    

    if ((lfit_good==False) & lines.best_fitx[0].any()):
        left_fit = lines.best_fit[0]
        left_fitx = lines.best_fitx[0]
    if ((rfit_good==False) & lines.best_fitx[1].any()):
        right_fit = lines.best_fit[1]
        right_fitx = lines.best_fitx[1]

    lines.updatebestfit()

    curvature = lines.radius_of_curvature
    disp = lines.line_base_pos


    return (leftx, lefty, rightx, righty),(left_fit,right_fit,lcov,rcov),(ploty,left_fitx,right_fitx,sig_left_fitx,sig_right_fitx),curvature,disp,lines

def plotFit(binary_warped,lanepts,fitvals,curvature,disp,lines, trdata='pTransformData.p'):
    leftx, lefty, rightx, righty = lanepts
    ploty,left_fitx,right_fitx,sig_left_fitx,sig_right_fitx = fitvals
    copy_img=np.zeros_like(binary_warped)
    out_img=np.dstack((copy_img,copy_img,copy_img))
    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,0,255]
    plt.figure();plt.imshow(out_img)    
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.fill_betweenx(ploty, left_fitx+sig_left_fitx, left_fitx-sig_left_fitx, color='m',alpha=.9)
    plt.fill_betweenx(ploty, right_fitx+sig_right_fitx, right_fitx-sig_right_fitx, color='m',alpha=.9)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    poly_warp=np.zeros_like(out_img)
    # Draw the lane onto the warped blank image
    cv2.fillPoly(poly_warp, np.int_([pts]), (0,255, 0))
    # Combine the result with the original image
    result = cv2.addWeighted(unwarp(out_img,trdata), 1, unwarp(poly_warp,trdata), 0.3, 0)
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

def process_frame(frame,lines):
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
    #Undistort
    undistorted = undistort(image)
    # plt.figure();plt.imshow(undistorted);plt.suptitle('undistorted image')
    #Filter
    filtered = gaussian_blur(undistorted,kernel_size=5)
    # plt.figure();plt.imshow(filtered);plt.suptitle('filtered image')
    hsv = converRGBto(filtered,'hsv')
    # displayAllComponents(hsv,"HSV Components")
    compThresh,overexposed = component_threshold(hsv[:,:,0],(10,50),.2) 
    # plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Component Threshold') 
    #Saturation Threshold
    if overexposed==True:
        # print("Computing Color")                
        compThresh, overexposed = component_threshold(filtered[:,:,0],(170,255),.1)
        # plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Component Threshold')
        if overexposed == True:
            # print("Computing Value")        
            compThresh,overexposed = component_threshold(hsv[:,:,2],(200,255),.1)
            # plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Value Threshold')
            if overexposed == True:
                # print("Computing Saturation")        
                compThresh,overexposed = component_threshold(hsv[:,:,1],(80,255),.15)
                # plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Component Threshold')
    
    # Convert to Gray
    gray = converRGBto(filtered,'gray')
    #plt.figure();plt.imshow(gray,cmap='gray');plt.suptitle('grayscale image')
    # gradThresh = gradient_thresh(gray, 5,(80,255),(100,255))
    gradThresh = gradient_thresh(gray, 5,(80,255),(150,255))
    #  plt.figure();plt.imshow(gradThresh,cmap='gray');plt.suptitle('Gradient Threshold')


    wThresh = white_thresh(filtered)
    # plt.figure();plt.imshow(wThresh,cmap='gray');plt.suptitle('White Threshold')
    binary_image = np.zeros_like(gray)
    binary_image[(compThresh==1) | (gradThresh==1) | (wThresh == 1)] = 1
    # plt.figure();plt.imshow(binary_image,cmap='gray');plt.suptitle('Combined Threshold : '+str(fidx))

    # Warp Image
    warped,image_wpoly,warped_wpoly =warp(binary_image,trdata='pv_pTransformData2.p')
    # plt.figure();plt.imshow(warped,cmap='gray');plt.suptitle('warped image')
    # plt.figure();plt.imshow(cv2.addWeighted(filtered,1,image_wpoly,.5,0));plt.suptitle('bin image with poly')
    # unwarped =unwarp(warped,trdata='pv_pTransformData2.p')
    # plt.figure();plt.imshow(unwarped,cmap='gray');plt.suptitle('unwarped image')

    
    lanepts,lanefit,fitvals,curvature,disp,lines = fit_lane(warped,lines)
    result =  plotFit(warped,lanepts,fitvals,curvature,disp, lines,trdata='pv_pTransformData2.p')    
    result = cv2.addWeighted(result,1,filtered,.7,0)  
    return result,lines

