
from ldpackage.ldetect import LaneDetector
import os

import glob
import ldpackage.process as fp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib
import pickle
matplotlib.use('QT5Agg')

fPath = os.path.abspath(__file__)
fDir = os.path.dirname(fPath)
os.chdir(fDir)


ld=LaneDetector('project_video.mp4','processed_project_video.mp4')
ld.process()
ld=LaneDetector('challenge_video.mp4','processed_challenge_video.mp4')
ld.process()
ld=LaneDetector('harder_challenge_video.mp4','processed_harder_challenge_video.mp4')
ld.process()



# Save all video Frames
# ld=LaneDetector('project_video.mp4','processed_project_video.mp4')
# ld.saveframes('projectvidframes')
# ld=LaneDetector('challenge_video.mp4','processed_challenge_video.mp4')
# ld.saveframes('challengevidframes')
# ld=LaneDetector('harder_challenge_video.mp4','processed_harder_challenge_video.mp4')
# ld.saveframes('harderchallengevidframes')

# Generate Perspective Transformation Matrix
# fname = '.\\projectvidframes\\frame1240.jpg'
# image = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
# plt.figure();plt.imshow(image);plt.suptitle('original image')
# undistorted = fp.undistort(image)    
# src = np.float32([[578,463],[700,463],[1160,719],[240,719]]) # top left, top right, bottom left, bottom right
# plt.figure()
# plt.imshow(cv2.polylines(undistorted.copy(),np.int32([src]),True,(255,0,0),3))
# plt.suptitle('undistored image')
# dst = np.float32([[400,150],[1000,150],[1000,719],[400,719]]) 
# M = cv2.getPerspectiveTransform(src,dst,cv2.INTER_LINEAR)
# warped = cv2.warpPerspective(undistorted,M,undistorted.shape[1::-1])
# plt.figure()
# plt.imshow(cv2.polylines(warped.copy(),np.int32([dst]),True,(0,0,255),3))
# plt.suptitle('warped image')
# Minv = cv2.getPerspectiveTransform(dst,src,cv2.INTER_LINEAR)
# unwarped = cv2.warpPerspective(warped,Minv,warped.shape[1::-1])
# plt.figure()
# plt.imshow(cv2.polylines(unwarped.copy(),np.int32([src]),True,(0,0,255),3))
# plt.suptitle('unwarped image')

# pTransform={}
# pTransform['M']=M
# pTransform['Minv']=Minv
# pTransform['src'] = src
# pTransform['dst'] = dst
# pickle.dump(pTransform,open('pv_pTransformData2.p','wb'))
# plt.show()


# images = glob.glob('.\projectvidframes\*.jpg')

# idx = range(0,len(images),len(images)//10)
# plt.close('all')
# for fidx in [1240]:
#     image = cv2.cvtColor(cv2.imread(images[fidx]),cv2.COLOR_BGR2RGB)    
#     #plt.figure();plt.imshow(image);plt.suptitle('original image')
#     #Undistort
#     undistorted = fp.undistort(image)    
#     plt.figure();plt.imshow(undistorted);plt.suptitle('undistorted image')
#     #Filter
#     filtered = fp.gaussian_blur(undistorted,kernel_size=5)
#     plt.figure();plt.imshow(filtered);plt.suptitle('filtered image')   

#     # View Channels in different formats
#     # fp.displayAllComponents(filtered,"RGB Components")
#     # hls = fp.converRGBto(filtered,'hls')
#     # fp.displayAllComponents(hls,"HLS Components")
#     hsv = fp.converRGBto(filtered,'hsv')
#     fp.displayAllComponents(hsv,"HSV Components")
#     # Component Threshold
#     compThresh,overexposed = fp.component_threshold(hsv[:,:,0],(10,50),.2)    
#     plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Hue Threshold')
#     if overexposed==True:
#         print("Computing Color")                
#         compThresh, overexposed = fp.component_threshold(filtered[:,:,0],(170,255),.1)
#         plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Color Threshold')
#         if overexposed == True:
#             print("Computing Value")        
#             compThresh,overexposed = fp.component_threshold(hsv[:,:,2],(220,255),.15)
#             plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Value Threshold')
#             if overexposed == True:
#                 print("Computing Saturation")        
#                 compThresh,overexposed = fp.component_threshold(hsv[:,:,1],(80,255),.15)
#                 plt.figure();plt.imshow(compThresh,cmap='gray');plt.suptitle('Saturation Threshold')
    
#     # Convert to Gray
#     gray = fp.converRGBto(filtered,'gray')
#     #plt.figure();plt.imshow(gray,cmap='gray');plt.suptitle('grayscale image')
    
#     # Gradient Thresh
#     # xThresh = fp.abs_sobel_thresh(gray,'x', 5,(80,255))
#     # plt.figure();plt.imshow(xThresh,cmap='gray');plt.suptitle('x Gradient Threshold')

#     # yThresh = fp.abs_sobel_thresh(gray, 'y' ,5,(100,255))
#     # plt.figure();plt.imshow(yThresh,cmap='gray');plt.suptitle('y Gradient Threshold')

#     gradThresh = fp.gradient_thresh(gray, 5,(80,255),(150,255))
#     plt.figure();plt.imshow(gradThresh,cmap='gray');plt.suptitle('Gradient Threshold')


#     wThresh = fp.white_thresh(filtered, thresh_low=180)
#     plt.figure();plt.imshow(wThresh,cmap='gray');plt.suptitle('White Threshold')
    
    
#     binary_image = np.zeros_like(gray)
#     binary_image[(compThresh==1) | (gradThresh==1) | (wThresh == 1)] = 1
#     plt.figure();plt.imshow(binary_image,cmap='gray');plt.suptitle('Combined Threshold : '+str(fidx))

#     # Warp Image
#     warped,image_wpoly,warped_wpoly =fp.warp(binary_image,trdata='pv_pTransformData2.p')
#     plt.figure();plt.imshow(warped,cmap='gray');plt.suptitle('warped image')
#     plt.figure();plt.imshow(cv2.addWeighted(filtered,1,image_wpoly,.5,0));plt.suptitle('bin image with poly')
#     unwarped =fp.unwarp(warped,trdata='pv_pTransformData2.p')
#     plt.figure();plt.imshow(unwarped,cmap='gray');plt.suptitle('unwarped image')

#     lines = fp.Lines()

    
#     lanepts,lanefit,fitvals,curvature,disp,lines = fp.fit_lane(warped,lines)
#     result =  fp.plotFit(warped,lanepts,fitvals,curvature,disp,lines, trdata='pv_pTransformData2.p')    
#     result = cv2.addWeighted(result,1,filtered,.7,0)
#     plt.figure();plt.imshow(result)
# plt.show()



