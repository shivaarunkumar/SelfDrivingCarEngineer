import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( ".\\SelfDrivingCarEngineer\\Lesson 6\\UndistortAndTransform\\wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('.\\SelfDrivingCarEngineer\\Lesson 6\\UndistortAndTransform\\test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    imgSize=img.shape[1::-1]
    ret,corners = cv2.findChessboardCorners(undist,(8,6))
    if ret:
        cv2.drawChessboardCorners(undist,(8,6),corners,ret)
        corner_row = corners.reshape(-1,8,2)
    src = np.float32([corner_row[0][0],corner_row[0][-1],corner_row[-1][0],corner_row[-1][-1]]) # top left, top right, bottom left, bottom right
    dst = np.float32([[100,100],[1180,100],[100,860],[1180,860]]) # Heuristic offset of 100 from each corner
    M = cv2.getPerspectiveTransform(src,dst,cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(undist,M,imgSize)
    # cv2.imshow('undistorted',undist)
    # cv2.waitKey()
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found: 
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    
    return warped, M
cv2.destroyAllWindows()
top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()