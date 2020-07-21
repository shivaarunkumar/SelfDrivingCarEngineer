import numpy as np
import cv2
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import matplotlib
import glob
import os
import pickle
matplotlib.use('QT5Agg')
fPath = os.path.abspath(__file__)
fDir = os.path.dirname(fPath)
os.chdir(fDir)

# The number of object points for this project is 9x6
# prepare object points
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Curate Image List
images = glob.glob('.\camera_cal\calibration*.jpg')

print(len(images))
# Step through all of the calibration images
for idx,fname in enumerate(images):
    image = cv2.imread(fname) #BGR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Identify Chessboard Corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (9,6), corners, ret)
        cv2.imshow('Image', image)
        
        cv2.waitKey(500)
cv2.destroyAllWindows()

img = cv2.imread(images[0])
img_size = img.shape
print(img_size[0:2])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size[0:2],None,None)
#Save the camera calibration results for later use
calibration = {}
calibration["CameraMatrix"] = mtx
calibration["DistortionCoeff"] = dist
calibration["RotationalVectors"] = rvecs
calibration["TranslationalVectors"] = tvecs
pickle.dump(calibration,open("calibration_params.p","wb"))

# Undistort calibration images
for fname in images:
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig(os.path.join('output_images', os.path.basename(fname))+'_visualize_calibration.png')