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

# Generate Perspective Transformation Matrix
fname = '.\\projectvidframes\\frame317.jpg'
image = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
plt.figure();plt.imshow(image);plt.suptitle('original image')
undistorted = fp.undistort(image)    
src = np.float32([[578,463],[700,463],[1160,719],[240,719]]) # top left, top right, bottom left, bottom right
plt.figure()
plt.imshow(cv2.polylines(undistorted.copy(),np.int32([src]),True,(255,0,0),3))
plt.suptitle('undistored image')
dst = np.float32([[400,150],[1000,150],[1000,719],[400,719]]) 
M = cv2.getPerspectiveTransform(src,dst,cv2.INTER_LINEAR)
warped = cv2.warpPerspective(undistorted,M,undistorted.shape[1::-1])
plt.figure()
plt.imshow(cv2.polylines(warped.copy(),np.int32([dst]),True,(0,0,255),3))
plt.suptitle('warped image')
Minv = cv2.getPerspectiveTransform(dst,src,cv2.INTER_LINEAR)
unwarped = cv2.warpPerspective(warped,Minv,warped.shape[1::-1])
plt.figure()
plt.imshow(cv2.polylines(unwarped.copy(),np.int32([src]),True,(0,0,255),3))
plt.suptitle('unwarped image')

pTransform={}
pTransform['M']=M
pTransform['Minv']=Minv
pTransform['src'] = src
pTransform['dst'] = dst
pickle.dump(pTransform,open('pv_pTransformData2.p','wb'))
plt.show()