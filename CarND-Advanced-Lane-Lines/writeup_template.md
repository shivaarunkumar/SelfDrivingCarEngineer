## Project Writeup 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.jpg_visualize_calibration.png "Calibration"
[image2]: ./output_images/undistorted_image.png "Undistorted"
[image3]: ./output_images/filtered_image.png "Filtered"
[image4]: ./output_images/Hue_Threshold.png "Hue Thresholding"
[image5]: ./output_images/Color_Threshold.png "Color Thresholding"
[image6]: ./output_images/Value_Threshold.png "Value Thresholding"
[image7]: ./output_images/Saturation_Threshold.png "Saturation Thresholding"
[image8]: ./output_images/Gradient_Threshold.png "Gradient Thresholding"
[image9]: ./output_images/White_Threshold.png "White Thresholding"
[image10]: ./output_images/Combined_Threshold.png "Combined Thresholding"
[image11]: ./output_images/Warped_Image.png "Warped Image"
[image12]: ./output_images/Unwarped_Binary_Image.png "Unwarped Image"
[image13]: ./output_images/Fit_Lane.png "Lane Fit"
[image14]: ./output_images/Final_Output.png "Final Output"
[image15]: ./output_images/Undistorted_WithSourcePoly.png "Undistorted With Source"
[image16]: ./output_images/Warped_WithDestinationPoly.png "Warped with Destination"
[image17]: ./output_images/Unwarped_WithSourcePoly.png "Unwarped with Source"
[image18]: ./output_images/Fit_WithBadDeviation.png "Fit With Bad Deviation"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---
### Camera Calibration

#### Computation and Results

**Code :**       ./cameracalibrate.py
**Images :**     ./output_images/calibration*.png
**Pickle File:** ./calibration_params.p

The provided chessboard calibration images have **9x6 (54)** calibration corners. 

Its assumed that the chessboard is fixed on a 2D (x,y) plane. The first step here is to define the _object points_ (`objp`), which are the target or the real world locations for the chessboard corners. (0,0) ... (8,5)

The next steps is to detect the actual "corners" or _image points_ using `cv2.findChessboardCorners()`

We loop through all of the provided calibration images and concatenate the object points in `objpoints` and image points in `imgpoints` respectivelu.  We then use the final set of values stored in these two arrays to compute the  camera calibration matrix (`mtx`) with the help of the `cv2.calibrateCamera` function.

Now I used this matrix to undistort one of the provided calibration images using the `cv2.undistort()` method. As you can see from the result below, I was able to successfully undistort the images:

![undistored calibration image][image1]


### Lane Detection

**Python Module Package :** ./ldpackage

In order to have a clear separation of code, I have authored the following modules:

1. `ldetect.LaneDetector` : Used to process the input MP4 video and write the output  to a desired MP4 video.
**Example**
`ld=LaneDetector('project_video.mp4','processed_project_video.mp4')`
`ld.process()`
2.  `ldpackage.process` : This includes all the required processing functions that can potentially be leveraged within the processing pipeline. It also includes the `Line()` class to keep track of lane history.

#### Processing Pipeline

Check the function `process_frame` in **./ldpackage/process.py**

#### 1. Distorion Correction

The `undistort` method in the frame processing module, leverages the previously generated calibration matrix to produce an undistorted image.

![undistorted frame][image2]

#### 2. Filtering

Before performing any further processing, the first step was to filter the image to remove any high frequency noise. For this, I use the standard Gaussian Blur Filtering with a kernel size of **5** as we have small features to worry about.

![filtered frame][image3]

#### 3. Thresholding

After trying the various different type of thresholding , I used the following combination of thresholding:

**Component/Channel Threshold + Gradient Threshold + White Color Threshold**

##### 3.1 Component Threshold

I visualized the various channels R,G,B,H,S,V,L  and finally identified that the following steps yield the best result. It uses a quality parameter called `overexposure`. This is to indicate if we let through too many pixel values.

`overexposure = (number of high pixels in bottom half of image)/(total number of pixels in bottom half)`

The order of precedence is (lines 427-440 in ./ldpackage/process.py):
Hue &rarr; Red &rarr; Value &rarr; Saturation

Here are typical examples:

![Hue Threshold][image4]
![Red Threshold][image5]
![Value Threshold][image6]
![Saturation Threshold][image7]

##### 3.2 Gradient Threshold

The second thresholding technique leveraged is the gradient threshold to detect the lane boundaries.

![Gradient Threshold][image8]

##### 3.3 White Threshold

Since the dashed lane markings are harder to detect, I wanted to squeeze the benefits of looking for the color white with a high threshold limit.

![White Threshold][image9]

##### 3.4 Combine Threshold

The final thresholded image is 

Component Threshold | Gradient Threshold | White Threshold

![Combined Threshold][image10]

#### 4. Perspective Transform.


**Code :** ./perspective.py 
**Pickle File :** ./pv_pTransformData2.p

In order to compute the _Transformation_ and _Inverse Transformation_ Matrices using `cv2.getPerspectiveTransform()`, I visualized the various frames of the project video (I used the test images as well first but wanted to get one that looked good on the project frames) and the hard coded the following `src` and `dst` points (Note : reversed for inverse transformation)

```python
src = np.float32([[578,463],[700,463],[1160,719],[240,719]]) # top left, top right, bottom left, bottom right
dst = np.float32([[400,150],[1000,150],[1000,719],[400,719]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578,463       | 400, 150      | 
| 700,463       | 1000, 150     |
| 1160, 719     | 1000, 719     |
| 240, 719      | 400, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here are the results with the bounding polygons:


![Undistorted with Source][image15]
![Warped with Destination][image16]
![Unwarped with Source][image17]

![Warped Binary Image][image11]
![Unwarped Binary Image][image12]

#### 5. Lane Fit

**Code File :** ./ldpackage/process.py
**Functions of Interest :** 
* `fit_lane(binary_warped, lines = Lines(),ym_per_pix = 30/720, xm_per_pix = 3.7/700)`
* `find_lane_pixels_fromprev(binary_warped,lines)`
* `find_lane_pixels(binary_warped)`

The fitting of the lane is very similar to what we did during class with a few slight modifications:
* To leverage historical results (`Lines()`)
* To determine quality of fit and take appropriate actions (`View Covariance Matrix`)
* Addinal hyperparameter to avoid pixels on either edges of the image (`padding`)

Let me start with showing the final result of lane fit for the 1st frame we have been looking at:

![Fit Lane][image13]

Now let us a dig a bit deeper into the lane fitting process:

In the absense of apriori information the process is simple:
* Determine base pixels by using the histogram along X
* Use the windowing technique to identify lane pixels
* Use `np.polyfit` to obtain a secord order polynomial fit
* Generate `left_fitx` and `right_fitx` values using `np.polyval`

##### Quality of Fit

In the code I call `np.polyfit` with the `cov=True` option. I then look at the deviations for the generated `*_fitx` values by computing:

`X.C.Xt`

where `t` indicates transpose, `X` is the generate fit values and `C` is the covariance matrix

I then look at th maximum value of deviation observed and consider it a bad fit if the deviation is more than a hueristic number `15`

**Lines 311-321 in ./ldpackage/process.py**
```python
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
```

Here is an example of a bad fit (the magenta indicates the amount of deviation in the predicted X values):

![Bad Deviation][image18]

##### History
**Class :** `Lines()`

* I use the information in the last 50 frames to predict the lane in the absence of a good fit. 
* If a good fit was obtained in the previous run then I look for lane pixel around the previous fit instead of using the histogram based approach.

#### 6. Curvature Computation

**File :** ./ldpackage/process.py
**Function :** `fit_lane`

The left and right curvatures were computed as we did in class.
* Choose the base value of `y=719` as the point where we will be computing the curvature. (Line : 299)
* Assume the pixel to meters conversion to be the same as what we used in the class. We are still dealing with US lanes and highways. `ym_per_pix = 30/720, xm_per_pix = 3.7/700`
* Compute real world fits after converting lane pixels to real world values. [`left_fit_cr`,`right_fit_cr`]  _Lines_ : 304 and 307
* Curvature is computed using the formula discussed during the course. [`left_curverad` and `right_curverad`] _Lines_ 305 and 308
* Now the final curvature value is typically the average of both the left and right curvature values , if both fits are good. If not it is the one associated with the good fit. If neither of the fits are good, then we use the one that was previously computer.

#### 7. Lane Deviation

_File_ ./ldpackage/process.py
_Function_  `fit_lane`
_Lines_ 326-332

Using the real world fits from the previous step, I computer the lane ends `left_limit` and `right_limit` using `np.polyval`. I then computer the center of this as `lane_center`, now the displacement of the car from the lane center `disp` was computed as the difference between the lane center and the image center in real world values. 


The final result of the process pipeline is as follows:


![Final Result][image14]

---

### Pipeline (video)

Here's a [link to my video result](./processed_project_video.mp4)

---

### Discussion

#### Potential Improvement 

* Currently I do not filter the current fit with the history of previous fits if I deem it good. This can lead to Jitter. 
* For the harder challenge video , I think I should use a higher order fit rather than a second order fit. 
* The thresholds are currently very permissive and this will lead to a lot of noise for the other challenge videos.
* The quality criteria for the two fits needs to be better. Intersection between the lanes within the image frame should be detected and flagged as an incorrect fit.
