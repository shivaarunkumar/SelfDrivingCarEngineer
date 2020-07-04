**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./processing.jpg 
[image2]: ./Processed0.jpg 

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps (including the final overlay). 
1. **Conversion to grayscale**
    In this step the image was converted from the imported RGB format to Grayscale
2. **Gaussian Blurring**
    Applied a low pass gaussian filter to eliminate high frequency noise. Used a kernel size of 5 as the feature set of interest to us is pretty large in dimensions. The low pass filtering will have minimal effect on feature extraction.
3. **Canny Edge Detection**
    Applied Canny edge detection with *low_threshold* of **100** and a *high_threshold* of **225**
    These values seemed to work best for the challenge exercise. I wider window seemed better for the first two exercises.
4. **Polygon Masking**
    I initially employed a heuristic threshold for the first two exercises. 
    ```python
    vertices = np.array([[(100,539),(450, 325), (520, 325), (880,539)]], dtype=np.int32)
    ```
    However, this will only work for images of this specific dimension. This proved to be a problem when attempting to solve the optional challenge. Hence, I adopted a more adaptive polygon selection that was dependent on the image shape.
    ```python
    vertices = np.array([[(.1*width,.95*height),(.45*width, .6*height), (.55*width, .6*height), (.9*width,.95*height)]], dtype=np.int32)
    ```
5. **Line Detection - Using Hough Transforms**
    Use the following parameter set to detect lines from the edge detected image. 
    ```python
    rho = 1
    theta = np.pi/180 # radians
    threshold = 10
    min_line_len = 50
    max_line_gap =50
    ```
    Detection of lines wasn't a big issue but extrapolation of the line required some work. As suggested I used the computed slopes to categorize the detected lines (left and right lane markers). However, there was some miscategorization due to smaller line segments on both sides of the aisle with slopes in the incorrect direction. 
    
    *So I decided to divide the image into two halves assuming the camera is mounted in the center of the vehicle and the image in itself is centerd.* 

    I looked for lines with a negative slope in the first vertical half of the image to detect the left lane and a similar approach on the second half for the right lane.

    This worked pretty well for the first two exercises but caused issues for the optional challenge. There were almost horizontal lines detected in the image that impacted the line fitting that follows. So I decided to only look at slopes > ~26deg

    After this I used linear fit on the extracted points to extrapolate the line between:
    ```python
    ylim = [int(.6*height),int(.95*height)]
    ```
    ![Intermediate Processing Results][image1]
6. **Overlay**
    This was just a simple overlay on the original RGB image.
    ![Processed Result][image2]


### 2. Identify potential shortcomings with your current pipeline


There are several potential shortcomings in this pipeline:
* The current detection alogrithms do not work very well when the image is over exposed. This is observable in the optional challenge where the left lane isn't detected briefly.
* If the camera or the image was a bit off centered this algorithm crumble owing to the relative segmentation of the image. 
* The filtering and detection is pretty simplistic, a noisy image in low light will cause a lot of issues.


### 3. Suggest possible improvements to your pipeline

* Better Filtering
* Run on low light images
* Improve the hough transform parameters
