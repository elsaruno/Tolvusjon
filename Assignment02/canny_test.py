# code from https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# used to determine the threshold used for the canny edge detection in real_time_line_detector.py
# took a photo of the edge i was testing the code with
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('test_photo2.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img,50,150)
 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()