'''
Docstring for corner_detction
use astype(int) for converting NumPy arrays.
Int Conversion: Changed np.intp(corners) to np.int0(corners) for converting corner values to integers properly.
Image Display: Used cv2.cvtColor(img, cv2.COLOR_BGR2RGB) to display the image correctly in matplotlib since it uses RGB while OpenCV uses BGR.
Axis Hiding: Added plt.axis('off') for better visual aesthetics by hiding the axes when displaying the image.
Removed cv2.waitKey: The waitKey function is not necessary for image display in this context, as plt.show() will handle the image lifecycle
'''


# Python program to illustrate
# corner detection with
# Shi-Tomasi Detection Method

# organizing imports
# Python program to illustrate
# corner detection with
# Shi-Tomasi Detection Method

# organizing imports
import cv2
import numpy as np 
import matplotlib.pyplot as plt

# path to input image specified and
# image is loaded with imread command
img = cv2.imread('images/shapes.jpg')

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi corner detection function
# We are detecting only 100 best corners
# You can change the number to get the desired amount
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10)

# convert corners values to integer
# So that we will be able to draw circles
# corners = np.int0(corners)  # Changed intp to int0 deprecated
corners = corners.astype(int)  # Ensure the conversion to integer is done correctly
# draw red color circles on all corners
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

# resulting image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Use cv2.cvtColor to display correctly in matplotlib
plt.axis('off')  # Hide axes for better visual
plt.show()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
