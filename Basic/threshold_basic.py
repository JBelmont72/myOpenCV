'''

'''

import cv2

# Loading the image named test.jpg
img = cv2.imread("/Users/judsonbelmont/Documents/untitled folder/myOpenCV/images/Cat.jpeg")

# Converting color mode to Grayscale
# as thresholding requires a single channeled image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Thresholding the image placing 127 intensity level as threshold
# Pixel values below 127 would be changed to Black
# Pixel  values above 127 would be changed to White (255)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#### inverse of threshold
ret, threshInv = cv2.threshold(img, 127, 225, cv2.THRESH_BINARY_INV)
### truncating the image
ret, threshTrunchated = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# Displaying the output image
cv2.imshow('Truncate Threshold', threshTrunchated)
### thresholding to zero
ret, threshToZero = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
## also has cv2.Thresh_tozero_inverse method
# Displaying the output image
cv2.imshow('Truncate Threshold To Zero', threshToZero)





# Displaying the output image
cv2.imshow('Binary Inverse Threshold', threshInv)



# Displaying the output image
cv2.imshow('Binary Threshold', thresh)


cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
