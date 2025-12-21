'''

'''

import cv2

# img =cv2.imread('/Users/judsonbelmont/Documents/Projects/OpenCV_2/images/Judson.jpg',-1)
# print(img)
# cv2.imshow('JUDSON',img)

# # cv2.imwrite('name_the_copy.png',img) # can use jpg. or .png etc.  makes a copy in images
# cv2.waitKey(0) # if 0 then you close with anyt key press on the image.  or specify # of millisecs
# cv2.destroyAllWindows
#########################
img =cv2.imread('/Users/judsonbelmont/Documents/Projects/OpenCV_2/images/Judson.jpg',-1)
print(img)
cv2.imshow('JUDSON',img)


k = cv2.waitKey(0) & 0xff # if 0 then you close with anyt key press on the image.  or specify # of millisecs
if k == 27:
    cv2.destroyAllWindows
elif k == ord('s'):  # press s to save
    cv2.imwrite('name_the_copy.png',img) # can use jpg. or .png etc.  makes a copy in image
    cv2.destroyAllWindows
# while True:
#     cv2.imread('/Users/judsonbelmont/Documents/Projects/OpenCV_2/images/Judson.jpg',0)
# # cv2.imread('/Users/judsonbelmont/Documents/Projects/OpenCV_2/images/lena.jpg',0)if cv2.waitKey(1) & 0xff ==ord('c'):
#     cv2.imshow('/Users/judsonbelmont/Documents/Projects/OpenCV_2/images/Judson.jpg')
#     if cv2.waitKey(1) & 0xff ==ord('c'):
#         break

# cv2.destroyAllWindows()

 