'''
how to track an object of interest based on color in OpenCV. 
how to create masks, contours, and then how to box the contour of the object of interest. 
way to train the system for finding the Object of Interest
PW lesson54 Raspberry pi series

'''

'''
import cv2
from picamera2 import Picamera2
import time
import numpy as np
picam2 = Picamera2()
dispW=1280
dispH=720
picam2.preview_configuration.main.size = (dispW,dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate=30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
'''

# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# import time
# print(f"This is version {sys.version}")
# print(np.__version__)
# myRadius =20
# dispW=640
# dispH=480

# cam = cv2.VideoCapture(1)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))


# fps=0
# pos=(30,60)
# font=cv2.FONT_HERSHEY_SIMPLEX
# height=1.5
# weight=3
# myColor=(0,0,255)

# def onTrack1(val):
#     global hueLow
#     hueLow=val
#     print('Hue Low',hueLow)
# def onTrack2(val):
#     global hueHigh
#     hueHigh=val
#     print('Hue High',hueHigh)
# def onTrack3(val):
#     global satLow
#     satLow=val
#     print('Sat Low',satLow)
# def onTrack4(val):
#     global satHigh
#     satHigh=val
#     print('Sat High',satHigh)
# def onTrack5(val):
#     global valLow
#     valLow=val
#     print('Val Low',valLow)
# def onTrack6(val):
#     global valHigh
#     valHigh=val
#     print('Val High',valHigh)

# cv2.namedWindow('myTracker')

# cv2.createTrackbar('Hue Low','myTracker',10,179,onTrack1)
# cv2.createTrackbar('Hue High','myTracker',20,179,onTrack2)
# cv2.createTrackbar('Sat Low','myTracker',100,255,onTrack3)
# cv2.createTrackbar('Sat High','myTracker',2555,255,onTrack4)
# cv2.createTrackbar('Val Low','myTracker',100,255,onTrack5)
# cv2.createTrackbar('Val High','myTracker',255,255,onTrack6)


# while True:
#     tStart=time.time()
#     # frame= picam2.capture_array()
#     _,frame = cam.read()
#     frame =cv2.flip(frame,-1)
#     #frame=cv2.flip(frame,-1)# if need to flip image depending on camera position
#     frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
#     lowerBound=np.array([hueLow,satLow,valLow])
#     upperBound=np.array([hueHigh,satHigh,valHigh])
#     myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
#     myMaskSmall=cv2.resize(myMask,(int(dispW/2),int(dispH/2)))
#     myObject=cv2.bitwise_and(frame,frame, mask=myMask)
#     myObjectSmall=cv2.resize(myObject,(int(dispW/2),int(dispH/2)))
    
#     contours,junk=cv2.findContours(myMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours)>0:
#         contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
#         #cv2.drawContours(frame,contours,-1,(255,0,0),3)
#         contour=contours[0]
#         x,y,w,h=cv2.boundingRect(contour)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    
#     cv2.imshow("Camera", frame)
#     cv2.imshow('Mask',myMaskSmall)
#     cv2.imshow('My Object',myObjectSmall)
#     if cv2.waitKey(1)==ord('q'):
#         break
#     tEnd=time.time()
#     loopTime=tEnd-tStart
#     fps=.9*fps + .1*(1/loopTime)
# cv2.destroyAllWindows()

## second copy of above that i comment extensively

import sys
import cv2
print(cv2.__version__)
import numpy as np
import time
print(f"This is version {sys.version}")
print(np.__version__)
myRadius =20
dispW=640
dispH=480

cam = cv2.VideoCapture(1)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))


fps=0
pos=(30,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(0,0,255)

def onTrack1(val):
    global hueLow
    hueLow=val
    print('Hue Low',hueLow)
def onTrack2(val):
    global hueHigh
    hueHigh=val
    print('Hue High',hueHigh)
def onTrack3(val):
    global satLow
    satLow=val
    print('Sat Low',satLow)
def onTrack4(val):
    global satHigh
    satHigh=val
    print('Sat High',satHigh)
def onTrack5(val):
    global valLow
    valLow=val
    print('Val Low',valLow)
def onTrack6(val):
    global valHigh
    valHigh=val
    print('Val High',valHigh)

cv2.namedWindow('myTracker')

cv2.createTrackbar('Hue Low','myTracker',10,179,onTrack1)
cv2.createTrackbar('Hue High','myTracker',20,179,onTrack2)
cv2.createTrackbar('Sat Low','myTracker',100,255,onTrack3)
cv2.createTrackbar('Sat High','myTracker',2555,255,onTrack4)
cv2.createTrackbar('Val Low','myTracker',100,255,onTrack5)
cv2.createTrackbar('Val High','myTracker',255,255,onTrack6)


while True:
    tStart=time.time()
    # frame= picam2.capture_array()
    _,frame = cam.read()
    frame =cv2.flip(frame,1) # 1 flips around y, 0 around x and _1 around both axes
    #frame=cv2.flip(frame,-1)# if need to flip image depending on camera position
    frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    lowerBound=np.array([hueLow,satLow,valLow])
    upperBound=np.array([hueHigh,satHigh,valHigh])
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound) # this is the black and white in the HSV frame
    myMaskSmall=cv2.resize(myMask,(int(dispW/2),int(dispH/2)))
    myObject=cv2.bitwise_and(frame,frame, mask=myMask)  # take the mask from the HSV and bitwise_and with the orignal frame toget the color object on black background
    myObjectSmall=cv2.resize(myObject,(int(dispW/2),int(dispH/2)))
    
    contours,junk=cv2.findContours(myMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   
    # if len(contours)>0: # this will draw lots of contours
    #     contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    #     cv2.drawContours(frame,contours,-1,(255,0,0),3) # the -1 is draw all, can make 0 to draw just the first
    #     # contour=contours[0]
    #     # x,y,w,h=cv2.boundingRect(contour)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
## create a box around the object of interest  
    if len(contours)>0:
        contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        #cv2.drawContours(frame,contours,-1,(255,0,0),3)
        contour=contours[0]
        x,y,w,h=cv2.boundingRect(contour)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
# cv2.boundingRect(). provides the smallest upright rectangle that contains the specified contour points,    
    cv2.imshow("Camera", frame)
    cv2.imshow('Mask',myMaskSmall)
    cv2.imshow('My Object',myObjectSmall)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps=.9*fps + .1*(1/loopTime)
cv2.destroyAllWindows()



'''
Explanation of how the key arguement for the lambda function is passed to the expression (key function proceese the key arguement).
The syntax  is not specific to the  (OpenCV) library itself, but rather a standard Python feature used with built-in functions like , , , or , which accept a  argument. It's used to specify a custom sorting or comparison criterion. [1, 2, 3, 4, 5]  
Explanation • : This Python keyword creates a small, anonymous (nameless) function inline. It is defined in a single line and automatically returns the result of its single expression. The basic syntax is . 
• : This is a keyword argument for functions that perform sorting or comparison. The function passed to  is called once for each element in the list/iterable. The value returned by this "key function" is then used for making the comparisons during the sort. 
• : This part is likely a placeholder in the user's mind, representing the elements being iterated over and processed by the lambda function. [2, 6, 7, 8, 9]  

How it works with  (Context) While  has many functions, it doesn't generally have functions that directly take a  argument in this manner. However, you might use this Python pattern in an OpenCV-related script for tasks such as: 

• Sorting contours: After finding contours using , you might sort them based on their area, perimeter, or position (e.g., to process them from left to right). 
• Sorting keypoints/features: When working with feature matching, you might sort a list of keypoints based on their size, response value, or location. [2, 11, 12, 13, 14]  

Example: Sorting contours by area A common use case involves sorting a list of contours (which are lists of points) based on their area, which is calculated using : [15, 16, 17]  
python program example:
import cv2

# ... assume 'contours' is a list of contours found by cv2.findContours() ...

# Sort contours in ascending order by area
sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c))

# Sort contours in descending order by area
sorted_contours_desc = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

Looks like within in the sorted method, the key arguement is being processed by the method in this case  cv2.contourArtea() and then passed to the sorted method that sorts them from ascending or optionally descending order


In this example, the  function takes a single contour () as input and returns its area (), which is then used as the actual value for sorting. [2, 18, 19, 20, 21]  

AI responses may include mistakes.

[1] https://stackoverflow.com/questions/13669252/what-is-lambda-in-python-code-how-does-it-work-with-key-arguments-to-sorte
[2] https://docs.python.org/3/howto/sorting.html
[3] https://blog.devgenius.io/how-to-use-python-lambda-functions-b5f2169d0976
[4] https://favtutor.com/blogs/python-sort-lambda
[5] https://hyperskill.org/university/python/max-in-python
[6] https://pentera.io/glossary/lambda-function/
[7] https://www.youtube.com/watch?v=a9hIUuv8hZs
[8] https://www.digitalocean.com/community/tutorials/lambda-expression-python
[9] https://towardsdatascience.com/simple-ways-to-apply-lambda-function-in-python-7382276403a4/
[10] https://www.aionlinecourse.com/blog/opencv-final-year-project-ideas-and-guidelines
[11] https://stackoverflow.com/questions/52344055/sorted-lambda-key-value-syntax
[12] https://medium.com/@ksakthivelan4002/extracting-white-boxes-in-python-using-opencv-987553d41e9c
[13] https://omgimanerd.medium.com/beating-maplestorys-captchas-with-opencv-3784519117fe
[14] https://www.nutanix.dev/nutanix-api-user-guide/
[15] https://medium.com/@staytechrich/computer-vision-001-color-detection-with-opencv-58426c880449
[16] https://datahacker.rs/006-opencv-projects-how-to-detect-contours-and-match-shapes-in-an-image-in-python/
[17] https://cvexplained.wordpress.com/2020/06/06/sorting-contours/
[18] https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
[19] https://medium.com/@kumar.atul.2122/java-8-lambda-functional-interface-method-reference-stream-api-and-optional-class-f685143635fb
[20] https://keremkargin.medium.com/license-plate-detection-with-opencv-python-e14ff6f9f2de
[21] https://medium.com/@evelynli_30748/map-apply-applymap-with-the-lambda-function-5e83028be759



'''