'''Started with Lesson 13 HOmework tracking two colors Find the contour on the mask, not on the frame
And now will try to understand and create CONTOURS
THis creates a bounding box and works quite well.  Good notes.
CV_14_TrackCOntours.py is a copied version from TOPTECHBOY
This is mine starting with his Lesson 13 HW
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!

Have to trace the perimeter on the mask!
Find the Mask first and then trace along two hue ranges.
My Mask is a composite of the two masks(one mask using the hue1 highand low and the second using hue 2 high and low)

Create contours which are arrays
findContours looks for white spots on a black background, We will need to filter such as by area
 How to deal with different contours within an object. we will just look for the EXTERNAL perimeter to simplify it.
 cv2.CHAIN_APPROX_SIMPLE will cut down onthe data returned
once we get 'contours,junk' , we want to draw them on the oriignal image using the command 
cv2.drawContours(frame,contours,-1)  we select -1 so all the contours are drawn

'''
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)

 #  640  380  <> 960 540
hueLow=90
hueHigh=100
 
hueLow2=90
hueHigh2=100
 
satLow=20
satHigh=200
valLow=20
valHigh=200
def onTrack1(val):
    global hueLow
    hueLow=val
    print('Low Hue: ',val)
def onTrack2(val):
    global hueHigh
    hueHigh=val
    print('High Hue: ',val)
def onTrack3(val):
    global satLow
    satLow=val
    print('Low Sat: ',val)
def onTrack4(val):
    global satHigh
    satHigh=val
    print('High Sat: ',val)
def onTrack5(val):
    global valLow
    valLow=val
    print('Low Val: ',val)
def onTrack6(val):
    global valHigh
    valHigh=val
    print('High Val: ',val)
 
 
def onTrack7(val):
    global hueLow2
    hueLow2=val
    print('Low Hue2: ',val)
def onTrack8(val):
    global hueHigh2
    hueHigh2=val
    print('High Hue2: ',val)
 
 
width=640
height=380
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker',width,0)
cv2.resizeWindow('myTracker',400,500)
cv2.createTrackbar('Hue Low','myTracker',15,180,onTrack1)
cv2.createTrackbar('Hue High','myTracker',30,180,onTrack2)
 
cv2.createTrackbar('Hue Low2','myTracker',50,180,onTrack7)
cv2.createTrackbar('Hue High2','myTracker',60,180,onTrack8)
 
cv2.createTrackbar('Sat Low','myTracker',10,255,onTrack3)
cv2.createTrackbar('Sat High','myTracker',255,255,onTrack4)
cv2.createTrackbar('Val Low','myTracker',10,255,onTrack5)
cv2.createTrackbar('Val High','myTracker',255,255,onTrack6)
 
 
while True:
    ignore,  frame = cam.read()
    frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
 
    lowerBound=np.array([hueLow,satLow,valLow])
    upperBound=np.array([hueHigh,satHigh,valHigh])
 
    lowerBound2=np.array([hueLow2,satLow,valLow])
    upperBound2=np.array([hueHigh2,satHigh,valHigh])
 
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    myMask2=cv2.inRange(frameHSV,lowerBound2,upperBound2)
 
    myMask=myMask | myMask2
    #myMask=cv2.add(myMask,myMask2)
    #myMask=np.logical_or(myMask,myMask2)
    contours,junk = cv2.findContours(myMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #the below line is commented out in favor of the for loop 
    # cv2.drawContours(frame,contours,-1,(255,0,0),3)
 # to refine this we wnt to trace more selectively,here by size
 #comment out the above cv2.drawContoure(frame....)
    for contour in contours:
        area =cv2.contourArea(contour)
        if area>200: 
            # cv2.drawContours(frame,contour,0,(255,0,0),3)
            
            #expecting an array of arrays so there is a shape/data mismatch
            # myContour=[contour]
            # cv2.drawContours(frame,myContour,0,(255,0,0),3)
            #alternative is: (puttong an array within an array to make fit!)
            # cv2.drawContours(frame,[contour],0,(255,0,0),3)
            # Will Create a BOUNDONG BOX:
            # the boundingRect wants the simple array which will be four parameters
            #the upper left corner and the lower right corner.
            #the x and y are the upper left corner and the w and h are width and height from that point
            x,y,w,h=cv2.boundingRect(contour) #wants one contour and it will return the info neede for a rectangle construction
            cv2.rectangle(frame,(x,y),(int(x+w),int(y+h)),(0,0,255),3)      #notice tuples




    #myMask=cv2.bitwise_not(myMask) #this command (for fun)reverses the images_see only the background and not the selected image(would be black)
    myMaskSmall=cv2.resize(myMask,(int(width/2),int(height/2)))
    mySelection=cv2.bitwise_and(frame,frame, mask=myMask)
    mySelection=cv2.resize(mySelection,(int(width/2),int(height/2)))
    cv2.imshow('My Selection', mySelection)
    cv2.moveWindow('My Selection',int(width/2),int(height+100))
 
    cv2.imshow('My Mask', myMaskSmall)
    cv2.moveWindow('My Mask',0,int(height+100))
    cv2.imshow('my WEBcam',frame)
    cv2.moveWindow('my WEBcam',0,int(100))
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()