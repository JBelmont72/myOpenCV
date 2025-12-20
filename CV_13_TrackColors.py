''' 
CV_13 TrackColors without notes!

720 1280 is the webcam
'''
'''
import numpy as np
import cv2
print(cv2.__version__)
 
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
 
# width=640
# height=360
width=320 # esp32s3eye for qvga
height=240
# url = "http://192.168.1.58/stream"

# cam = cv2.VideoCapture(url)
cam = cv2.VideoCapture(1)



# cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
 
cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker',width,0)
 
hueLow=10
hueHigh=20
satLow=10
satHigh=250
valLow=10
valHigh=250
 
cv2.createTrackbar('Hue Low','myTracker',10,179,onTrack1)
cv2.createTrackbar('Hue High','myTracker',20,179,onTrack2)
cv2.createTrackbar('Sat Low','myTracker',10,255,onTrack3)
cv2.createTrackbar('Sat High','myTracker',250,255,onTrack4)
cv2.createTrackbar('Val Low','myTracker',10,255,onTrack5)
cv2.createTrackbar('Val High','myTracker',250,255,onTrack6)
 
while True:
    ignore,  frame = cam.read() #step 1 take the frame and create an HSV version
    flip=cv2.flip(frame,1)
    # resize=cv2.resize(frame,(320,240),cv2.INTER_LANCZOS4)
    # resize=cv2.resize(frame,(320,240),cv2.INTER_CUBIC)
    # resize=cv2.resize(frame,(160,120),cv2.INTER_CUBIC)
    
    frameHSV=cv2.cvtColor(flip,cv2.COLOR_BGR2HSV)
    # frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    print(frame.shape)
    #   step 2 want to box in the values high and low to isolate the deired pixels
    #   will have two arrays that bound the 'box'
    lowerBound=np.array([hueLow,satLow,valLow])
    upperBound=np.array([hueHigh,satHigh,valHigh])
    #   step 3 want to create a MSK which is an array whjich will be the size of HSV
    #   and want to keep the pixels that are in the range between upperBound and lowerBounD
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
      #step 4 want to resize "myMask" to make smaller.Note:use a tuple
    # and renamed it to 'myMaskSmall'
    myMaskSmall=cv2.resize(myMask,(int(width/2),int(height/2)))
    #step 5 we already have a myMask and displaying the myMaskSmall.
    #   now want to take the pixels that we have dialed in with the upper and lower bounds
    #   and compute a bitwiise "AND" operation.
    #   take the pixels that are on both the orignal frame and the dialed in myMaskSmall==
    #   and display in "myObject"(will resize that as well) and then iShow/moveWindow
    
    #myMask=cv2.bitwise_not(myMask)
    myObject=cv2.bitwise_and(frame,frame,mask=myMask)
    myObjectSmall=cv2.resize(myObject,(int(width/2),int(height/2)))
    cv2.imshow('My Object',myObjectSmall)
    cv2.moveWindow('My Object',int(100+width/2),int(height+100))
    
  
    cv2.imshow('My Mask',myMaskSmall)
    cv2.moveWindow('My Mask',0,int(height+100))
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',width+200,int(100))
    cv2.moveWindow('myTracker',width,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
'''
# link https://toptechboy.com/tag/artificial-intelligence/
import numpy as np
import cv2
print(cv2.__version__)
 
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
 
width=640
height=360
url = "http://192.168.1.58/stream"

cam = cv2.VideoCapture(url)
# cam=cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
 
cv2.namedWindow('myTracker')
cv2.moveWindow('myTracker',width,0)
 
hueLow=10
hueHigh=20
satLow=10
satHigh=250
valLow=10
valHigh=250
 
cv2.createTrackbar('Hue Low','myTracker',10,179,onTrack1)
cv2.createTrackbar('Hue High','myTracker',20,179,onTrack2)
cv2.createTrackbar('Sat Low','myTracker',10,255,onTrack3)
cv2.createTrackbar('Sat High','myTracker',250,255,onTrack4)
cv2.createTrackbar('Val Low','myTracker',10,255,onTrack5)
cv2.createTrackbar('Val High','myTracker',250,255,onTrack6)
 
while True:
    ignore,  frame = cam.read()
    frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerBound=np.array([hueLow,satLow,valLow])
    upperBound=np.array([hueHigh,satHigh,valHigh])
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    #myMask=cv2.bitwise_not(myMask)
    myObject=cv2.bitwise_and(frame,frame,mask=myMask)
    myObjectSmall=cv2.resize(myObject,(int(width/2),int(height/2)))
    cv2.imshow('My Object',myObjectSmall)
    cv2.moveWindow('My Object',int(width/2),int(height))
    myMaskSmall=cv2.resize(myMask,(int(width/2),int(height/2)))
    cv2.imshow('My Mask',myMaskSmall)
    cv2.moveWindow('My Mask',0,height)
    cv2.imshow('my  WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()