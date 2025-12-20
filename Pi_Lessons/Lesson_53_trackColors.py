'''
1. create trackbars for hue 0-180 value 0-255 saturation 0 - 255
2. convert frame bgr to hsv

low bound. np.array[hue,val,sat]
upper bound
mask1 = np.inrange(hsvFrame,lowerBound,upperBound)
if make two np.inarnge() then use bitwise
  myMask=myMask | myMask2
    #myMask=cv2.add(myMask,myMask2)
    #myMask=np.logical_or(myMask,myMask2)
 
    #myMask=cv2.bitwise_not(myMask)
then bitwise with the original frame


note the second program is from PaulMcWhorter (same as mine) I commented out the picamera specific commands
'''

'''
import sys      # my version, Works
import cv2
print(cv2.__version__)
import numpy as np
import time
print(f"This is version {sys.version}")
print(np.__version__)
myRadius =20
width=640
height=480
def hueLow(val):
    global LowHue
    LowHue=val
    print('hueLow: ',LowHue)
def hueHigh(val):
    global HighHue
    HighHue=val
    print('hueLow: ',HighHue)

def satLow(val):
    global LowSat
    LowSat=val
    print('satLow: ',LowSat)   
def satHigh(val):
    global HighSat
    HighSat=val
    print('satHigh: ',HighSat)   
def valLow(val):
    global LowVal
    LowVal = val
    print('valLow: ',LowVal)
def valHigh(val):
    global HighVal
    HighVal = val
    print('valHigh: ',HighVal)

cam = cv2.VideoCapture(1)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow('my Trackbars')
startTime=time.time()
FPS=30
cv2.namedWindow('myTrackbars')
cv2.createTrackbar('hueLow','myTrackbars',0,179,hueLow)
cv2.createTrackbar('hueHigh','myTrackbars',125,179,hueHigh)
cv2.createTrackbar('satLow','myTrackbars',0,255,satLow)
cv2.createTrackbar('satHigh','myTrackbars',125,255,satHigh)
cv2.createTrackbar('valLow','myTrackbars',0,255,valLow)
cv2.createTrackbar('valHigh','myTrackbars',125,255,valHigh)

while True:
    _,frame=cam.read()
    timeDiff=startTime-time.time()
    startTime=time.time()
    fps=int(FPS*.97 + .03*(1/timeDiff))
    print(fps)
    HSVframe=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowBound=np.array([LowHue,LowSat,LowVal])
    highBound=np.array([HighHue,HighSat,HighVal])
    mask1=cv2.inRange(HSVframe,lowBound,highBound)
    img=cv2.bitwise_and(frame,frame,mask=mask1)
    cv2.putText(frame,f'{str(fps)} fps',(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    # cv2.imshow('myWebcam',frame)
    cv2.imshow('myWebcam',img)
    cv2.moveWindow('myWebcam',20,200)
    cv2.imshow('Mask_1',mask1)
    if cv2.waitKey(1) & 0xff == ord('c'):
        break
cv2.destroyAllWindows()
cam.release()
'''

###  PW s version for picamera, i commented out the picamera commands so i could run it on my macwith just opencv

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






# import cv2
# from picamera2 import Picamera2
# import time
# import numpy as np
# picam2 = Picamera2()
# dispW=1280
# dispH=720
# picam2.preview_configuration.main.size = (dispW,dispH)
# picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.controls.FrameRate=30
# picam2.preview_configuration.align()
# picam2.configure("preview")
# picam2.start()
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
    print('Hue Low',valHigh)
 
 
cv2.namedWindow('myTracker')
 
cv2.createTrackbar('Hue Low','myTracker',10,179,onTrack1)
cv2.createTrackbar('Hue High','myTracker',30,179,onTrack2)
cv2.createTrackbar('Sat Low','myTracker',100,255,onTrack3)
cv2.createTrackbar('Sat High','myTracker',255,255,onTrack4)
cv2.createTrackbar('Val Low','myTracker',100,255,onTrack5)
cv2.createTrackbar('Val High','myTracker',255,255,onTrack6)
 
while True:
    tStart=time.time()
    
    _,frame=cam.read()
    
    # frame= picam2.capture_array()
    cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    lowerBound=np.array([hueLow,satLow,valLow])
    upperBound=np.array([hueHigh,satHigh,valHigh])
    frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    myMaskSmall=cv2.resize(myMask,(int(dispW/2),int(dispH/2)))
    myObject=cv2.bitwise_and(frame, frame, mask=myMask)
    myObjectSmall=cv2.resize(myObject,(int(dispW/2),int(dispH/2)))
    cv2.imshow("Camera", frame)
    cv2.imshow('my Mask',myMaskSmall)
    cv2.imshow('My Objest',myObjectSmall)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps=.9*fps + .1*(1/loopTime)
# picam.stop()
cv2.destroyAllWindows()
