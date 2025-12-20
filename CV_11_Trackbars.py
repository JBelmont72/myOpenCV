'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
program 2 creates trackbars with x, y , width, height so i can use them to create rectangles of any size. 
could be used for creating a ROI
'''
# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# print(f"This is version {sys.version}")
# print(np.__version__)
# myRadius =20
# width=640
# height=480
# xPos = int(width/2)
# yPos =int(height/2)
# myBlue=40
# myColor= [myBlue,0,255]
# def myCallBack1(val):
#     global xPos
#     print('xPos: ',val)
#     xPos =val
# def myCallBack2(val):
#     global yPos
#     print('yPos: ',val)
#     yPos = val
# def myCallBack3(val):
#     global myRadius
#     print('myRadius: ',val)
#     myRadius = val
# def myCallBack4(val):
#     global myBlue
#     print('my Blue', val)
#     myBlue =val

# cam = cv2.VideoCapture(0)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
# cv2.namedWindow('my Trackbars')
# cv2.resizeWindow('my Trackbars',500,300)
# cv2.moveWindow('my Trackbars',width,height)
# cv2.createTrackbar('xPos','my Trackbars',xPos,width,myCallBack1)
# cv2.createTrackbar('yPos','my Trackbars',yPos,height,myCallBack2)
# cv2.createTrackbar('my Radius','my Trackbars',myRadius,int(height/2),myCallBack3)
# cv2.createTrackbar('my Blue','my Trackbars',myBlue,255,myCallBack4)
# while True:
#     ignore,  frame = cam.read()
#     cv2.circle(frame,(xPos,yPos),myRadius,myColor,2)
#     print(myColor, frame.shape)
#     myColor =[myBlue,0,255]
#     cv2.imshow('my WEBcam', frame)
#     cv2.moveWindow('my WEBcam',0,100)
#     if cv2.waitKey(1) & 0xff ==ord('q'):
#         break
# cam.release()

'''
cv2. is width,heigt
numpy is height ,width
Width 640
height 480
y=720, x =1280
'''



import sys
import cv2
print(cv2.__version__)
import numpy as np
import time
print(f"This is version {sys.version}")
print(np.__version__)
myRadius =20
width=320
height=240
# width=640
# height=480
myBlue=125
myRed=125
myColor=[myBlue,0,0]

def xPos(val):
    global xVal
    xVal=val
    print('x Val: ',xVal)
def yPos(val):
    global yVal
    yVal=val
    print('y Val: ',yVal)
def Width(val):
    global Width
    Width= val
    print('width: ',Width)
def Height(val):
    global Height
    Height=val
    print('height: ',Height)
    
def Blue(val):
    global myBlue
    myBlue=val
    print('blue val: ',myBlue)
def Red(val):
    global myRed
    myRed=val
    print('my Red: ',myRed)
url = "http://192.168.1.58/stream"

cam = cv2.VideoCapture(url)
# cam = cv2.VideoCapture(1)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow('my Trackbars')
startTime=time.time()
FPS=30
cv2.namedWindow('myTrackbars')
cv2.createTrackbar('x Pos','myTrackbars',10,width,xPos)
cv2.createTrackbar('y Pos','myTrackbars',10,height,yPos)
cv2.createTrackbar('width','myTrackbars',10,width,Width)
cv2.createTrackbar('height','myTrackbars',10,height,Height)
cv2.createTrackbar('blue','myTrackbars',myBlue,255,Blue)
cv2.createTrackbar('red','myTrackbars',myRed,255,Red)

while True:
    _,frame=cam.read()
    timeDiff=startTime-time.time()
    startTime=time.time()
    fps=int(FPS*.97 + .03*(1/timeDiff))
    print(fps,'  ',frame.shape)
    
    HSVframe=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    myColor=[myBlue,30,myRed]
    cv2.rectangle(frame,(xVal,yVal),((xVal+Width),yVal+Height),myColor,2)
    print(myColor)
    cv2.putText(frame,f'{str(fps)} fps',(30,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow('myWebcam',frame)
    cv2.moveWindow('myWebcam',20,200)
    if cv2.waitKey(1) & 0xff == ord('c'):
        break
cv2.destroyAllWindows()
cam.release()