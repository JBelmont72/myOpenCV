'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
'''
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
width=640
height=360
evt =0
xPos =0
yPos =0
Thickness = 2
CircleSize =25
Color =[0,0,255]

def mouseClick(event,xPos,yPos,flags,params):
    global pnt1
    global pnt2
    global evt
    if event == cv2.EVENT_LBUTTONDOWN:        
        
        print('Left Mouse Down was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        evt = event
        pnt1 = (xPos,yPos)
    if event == cv2.EVENT_LBUTTONUP:
        
        print('Left Mouse Up was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        pnt2 = (xPos,yPos)
        evt = event
    if event  == cv2.EVENT_RBUTTONUP:
        
        print('Right Mouse Up was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        evt = event         #evt will be set to 5 and will not put the circle on the frame. thus the circle wil disappear
   



cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow('my WEBcam')
cv2.setMouseCallback('my WEBcam', mouseClick)
while True:
    ignore,  frame = cam.read() # left button down is 1 and left button up is 4
    if evt  == 4:
        cv2.rectangle(frame,pnt1,pnt2,Color,Thickness) 
        #up to this point we have a rectangle, now we have to create our region of interest.
        ROI =frame[  pnt1[1]:pnt2[1],pnt1[0:pnt2[0]]]
        cv2.imshow('my ROI',ROI)
        cv2.moveWindow('my ROI',0,int(height+30))     
        cv2.imshow('my WEBcam', frame)
        cv2.moveWindow('my WEBcam',0,0)
    if evt == 5:
        cv2.destroyWindow('my ROI')
        evt = 0
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()