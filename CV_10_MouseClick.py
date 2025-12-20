'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
'''
import sys
import cv2
print(f'OpenCV version: {cv2.__version__}')
import numpy as np
print(f"This python is version {sys.version}")
print(f'Numpy version is: {np.__version__}')
evt =0
xPos =0
yPos =0
Thickness = 2
CircleSize =25
Color =[0,0,255]

evt =0

def mouseClick(event,xPos,yPos,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:        
        global evt
        global pnt
        print('Left Mouse Down was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        evt = event
        pnt = (xPos,yPos)
    if event == cv2.EVENT_LBUTTONUP:
        global evt 
        
        print('Left Mouse Up was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        evt = event
    if event  == cv2.EVENT_RBUTTONUP:
        global evt
        print('Right Mouse Up was: ',event)
        print('at Postion',xPos,' x  ',yPos, ' y')
        evt = event         #evt will be set to 5 and will not put the circle on the frame. thus the circle wil disappear
          
    
# want to put a circle when the button is clicked.  Have to put it in the while loop because the cicle would be present only for One frame if it was up here.    
width=640
height=360
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
cv2.namedWindow('my WEBcam')
cv2.setMouseCallback('my WEBcam', mouseClick)
while True:
    ignore,  frame = cam.read()
    if evt == 1 or evt == 4:
        cv2.circle(frame,pnt,CircleSize,Color,Thickness)       
        cv2.imshow('my WEBcam', frame)
        cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()