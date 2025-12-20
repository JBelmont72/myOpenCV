'''Create Rectangles, Circles, Text
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

import time #added for this sketch
width=640
height=360
myThicknessCircle = 2
RectangleColor =[225,0,0]
CircleColor = [0,0,255]
myRadius = 20
myThicknessRec = 5
myText ='Success is so Sweet'
fontH = 1  #font height
fpsFilter = 30
myFont =  cv2.FONT_HERSHEY_COMPLEX
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

tLast = time.time() #added for this sketch
time.sleep(.1)  # need a delay because  dt and time.time () are the same number the first time through.
'''
 if it takes x time to process a frame, then FPS = 1/x  here the xtime is dT
 '''   
    
while True:
    dT =time.time() -tLast
    # print(dT)
    
    tLast =time.time()
    fps = 1/dT
    print(fps)
    fpsFilter=(fpsFilter * .97 )+ fps * .03
    ignore,  frame = cam.read()
    # 40 up and down from center and 320 -60 320+60
    # frame[140:220,260:380]=[225,0,0]
    frame[int((height/2)-40):int((height/2)+40),int((width/2)-60):int((width/2)+60)]= RectangleColor
    # rectangle =cv2.rectangle(frame,(260,140),(380,220),(0,255,0),5)
    rectangle =cv2.rectangle(frame,(int((width/2)-60),int((height/2)-40)),(int((width/2)+60),int((height/2)+40)),CircleColor,myThicknessRec)
    
    circle = cv2.circle(frame,(int(width/2),int(height/2)),myRadius,(0,0,255),myThicknessCircle)
    # arguemnts frame, a tuple of the center, radius, color,thickness (if -1 will be solid)
    cv2.putText(frame,myText,(120,80),cv2.FONT_HERSHEY_COMPLEX,fontH,(255,0,0),2)
    cv2.rectangle(frame,(10,5),(130,45),(255,0,255),-1)     #magenta
    cv2.putText(frame,(str(int(fpsFilter))+' FPS'), (10,30),myFont,1,(0,0,0),1)
    
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()