'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
Notes for project of having a ROIframe which will float around in the frame in x and y axis.
The full frame has a width which is the # of columns. 
THe height is the # of rows.
The snippet box has 'snipH'  want to and 'snipW'
Identify the center of the box with designations boxCR and 'boxCC' for box center column and box center Row
THe center of the box will be row 1/2 * Height and column   is 1/2 * width
Conditions we wnat to address are when the edges of the snip hit the edges of the frame.
the right border of the snip is 'boxCC +(snipW / 2)'
the left border of the snip is 'boxCC -(snipW / 2)'
the top of the box is 'boxCR -(snipH / 2)'
the bottom of the box is 'boxCR  + (snipH / 2)'
Condtions where you change direction.
    when boxCC +(snipW/2) >= frame width
    when boxCC -((snipW/2) <= 0
    when boxCR +(snipH/2)   >=height
    when boxCR -(snipH/2)  <= 0


'''
# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# print(f"This is version {sys.version}")
# print(np.__version__)
# # width=320
# # height=240
# # width=640
# # height=360
# width=1280
# height=720
# snipW = 120
# snipH = 60
# boxCR =int(height/2)
# boxCC = int(width/2)
# deltaRow =6
# deltaColumn = 4
# # url = "http://192.168.1.58/stream"

# # cam = cv2.VideoCapture(url)
# cam = cv2.VideoCapture(0)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

# while True:
#     ignore,  frame = cam.read()
#     frameROI=frame[ int(boxCR -snipH/2):int(boxCR +snipH/2), int(boxCC-snipW/2):int(boxCC + snipW/2) ]
#     #for this exercise we are going to convert the whole frame to gray
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)     # now the ROI frame is still grayscale but has a three digit tuple
    
#     frame[ int(boxCR -snipH/2):int(boxCR +snipH/2), int(boxCC-snipW/2):int(boxCC + snipW/2) ]=frameROI
#     #NOTE IMPORTANT I had to set the 'frame' which is color equal to the frameROI
#     # the frame has been turned GRAY(with 3 digit tuple present )
#     # without this line the frameROI is off to the side.This line will place it in the center  as well!!
#     # next have to make the box move. everything is defined in relation to the center position
#     boxCR =boxCR + deltaRow
#     boxCC =boxCC + deltaColumn
#     if int(boxCR - (snipH/2)) <=0 or int(boxCR+ snipH/2) >= height:
#         deltaRow = deltaRow*(-1)
#     if int(boxCC - snipW/2) <=0 or int(boxCC + snipW/2)>= width:
#         deltaColumn = deltaColumn *(-1)
    
    
#     cv2.imshow('my ROI',frameROI)
#     cv2.moveWindow('my ROI',width,0)
#     cv2.imshow('my WEBcam', frame)
#     cv2.moveWindow('my WEBcam',0,0)
#     if cv2.waitKey(1) & 0xff ==ord('q'):
#         break
# cam.release()

import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
# width=320
# height=240
width=640
height=360
# width=1280
# height=720
BW = 100
BH = 60

deltaRow =3
deltaColumn = 3
# url = "http://192.168.1.58/stream"

# cam = cv2.VideoCapture(url)
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
BoxColor= (255,0,0)
tLC=width/2
tLR=height/2
while True:
    ignore,  frame = cam.read()
    print(frame.shape)
    frameFlip=cv2.flip(frame,1)
    ResizeFrame=cv2.resize(frameFlip,(width,height),interpolation=cv2.INTER_CUBIC)
    box=cv2.rectangle(ResizeFrame,(int(tLC),int(tLR)),(int(tLC+BW),int(tLR+BH)),BoxColor,-1)
    tLC+=deltaColumn
    tLR += deltaRow
    if tLC+deltaColumn <= 0 or (tLC +BW+deltaColumn)>= width-1:
        deltaColumn = (-1)* deltaColumn
    
    if tLR+deltaRow <= 0 or (tLR+BH+deltaRow)>= height-1:
        deltaRow =(-1) * deltaRow
    
    # #for this exercise we are going to convert the whole frame to gray
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)     # now the ROI frame is still grayscale but has a three digit tuple
    
    # frame[ int(boxCR -snipH/2):int(boxCR +snipH/2), int(boxCC-snipW/2):int(boxCC + snipW/2) ]=frameROI
    #NOTE IMPORTANT I had to set the 'frame' which is color equal to the frameROI
    # # the frame has been turned GRAY(with 3 digit tuple present )
    # # without this line the frameROI is off to the side.This line will place it in the center  as well!!
    # # next have to make the box move. everything is defined in relation to the center position
    
    cv2.imshow('my WEBcam', ResizeFrame)
    cv2.moveWindow('my WEBcam',0,height)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()