'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
'''
# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# print(f"This is version {sys.version}")
# print(np.__version__)
# # width=640
# # height=360
# width=320
# height=240
# ROIwidth =100
# ROIheight =70
# # width = 1280
# # height =720
# url = "http://192.168.1.58/stream"

# cam = cv2.VideoCapture(url)
# # cam = cv2.VideoCapture(0)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

# while True:   
#     ignore,  frame = cam.read()
#     # frameROI = frame[int((height/2)-ROIheight/2):int((height/2)+ROIheight/2),int((width/2)-ROIwidth/2):int((width/2)+ROIwidth/2)]
#     frameROI = frame[int((height/2)-ROIheight):int((height/2)+ROIheight),int((width/2)-ROIwidth):int((width/2)+ROIwidth)]
#     frameROIGray =cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
#     frameROIGray_BGR =cv2.cvtColor(frameROIGray,cv2.COLOR_GRAY2BGR)
#     # cv2.resizeWindow('frameROIGray_BGR',(int(ROIwidth/3),int(ROIheight/3))) #save this and look at later
#     frame[0:int(2*ROIheight),0:int(2*ROIwidth)]=frameROIGray_BGR    #note that I had to multiply by 2 both the height and width
    
#     cv2.imshow('frameROIGray_BGR',frameROIGray_BGR)
#     cv2.moveWindow('frameROIGray_BGR',int(width),int(height+30))
#     # cv2.moveWindow('frameROIGray_BGR',int((width/2)+ROIwidth),int(height+30))
#     cv2.imshow('my Gray ROI',frameROIGray)
#     # cv2.moveWindow('my Gray ROI',int((width/2)-ROIwidth),int(height+30))
#     cv2.moveWindow('my Gray ROI',int(width),int(height+30))
#     cv2.imshow('my ROI',frameROI)
#     cv2.moveWindow('my ROI',0,int(height+30))
#     cv2.imshow('my WEBcam', frame)
#     cv2.moveWindow('my WEBcam',0,0)
#     # cv2.imshow('Window Frame',frameROIGray_BGR)   #this pastes the gray_BGR frame onto the mainframe but just as an only
#     # cv2.moveWindow('Window Frame',30,0)
    
#     if cv2.waitKey(1) & 0xff ==ord('q'):
#         break
# cam.release()



## 16 dec 2025 I created a ROI turned it gray then made it gray to bgr to make the array fit and moved it back to center of image
import sys      #worked has the gray ROI in center of image
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
# width=640
# height=360
width=320
height=240
ROIwidth =100
ROIheight =70
# width = 1280
# height =720
url = "http://192.168.1.58/stream"

cam = cv2.VideoCapture(url)
# cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

while True:   
    ignore,  frame = cam.read()
    frameROI =frame[int(height/2 -ROIheight/2):int(height/2 +ROIheight/2),int(width/2 -ROIwidth/2):int(width/2 +ROIwidth/2)]
    frameROIGray=cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
    GrayRoiBackToFrame=cv2.cvtColor(frameROIGray,cv2.COLOR_GRAY2BGR)
    print(GrayRoiBackToFrame.shape)
    frame[int(height/2 -ROIheight/2):int(height/2 +ROIheight/2),int(width/2 -ROIwidth/2):int(width/2 +ROIwidth/2)]=GrayRoiBackToFrame
    


    cv2.imshow('frameROI',frameROI)
    cv2.moveWindow('frameROI',0,(int(height*1.5)))
    cv2.imshow('frame',frame)
    cv2.moveWindow('frame',0,40)
    
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()