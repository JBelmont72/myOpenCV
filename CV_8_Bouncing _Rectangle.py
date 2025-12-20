'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
come back to this later

cv2. is width,heigt
numpy is height ,width
Width 640
height 480
'''
# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# import time
# print(f"This is version {sys.version}")
# print(np.__version__)
# DispW=320
# DispH=240
# # DispW=640
# # DispH=480
# url = "http://192.168.1.58/stream"

# cam = cv2.VideoCapture(url)
# # cam = cv2.VideoCapture(1)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, DispW)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,DispH)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
# startTime=time.time()
# FPS=30

# while True:
#     ignore,  frame = cam.read()
#     timeDiff=startTime-time.time()
#     startTime=time.time()   ## This gives a new start time for the next loop
#     fps=int((.95 * FPS +(0.05*1/timeDiff)))
#     print(fps)
#     cv2.putText(frame,f'{str(fps)} FPS',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2 )
#     HSVframe=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     # rectangle =cv2.rectangle(frame,(int((width/2)-50+xPos),int((height/2)-50)-yPos),(int((width/2)+50+xPos),int((height/2)+50-yPos)),(0,0,255),circleThickness)
#     ROI=frame[int(DispH/2):DispH,int(DispW/2):DispW]
#     print(ROI.shape,'   ',frame.shape)
#     frame[int(DispH/2):DispH,0:int(DispW/2)]=ROI
#     print(f'(frame pixel: {frame[int(DispH/2),int(DispW/2)]}')
#     print(f'(HSV frame pixel: {HSVframe[int(DispH/2),int(DispW/2)]}')
#     # frame[DispH-(int(DispH/2)):DispH+(int(DispH/2)),DispW-int(DispW/2):DispW+int(DispW/2)]=ROI
#     print(ROI.shape)
    
#     cv2.imshow('ROI',ROI)
    
#     cv2.imshow('my WEBcam', frame)
#     # cv2.imshow('my WEBcam', HSVframe)
#     cv2.moveWindow('my WEBcam',0,0)

    
#     if cv2.waitKey(1) & 0xff ==ord('q'):
#         break
# cam.release()


# import sys
# import cv2
# print(cv2.__version__)
# import numpy as np
# import time
# print(f"This is version {sys.version}")
# print(np.__version__)

# # my ESP32S3 EYE  camera IP address
# # url = "http://192.168.1.58/stream"

# # cap = cv2.VideoCapture(url)
# cap=cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
# startTime=time.time()
# fps=20
# new_width=640
# new_height=480
# ROIwidth=60
# ROIheight=40
# if not cap.isOpened():
#     print("Failed to connect to camera stream")
#     exit(1)

# print("Connected to camera stream. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     timedelta=time.time() - startTime
#     startTime=time.time()
#     FPS =1/timedelta 
#     fps =.95*fps + 0.05 * FPS
#     # print(frame.shape)
    
#     print(int(fps) )
#     frame=cv2.flip(frame,1)
#     Rframe = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#     print(Rframe.shape)
#     ROI=frame[int((new_height -ROIheight)/2):int((new_height + ROIheight)/2),int((new_width  -ROIwidth)/2):int((new_width +ROIwidth)/2)]
#     RframeGray=cv2.cvtColor(Rframe,cv2.COLOR_BGR2GRAY)
#     RframeGrayBGR =cv2.cvtColor(RframeGray,cv2.COLOR_GRAY2BGR)
#     RframeGrayBGR[int((new_height -ROIheight)/2):int((new_height + ROIheight)/2),int((new_width  -ROIwidth)/2):int((new_width +ROIwidth)/2)]=ROI
#     cv2.imshow('ROI',ROI)
#     cv2.moveWindow('ROI',new_width,30)
    
#     cv2.putText(RframeGrayBGR,str(int(fps))+' fps',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    
#     cv2.imshow("ESP32 Camera", RframeGrayBGR)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()   
    





import sys
import cv2
print(cv2.__version__)
import numpy as np
import time
print(f"This is version {sys.version}")
print(np.__version__)

# my ESP32S3 EYE  camera IP address
# url = "http://192.168.1.58/stream"

# cap = cv2.VideoCapture(url)
cap=cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
startTime=time.time()
fps=20
new_width=640
new_height=480
ROIwidth=60
ROIheight=40
deltaX=2
deltaY=2
y =new_height
x=new_width
if not cap.isOpened():
    print("Failed to connect to camera stream")
    exit(1)

print("Connected to camera stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    timedelta=time.time() - startTime
    startTime=time.time()
    FPS =1/timedelta 
    fps =.95*fps + 0.05 * FPS
    # print(frame.shape)
    
    print(int(fps) )
    frame=cv2.flip(frame,1)
    Rframe = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(Rframe.shape)
    ROI=frame[int((new_height -ROIheight)/2):int((new_height + ROIheight)/2),int((new_width  -ROIwidth)/2):int((new_width +ROIwidth)/2)]
    RframeGray=cv2.cvtColor(Rframe,cv2.COLOR_BGR2GRAY)
    RframeGrayBGR =cv2.cvtColor(RframeGray,cv2.COLOR_GRAY2BGR)
    
    
    RframeGrayBGR[int((y -ROIheight)/2):int((new_height + ROIheight)/2),int((new_width  -ROIwidth)/2):int((new_width +ROIwidth)/2)]=ROI
    cv2.imshow('ROI',ROI)
    cv2.moveWindow('ROI',new_width,30)
    
    cv2.putText(RframeGrayBGR,str(int(fps))+' fps',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    
    cv2.imshow("ESP32 Camera", RframeGrayBGR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()   
    
