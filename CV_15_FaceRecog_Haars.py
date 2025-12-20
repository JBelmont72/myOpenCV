'''
show a simple method to detect faces in a WEB cam frame using openCV and Haar Cascades. 
https://youtu.be/xWo90YfgYeg
facial recogniton is identify a face is precent, facial detection is identifying a person

python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
in venv folder, selected lib/site_packages then select "cv2" then 'Data"
which will show the haarcascade
'''
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is Python version {sys.version}")
print(np.__version__)
import time

width=int(400)
height=int(300)
# width=int(640)
# height=int(360)

# my ESP32S3 EYE  camera IP address
url = "http://192.168.1.58/stream"

cam = cv2.VideoCapture(url)

# cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
#   /Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/haar/haarcascade_frontalface_default.xml
# the following three ways of loading the haarcascade work
faceCascade = cv2.CascadeClassifier('/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/haar/haarcascade_frontalface_default.xml')
# faceCascade =cv2.CascadeClassifier('/Users/judsonbelmont/Documents/Projects/OpenCV_1/.venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
startTime = time.time()
FPS=15
while True:
    
   
    ignore,  frame = cam.read()
    timeDiff = time.time() - startTime
    startTime= time.time()
    fps=1/timeDiff
    fps= 0.95 * FPS + 0.05 * fps
    print(int(fps))
    flipFrame =cv2.flip(frame,1)
    frame=cv2.resize(flipFrame,(400,300))
    frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(frameGray,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()