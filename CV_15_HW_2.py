
'''
show a simple method to detect faces in a WEB cam frame using openCV and Haar Cascades. 
https://youtu.be/xWo90YfgYeg
facial recogniton is identify a face is precent, facial detection is identifying a person

python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
in venv folder, selected lib/site_packages then select "cv2" then 'Data"
which will show the haarcascade
this was a comment- addressed the eye window differently than Paul
applied a slightly different approach to setting the eye rectangles. 
cv2.rectangle(frame, (x+xeye,y+yeye), (x+xeye+weye,y+yeye+heye),(0,0,255), 2)

'''
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is Python version {sys.version}")
print(np.__version__)
import cv2.data
import time
width=  640
height= 360
# my ESP32S3 EYE  camera IP address
url = "http://192.168.1.58/stream"

cam = cv2.VideoCapture(url)
# cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

# the following three ways of loading the haarcascade work
faceCascade = cv2.CascadeClassifier('/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/haar/haarcascade_frontalface_default.xml')
# faceCascade =cv2.CascadeClassifier('/Users/judsonbelmont/Documents/Projects/OpenCV_1/.venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/haar/haarcascade_eye.xml')

fps  = 15
FLTfps = 15
timeStamp = time.time()
while True:
       
    ignore,  frame = cam.read()
    flipFrame = cv2.flip(frame,1)
    frame=cv2.resize(flipFrame,(400,300))
    frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(frameGray,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
 #this part is different from CV_15_HW_1.py
        frameROI= frame[y:y+h,x:x+w]
        frameROIGray= cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
        eyes =eyeCascade.detectMultiScale(frameROIGray,1.3,5)
        # frameROIGray2BGR = cv2.cvtColor(frameROIGray,cv2.COLOR_GRAY2BGR) 
        #creating a rectangle in the ROI which is the face and create a rectangle in the ROI     
        # the rectangle will be for a slice of the frame and then pass the eye variables into it
        for eye in eyes:
            xEYE,yEYE,wEYE,hEYE = eye
            cv2.rectangle(frame[y:y+h,x:x+w],(xEYE,yEYE),(xEYE+wEYE,yEYE+hEYE),(0,0,255),3)

    looptime = time.time() - timeStamp
    timeStamp =time.time()
    fps = 1/looptime
    FLTfps =FLTfps *.95 +fps * 0.05
    myText =(str(int(FLTfps))+ ' fps')
    cv2.rectangle(frame,(30,10),(140,50),(125,125,0),-1)
    cv2.putText(frame,(myText),(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    
    
    
    
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()