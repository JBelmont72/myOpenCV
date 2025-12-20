'''
Lesson 60 tracking faces
use Haar Cascades in OpenCV on the Raspberry Pi to find and track  faces and eyes. We show the intelligent way to find eyes, such that CPU resources are not wasted
/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/Pi_Lessons/Lesson_60_FaceTrack.py
'''
# import cv2
# from picamera2 import Picamera2
# import time
# picam2 = Picamera2()
# dispW=1280
# dispH=720
# picam2.preview_configuration.main.size = (dispW,dispH)
# picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.controls.FrameRate=30
# picam2.preview_configuration.align()
# picam2.configure("preview")
# picam2.start()
# fps=0
# pos=(30,60)
# font=cv2.FONT_HERSHEY_SIMPLEX
# height=1.5
# weight=3
# myColor=(0,0,255)

# faceCascade=cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
# eyeCascade=cv2.CascadeClassifier('./haar/haarcascade_eye.xml')


# while True:
#     tStart=time.time()
#     frame= picam2.capture_array()
#     frame=cv2.flip(frame,-1)
#     frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=faceCascade.detectMultiScale(frameGray,1.3,5)
#     for face in faces:
#         x,y,w,h=face
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
#         roiGray=frameGray[y:y+h,x:x+w]
#         roiColor=frame[y:y+h,x:x+w]
#         eyes=eyeCascade.detectMultiScale(roiGray,1.3,5)
#         for eye in eyes:
#             x,y,w,h=eye
#             cv2.rectangle(roiColor,(x,y),(x+w,y+h),(255,0,0),3)
#     cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
#     cv2.imshow("Camera", frame)

#     if cv2.waitKey(1)==ord('q'):
#         break
#     tEnd=time.time()
#     loopTime=tEnd-tStart
#     fps=.9*fps + .1*(1/loopTime)
# picam.stop()
# cv2.destroyAllWindows()

## 2d copy with comments
import cv2
from picamera2 import Picamera2
import time
picam2 = Picamera2()
dispW=1280
dispH=720
picam2.preview_configuration.main.size = (dispW,dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate=30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
fps=0
pos=(30,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(0,0,255)

faceCascade=cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
eyeCascade=cv2.CascadeClassifier('./haar/haarcascade_eye.xml')


while True:
    tStart=time.time()
    frame= picam2.capture_array()
    frame=cv2.flip(frame,-1)
    frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(frameGray,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        roiGray=frameGray[y:y+h,x:x+w]
        roiColor=frame[y:y+h,x:x+w]
        eyes=eyeCascade.detectMultiScale(roiGray,1.3,5)
        for eye in eyes:
            x,y,w,h=eye
            cv2.rectangle(roiColor,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.putText(frame,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps=.9*fps + .1*(1/loopTime)
# picam.stop()
cv2.destroyAllWindows()