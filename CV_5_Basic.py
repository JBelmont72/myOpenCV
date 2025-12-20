'''
python -m venv .venv
source .venv/bin/activate

'''

'''
use the 3.9.6 ('venv':venv)  version in the interpreter!!

THis is important 29 Nov 2025 /Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/.venv/lib/python3.9/site-packages/cv2
this was changed from
if i run into trouble , maybe i will go to this file and change it back  to True for the 4.9.0.80
opencv_version = "4.9.0.80" This is the version  not 4.11!!
contrib = False
TO:
opencv_version = "4.11.0.86"
contrib = True
headless = False
rolling = False
ci_build = True


'''
import sys
import cv2
print(cv2.__version__)
import numpy as np


print(f"This is version {sys.version}")
print(np.__version__)

cam=cv2.VideoCapture(0)
while True:
    ignore,  frame = cam.read()
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()


