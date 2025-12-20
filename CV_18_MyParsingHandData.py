'''
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
https://developers.google.com/mediapipe/solutions/guide
 Study this video  https://www.youtube.com/watch?v=qAw5tuYgVec
from 'code with price'

about using it on Python https://github.com/google/mediapipe/issues/2765

Plan_ watch videos by 'coding with Prince'
the https://developers.google.com/mediapipe/framework/getting_started/install
mdiapipe homepage is specific aboput using pythone 3.6 if want to use tensorflow.
Create a new Project with pythone 3.6 following the instructions in the Mediapipe homepage.

this is my version of parsing hand data
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
important note.  if I just activate the virtual environment , it goes to venv on python interpreteor
If I want to stay in the pyenv  then I MUST use the python -m venv .venv as the first command when I activate the virtual environment with source venv/bin/activate

Thank you. I am also working on a MAC (3.9.6 python_Mediapipe says python is  supported  up to 3.10). Running into a MAC related problem.  My webcam launches but the data is being directed to STDER which is apparently a MAC thing. Thus, the data points are not shown either by print() or on the webCam. The specific message is 'WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1707495418.390307 '  found a GitHub link https://github.com/abseil/abseil-py/issues/141 that addresses it but not in a way I can understand. Hoping you have crossed that hurdle. Any insights appreciated.

THis video is about python logging modules  https://www.youtube.com/watch?v=pxuXaaT1u3k

'''
import mediapipe as mp
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
width= 1280  #640
height= 720  #360
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
hands=mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)
'''
The problem with your code lies in line 11-12:
self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
The class Hands() takes in 5 arguments, but it seems like you have missed the "complexity" parameter.

https://stackoverflow.com/questions/73409691/how-to-fix-an-opencv-ai-hands-critical-code-error-in-python
'''


# hands =mp.solutions.hands.Hands(False,2,1,1)  # the false tells the method that the image is NOT static, but will move
# he said that there are 'solutions' with lower case hands and uppercase Hands
# the 2 is the number of hands   the .5 and .5 is level of confidence
# to draw the hands follows:
mpDraw = mp.solutions.drawing_utils     # this is to annotate the frame with the data, if want to analyze the frame TO DRAW HANDS  use the mp.solutions.hands.Hands(   , , )

while True:
    ignore,  frame = cam.read()
    frame = cv2.resize(frame,(width,height))
    frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # the arrays of hand data come back in results.multi_hand_landmarks.  if = NOne +Go look for more hands.If not nothing then look further
    results = hands.process(frameRGB)       # call the HANDS object and process it in RGB
    if results.multi_hand_landmarks != None:
        for handLandMarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handLandMarks)
            print(handLandMarks)
    
    
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()