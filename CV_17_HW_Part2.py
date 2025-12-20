'''THis is part two of HW for CV_17. NOte that we copied the program from the HW for laesson 16 where we were 
training on faces we loaded into the program by their individual paths.
Here we will treain it on all the photos in the 'known' folder.
learn how to not only detect faces on a webcam, 
but also recognize who the person is. We start by training our model with known faces,
and then find and identify those faces in other pictures.
His HW 17 which starts with solution from HW for 16
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('pyenv':venv)  version in the interpreter!!
Pickle Lesson       https://www.youtube.com/watch?v=eWrTSBIQess
a participants homework using the WebCam  https://www.youtube.com/watch?v=BQV-_AYJRnE
'''
import pickle
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
import face_recognition as FR
width=640
height=360
cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

font=cv2.FONT_HERSHEY_SIMPLEX
 ## the below was from CV_16 homework wher ewe entered PATHS
# judsonFace=FR.load_image_file('/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/known/Judson Belmont.jpg')
# faceLoc=FR.face_locations(judsonFace)[0]
# judsonFaceEncode=FR.face_encodings(judsonFace)[0]
 
 
# nancyFace = FR.load_image_file('/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/known/Nancy Pelosi.jpg')
# faceLoc=FR.face_locations(nancyFace)[0]
# nancyFaceEncode=FR.face_encodings(nancyFace)[0]
 
# penceFace=FR.load_image_file('/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/known/Mike Pence.jpg')
# faceLoc=FR.face_locations(penceFace)[0]
# penceFaceEncode=FR.face_encodings(penceFace)[0]
 
# knownEncodings=[judsonFaceEncode,nancyFaceEncode,penceFaceEncode]
# names=['Judson Belmont','Nancy Pelosi','Mike Pence']
'''want to unpickle the files that we pickled into Part 1 of CV_17_HW
have to load here the files in the same order that we 'dumped' the files in part one
/Users/judsonbelmont/Documents/Projects/OpenCV_1/train.pkl
can name the pickled files whatever but want them to agree with the program we are moving them inot 
'''
# with open('train.pkl ','rb') as f:
with open(' /Users/judsonbelmont/Documents/Projects/OpenCV_1/train.pkl','rb') as f:
    names = pickle.load(f)
    knownEncodings = pickle.load(f)

while True:
    ignore, unknownFace = cam.read()
# he renamed the RGB2BGR as unknownFace but I chose to use unknownFaceBGR to keep myself straight\    
# unknownFace=FR.load_image_file('/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/unknown/uJB_Doug_Karl.jpg')
    unknownFaceBGR=cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)
    # unknownFaceBGR=cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)
    faceLocations=FR.face_locations(unknownFaceBGR)
    # faceLocations=FR.face_locations(unknownFace)
    unknownEncodings=FR.face_encodings(unknownFace,faceLocations)
 
    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        top,right,bottom,left=faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFaceBGR,(left,top),(right,bottom),(255,0,0),3)
        name='Unknown Person'
        matches=FR.compare_faces(knownEncodings,unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name=names[matchIndex]
        cv2.putText(unknownFaceBGR,name,(left,top),font,.75,(0,0,255),2)
 
    cv2.imshow('My Faces',unknownFaceBGR)
     

    if cv2.waitKey(1) & 0xff ==ord('q'):
            break
cam.release()
cv2.destroyAllWindows