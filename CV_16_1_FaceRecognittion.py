'''CV_16_1 this FINDS the face and displays it in a box. 
CV_16_2 will encode the face
Note - we obtained the images from github/com/mcwhorpj
train openCV to detect and recognize faces. find and identify the ones it has been trained for.
(venv) judsonbelmont@Judsons-Air OpenCV_1 % python -m pip install
 face_recognition == 1.2.3
need to do first
1 download Python 3.6.8 64bit is what was recommended, i have 3.9.6
2 create a virtual environment 
3 download Cmake 3.21
4 download dlib  (i could not the 19.7 version so downloaded latest)
5 download face_recognition 1.2.3
python -m venv .venv
source .venv/bin/activate
USE 3.9.6 pyenv!!!!!!!
use the 3.9.6 ('venv':venv)  version in the interpreter!!
Note that the array for the face is an array within in array so as to accommodate multiple faces.
Note that the digits are first the Right SIde, then the Top, then bottom and then the left side!

note: if going to show images in OpenCV then have to be BGR!!
The Jpg files are RBG
so need to convert the RGB to gbr so we can show them in OpenCV
'''
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
import face_recognition as FR
print(f"Face Recogniton Version: {FR.__version__}")
font = cv2.FONT_HERSHEY_SIMPLEX
donFace = FR.load_image_file('/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/known/Donald Trump.jpg')
faceLoc =FR.face_locations(donFace)[0]  #the zero is so we detect the first face
topLoc  = faceLoc[0]
rightLoc   = faceLoc[1]
bottomLoc= faceLoc[2]
leftLoc = faceLoc[3]

# he has top,right,bottom,left = faceLoc  
# top left = (leftLoc,topLoc)
# bottom right = (rightLoc,bottomLoc)
print(faceLoc)
print(faceLoc[0])
cv2.rectangle(donFace,(leftLoc,topLoc),(rightLoc,bottomLoc),(0,0,255),2,)
donFaceBGR=cv2.cvtColor(donFace,cv2.COLOR_RGB2BGR)
cv2.imshow('My Image',donFaceBGR)
cv2.waitKey(5000)


'''
# direct copy. https://toptechboy.com/tag/artificial-intelligence/
import cv2
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX
width=640
height=360
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
 
paulFace=FR.load_image_file('C:/Users/Valued Customer/Documents/Python/demoImages/known/Paul McWhorter.jpg')
faceLoc=FR.face_locations(paulFace)[0]
paulFaceEncode=FR.face_encodings(paulFace)[0]
 
gavFace=FR.load_image_file('C:/Users/Valued Customer/Documents/Python/demoImages/known/Gavriella Joy.jpg')
faceLoc=FR.face_locations(gavFace)[0]
gavFaceEncode=FR.face_encodings(gavFace)[0]
 
knownEncodings=[paulFaceEncode,gavFaceEncode]
names=['Paul McWhorter','Gavriella Joy']
 
while True:
    ignore,  unknownFace = cam.read()
 
    unknownFaceRGB=cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)
    faceLocations=FR.face_locations(unknownFaceRGB)
    unknownEncodings=FR.face_encodings(unknownFaceRGB,faceLocations)
 
    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        top,right,bottom,left=faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFace,(left,top),(right,bottom),(255,0,0),3)
        name='Unknown Person'
        matches=FR.compare_faces(knownEncodings,unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name=names[matchIndex]
        cv2.putText(unknownFace,name,(left,top),font,.75,(0,0,255),2)
 
    cv2.imshow('My Faces',unknownFace)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
'''