'''
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('venv':venv)  version in the interpreter!!
we are going to use pickle to save the files
Pickle lesson   https://www.youtube.com/watch?v=eWrTSBIQess

one persons solution hard to read for me    https://www.youtube.com/watch?v=AOZCG8lcpqE
'''
import pickle
import os
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
import face_recognition as FR
width=640
height=360
encodings = []
names = []
# in windows use a double \\  after the 'C:\\
imageDir ='/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages/known'
'''
os.walk it lists where it is( the root) 1st time will be imageDir, then all the folders in the directory and then all the folders inside the first foleder and the folders within each folder within each folder until it gets down to the files in each folder.
We want to return the 'root' (or path to the the folder/directory we want to start at)
Returns Root directory, directories(or Folders), and all the files
'''
for  root,dirs,files in os.walk(imageDir):
    # print('My working folder(root): ',root)
    # print('\nDIRS in root: ',dirs)
    # print('\nMy FIles in root: ',files)
    for file in files:
        # print('Person Name(with .jpg) is: ',file)
        fullFilePath = os.path.join(root,file)
        print(fullFilePath)
        myPicture = FR.load_image_file(fullFilePath)
        # want to encode one face in the one photo from the myPicture
        # this will encode the file in the 'myPicture'
        encoding =FR.face_encodings(myPicture)[0]       
        name = os.path.splitext(file)[0]
        print(name)
        encodings.append(encoding)
        names.append(name)
# with open('train.pkl','wb')as f:
with open('/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/train.pkl','wb')as f:
    pickle.dump(names,f)
    pickle.dump(encodings,f)
        
        
    #     myPicture = FR.load_image_file(fullFilePath)
    #     myPicture = cv2.cvtColor(myPicture,cv2.COLOR_RGB2BGR)
    #     cv2.imshow(name,myPicture)
    #     cv2.moveWindow(name,0,0)
        
    # cv2.waitKey(2500)
    # cv2.destroyAllWindows
            
'''    
    #my solution to concatenate the file name with the path
    for file in files:
        print('Person name is: ,file)
        filePath = (root+'/'+str(file))
        print('')
        print(filePath)
    # want to get the path to each file and also the file names without the .jpg
'''    
# cam = cv2.VideoCapture(0)
# # cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
# cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

# while True:
#     ignore,  frame = cam.read()
#     cv2.imshow('my WEBcam', frame)
#     cv2.moveWindow('my WEBcam',0,0)
    # if cv2.waitKey(1) & 0xff ==ord('q'):
    #     break
# cam.release()