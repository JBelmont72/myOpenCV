'''CV_17.py  used os.walk to parse through path(root directory), folders(directories), and files and put them in a useabel form_
the form desired is the path of each photo and the name (sans '.jpg')
python -m venv .venv
source .venv/bin/activate

use the 3.9.6 ('pyenv':venv)  version in the interpreter!!
'''
import os
import sys
import cv2
print(cv2.__version__)
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
width=640
height=360
imageDir ='/Users/judsonbelmont/Documents/Projects/OpenCV_1/demoImages'
'''
os.walk it lists where it is( the root) 1st time will be imageDir, then all the folders in the directory and then all the folders inside the first foleder and the folders within each folder within each folder until it gets down to the files in each folder.
We want to return the 'root' (or path to the the folder/directory we want to start at)
Returns Root directory, directories(or Folders), and all the files
'''
for  root,dirs,files in os.walk(imageDir):
    print('My working folder(root): ',root)
    print('\nDIRS in root: ',dirs)
    print('\nMy FIles in root: ',files)
    for file in files:
        print('Person Name is(with .jpg): ',file)
        fullFilePath = os.path.join(root,file)
        print(fullFilePath)
        name = os.path.splitext(file)[0]
        print(name)
            
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