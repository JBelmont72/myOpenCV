
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
Pickle Lesson       https://www.youtube.com/watch?v=eWrTSBIQess


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

print(names)
print(encodings)
# with open('train.pkl','wb')as f:
# with open('/Users/judsonbelmont/Documents/Projects/OpenCV_1/train.pkl','wb')as f:
#     pickle.dump(names,f)
#     pickle.dump(encodings,f)


# with open('/Users/judsonbelmont/Documents/Projects/OpenCV_1/train.pkl ','rb') as f:
#     names = pickle.load(f)
#     knownEncodings = pickle.load(f)
# a = pickle.load(f)
# b = pickle.load(f)
# print(a)
# print(b)