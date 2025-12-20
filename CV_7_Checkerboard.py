'''this is the second part of lesson 7, just making the board
complete project is CV_7_HW_CheckerBoard.py
'''
import sys
import cv2
import numpy as np
print(f"This is version {sys.version}")
print(np.__version__)
print(cv2.__version__)
width=640
height=360
#   HOW MANY SQUARES ACROSS. BOARD WILL BE SQUARE
# if bardize is 1000 and there are 10 squares across then each square is 100
boardSize = int(input("What size board do you want?"))
numSquares = int(input("How many squares?"))
squareSize = int(boardSize/numSquares)

while True:
    # the size of the board will be boardSize by BoardSize e.g. 1000 x 1000
    x = np.zeros([boardSize,boardSize,3], dtype=np.uint8)
    #if grayscale we are done but for color we have to give an array for each pixel so add the 3 above
    cv2.imshow('my CheckeBoard', x)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

