import sys
import cv2
import numpy as np
print(f"This is Python version {sys.version}")
print(np.__version__)
print(cv2.__version__)

#   HOW MANY SQUARES ACROSS. BOARD WILL BE SQUARE
# if bardize is 1000 and there are 10 squares across then each square is 100
boardSize = int(input("What size board do you want?"))
numSquares = int(input("How many squares?"))
squareSize = int(boardSize/numSquares)
darkColor = (0,0,0)
lightColor = (0,0,255)
nowColor = darkColor

while True:
    # the size of the board will be boardSize by BoardSize e.g. 1000 x 1000
    x = np.zeros([boardSize,boardSize,3], dtype=np.uint8)
    #if grayscale we are done but for color we have to give an array for each pixel so add the 3 above
    for row in range(0,numSquares,1):
        for columns in range(0,numSquares,1):
            x[int(squareSize * row) :int(squareSize * (row +1 )),int(squareSize*columns):int(squareSize*(columns + 1)) ] = nowColor
            if nowColor == darkColor:
                nowColor = lightColor
            else:
                nowColor = darkColor
        if nowColor == darkColor:
            nowColor = lightColor
        else:
             nowColor = darkColor        
    cv2.imshow('my CheckerBoard', x)
    # cv2.moveWindow('my CheckerBoard',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

