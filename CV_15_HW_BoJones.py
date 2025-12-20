# HOMEWORK Assignment
    # 1.  Find a face on the WEBcam and display a BLUE box around it
    # 2.  Display the frames / second for JUST the framed face
    # 3.  And add detection of the eyes and display a RED box around each eye
    # 4.  Display the new frame rate / second with ALL of the detected items

import cv2
import time
import os

print('cv2 Version is: ',cv2.__version__)

#**********************************************************
# declare variables and set them to initial values
global x,y,w,h

width=640
height=360
numFrames = 3
REDnumFrames = 4
fpsUpperLeft = (4, 2)
fpsLowerRight = (152,36)
veryUpperLeft = (6,28) 
REDveryUpperLeft = (6,58)             
myFont =  cv2.FONT_HERSHEY_DUPLEX
myBlue = (255,0,0)
myGreen = (0,255,0)
myRed = (0,0,255)
fontH1 = 1
fontT1 = 1

#**********************************************************

#**********************************************************
# use openCV commands to set up program to display video at desired specification
cam=cv2.VideoCapture(0)
# cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

#**********************************************************

#**********************************************************
# create an object for use with the haarcascade models

faceCascade = cv2.CascadeClassifier('/Users/judsonbelmont/Documents/Projects/OpenCV_1/haar/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/Users/judsonbelmont/Documents/Projects/OpenCV_1/haar/haarcascade_eye.xml')

# NOTE - it is okay to use the relative path, but it is better practice
# to use the full path.  The PATH is listed INSIDE the single ' ' quotes

# NOTE 2 THE FULL PATH IS actually:
#   'C:\Users\bobjo\Documents\python\pyAI3.6\haar\haarcascade_frontalface_default.xml'
# when you DO use the full path be sure to add the 'r' character before
# the quotes.  
# Here is the explanation from python.org
# An 'r' preceding a string denotes a raw, (almost) un-escaped string. 
# The escape character is backslash, that is why a normal string will not work as a 
# Windows path string. The lower case r prefix on strings stands for “raw strings"

#**********************************************************

#**********************************************************
while True:
    ignore,  frame = cam.read()         # read a frame
    fps = 10
    REDfps = 10


#**********************************************************
# define a new variable frameGray and make it a BGR2GRAY grayscale image based on
# our frame variable  
    frameGray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#**********************************************************
#create a variable named faces and set it equal to faceCascade.detectMultiScale(frameGray,1.3,5)

    faces = faceCascade.detectMultiScale(frameGray,1.3,5)
    eyes  = eyeCascade.detectMultiScale(frameGray,1.3,5)


# We are searching for ANY faces found in frameGray.  The other parameters passed relate to 
# a sensitivity scale regarding finding face(s) and then Paul recommends the final 
# parameter setting as 5.  Feel free to google cv2.faceCascade.detectMultiScale() to 
# learn more about the parameters.
# the actual FULL function call syntax is:
# objects = cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]
    # during program development for the lesson we display the data in faces using the
    # print(faces) command to output the array contents to the terminal - the actual values of the 
    # x, y, w, h data for the faces found on the frame.
    # (x,y) is a tuple that describes the upper left corner of an imaginary box around the face
    # (x+w, y+h) is tuple that describes the lower right corner of box
    
#**********************************************************
# generate a for loop to search through all faces found in the faces array
# and draw a rectangle around the found face(s)

    for face in faces:
        tStart=time.time()  # start the timer at the start of the loop

            # create 4 variables to receive the parameters passed out of the faces
            # array.  These parameters are passed in the form of x,y,w,h as explained in the 
            # previous comment. For this first run through the program lesson we printed these
            # values using the command:
            # print('x = ', x, 'y = ', y,'width = ', w, 'height = ', h)
            # display the actual values from the face array.
        x,y,w,h = face

        y = y + 4       # enlarge the rectangle to be taller than wide
        h = h + 12      # enlarge the rectangle to be taller than wide
                        # I added this to improve the esthetics of the
                        # view of the face
        
        cv2.rectangle(frame,(x,y),(x+w , y+h),myBlue,2)
            # use the cv2.rectangle() function we've used in previous lessons
            # and pass in the name of the frame where we want to PLACE the rectangle
            # and then also a tuple to describe upper left corner
            # and then a second tuple to describe the lower right corner
            # of the rectangle
            # we pass another 3 value tuple to describe the BGR color for the rectangle
            # and the last parameter describes the line thickness in pixels

#**********************************************************

#**********************************************************
# This segment is a loop timer from a lesson for calculating frames per second with
# a raspberry Pi camera.
# That video can be found at: https://youtu.be/vzuBc7uoCrw
# The timer uses a Low Pass Filter process to normalize the data.  Paul gives a great description
# of how a programmed Low Pass Filter provides data that eliminates any wild swings in values.
# THAT video can be found at: https://youtu.be/IKgqDzfKNW0

        tEnd=time.time()
        loopTime = tEnd - tStart
        fps= ((.9 * fps) + (.1 * loopTime))
        fps= int(fps)

#**********************************************************
# Display the fps with formatting 

        # cv2.putText(frame,str('{0:0.2f}'.format(fps) +' FPS'), veryUpperLeft,myFont,fontH1,myBlue,fontT1)
        cv2.putText(frame,(str(fps) +' FPS'), veryUpperLeft,myFont,fontH1,myBlue,fontT1)

#**********************************************************
# create a Region Of Interest from just the FACE region detected on the frame

        frameROI = frame[y:y+h,x:x+w] 
        frameROI = cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
        eyes = eyeCascade.detectMultiScale(frameROI)

#**********************************************************
# look for any eyes in the Region Of Interest
        for eye in eyes:
            REDtStart=time.time()  # start the timer at the start of the loop
            ex,ey,ew,eh = eye

#**********************************************************
# Now put rectangles around each eye found.

# It is important to remember that we want to frame the eyes INSIDE the face box.  So we 
# set the location based on the ARRAY locations in frame that describe the box bounding corners
# This is why we use array notation to describe where the eye frames will placed.
# This notation to describe the rectangle location for the face is:

# frame[y:y+h,x:x+w]
# ARRAYS COME IN ROWS AND COLUMNS  So to set the location for the UPPER LEFT corner of the face frame
# We use the ROW y:y+h to describe the upper left and lower left locations of the frame
# And then use x:x+w to describe the upper right and lower right locations of the frame
# THEN we can use the information provided by eyes to the loop that loads the ex,ey,ew,eh variables 
# so we can perform our calculations and display the red box around the eye(s)

            cv2.rectangle(frame[y:y+h,x:x+w],(ex,ey),(ex+ew , ey+eh),myRed,2)

        REDtEnd=time.time()             # stop the timer that we started at the top of this for loop
        loopTime = REDtEnd - REDtStart
        REDfps= ((.9 * REDfps) + (.1 * loopTime)) # use a low pass filter to reduce noise in the time calculations
        REDfps= int(REDfps)

#**********************************************************

#**********************************************************
# Now we display the frames per second for the loop that calculates the REDfps for the RED frames around the eye(s)

        cv2.putText(frame,str('{0:0.2f}'.format(REDfps) +' FPS'),REDveryUpperLeft,myFont,fontH1,myRed,fontT1)

#**********************************************************

#**********************************************************
    cv2.imshow('my WEBcam', frame)      # show a frame
    cv2.moveWindow('my WEBcam',0,0)     # move the frame to x,y coordinates on the display

    if cv2.waitKey(1) & 0xff ==ord('q'):# keep repeating until cv2.waitKey() detects the 'q' key 
                                        # being pressed
        break                           # then exit the while True: loop with a break
#**********************************************************

#**********************************************************
# Basic housekeeping chores

cv2.destroyAllWindows()           # Destroy all open windows - Good Housekeeping practice
cam.release()                           # release the camera for other programs to use

os.system('cls')                        # Use the Operating system cls command to clear
                                        	 # the terminal
#**********************************************************
        #  END OF PROGRAM  


