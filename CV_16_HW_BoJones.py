# ***************************************************
# HOMEWORK ASSIGNMENT
#  1. Get comfortable with training for known and unknown faces.  Use Paul's images
#  2. Create images of YOUR household.  Train for these known faces
#  3. Operate your WEBcam and have the program recognize the household faces LIVE and 
#     have them labeled as they are moving.  Have the label track the moving face. 
#  4. Instrument the code so that it displays the frames/second to see how the system
#     is performing under this computational load.

# ***************************************************
# import the needed libraries for this program


import os                               # import the system os functions for use to clear terminal
import time                             # import the time() library
import cv2                              # import the cv2() library
import face_recognition as FR           # import the face_recognition library and reference
                                        # it as FR.just to make the calling simpler... 
                                        # Less typing is better!
myFont=cv2.FONT_HERSHEY_SIMPLEX         # defines values for myFont

width = width=640                       # for setting width of camera frame
height=360                              # for setting height of camera frame
                              
veryUpperLeft = (6,28)                  # location to place the FPS text

myBlue = (255,0,0)                      # tuple to describe BLUE
myGreen = (0,255,0)                     # tuple to describe GREEN
myRed = (0,0,255)                       # tuple to describe RED

fontH1 = 1                              # variable to set the font height
fontT1 = 1                              # variable to set font thickness 

# ***************************************************

#**********************************************************
# use openCV commands to set up program to display video at desired specification

cam=cv2.VideoCapture(0)
# cam=cv2.VideoCapture(1,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

# ***************************************************

# ***************************************************
# Set up the library tools to load a KNOWN image file to train the program for locating
# particular face(s) - We 'load' the image into this variable using the FR.load_image_file() function.
# This is then encoded using the FR.face_encodings() function.  
# The parameter passed to the FR.face_encodings() function is the variable created to reference 
# the image loaded by the FR.load_image_file() function.  

# In this lesson the first image loaded is # an image of Donald Trump.  
# We name the variable 'meFace' This becomes the face that we will train as 'meFace'
# NOTE the use of the 'r' character preceding the image file path to allow us to 
# use the full path string as a RAW string.

meFace = FR.load_image_file('    .jpg')

# Next we declare a variable to receive the location of the first [0] index of the first face found
# in the 'donFace' image.  In this lesson we call it 'faceLoc'

faceLoc = FR.face_locations(meFace)[0]

# Now we have enough information from the image we can encode the location and other characteristics
# of the face found at 'faceLoc' for use in comparison to other KNOWN face(s)  This is done by calling
# FR.face_encodings()and passing it the 'donFace' parameter along with an index location 
# of the face found at faceLoc -  since we expect it to be the first face found at 'faceLoc' we 
# just pass in the index of the first location which is [0]

meFaceEncode=FR.face_encodings(meFace)[0]

# ***************************************************
# for this encoding example please see the comments for uploading, finding, and encoding 
# the face in the segment above

einsteinFace = FR.load_image_file(r'C:\Users\bobjo\Documents\python\demoImages\known\AlbertEinstein.jpg')
faceLoc = FR.face_locations(einsteinFace)[0]
einsteinFaceEncode=FR.face_encodings(einsteinFace)[0]

# ***************************************************

# ***************************************************
# create an array of the known encodings we have generated for each face we will be trying to recognize
# in some other image

knownEncodings=[meFaceEncode,einsteinFaceEncode] #meFaceEncode

# ***************************************************
# since we will want to label the face(s) found in our 'unknown image' we create an array of string
# labels that will be applied to the found face(s). We make certain to place the label strings in the same order that
# we listed the encodings in the knownEncodings array.  Otherwise they won't be labeled correctly. 
# Don't ask me how I know this...

names=['Me', 'Albert']

# ***************************************************

# ***************************************************

while True:
    fps = 10
    ignore,  unknownFace = cam.read()   # use the FR.load_image_file() function to load an image 
                                        #that we wish to search for faces

    unknownFaceRGB=cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)  # convert cam image image to BGR
                                                                # so it can be used by cv2
    faceLocations=FR.face_locations(unknownFaceRGB)             # search it for faces 
    unknownEncodings=FR.face_encodings(unknownFaceRGB,faceLocations)    # encode any faces found

    name = 'Unknown Person'             # create a string variable called name
                                        # and assign the string 'Unknown Person'
                                        # NOTE - this is the name that will be assigned to any 
                                        # faces that are not known to the program

    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        tStart=time.time()              # start the timer at the start of the loop

        top,right,bottom,left=faceLocation  # assign the locations of the face location found
                                            # and put a rectangle around it
        cv2.rectangle(unknownFace,(left,top - 10 ),(right,bottom + 10),myBlue,3)

        matches=FR.compare_faces(knownEncodings,unknownEncoding)    # call FR.compare_faces()
                                                                    # and compare any faces found
                                                                    # in the knownEncodings array
                                                                    # to the face(s) found in the
                                                                    # unknown Encoding array
                                                                    # and assign that True / False 
                                                                    # to the matches array

# ***************************************************
#
        if True in matches:                                         # parse through the matches array
            matchIndex=matches.index(True)                          # until we find a True condition
            name=names[matchIndex]                                  # assign the name string value found at 
                                                                    # that location to the name variable

# ***************************************************
#       and use the cv2.putText() function to put the name of the face on the bottom of the rectangle

        cv2.putText(unknownFace,name,(left + 40,bottom + 30),myFont,.75,myRed,2)

# ***************************************************

# ***************************************************
#       cv2.moveWindow('My Faces',0,0)              # move the frame to x,y coordinates on the display
        tEnd=time.time()                            # finish the fps timer loop
        loopTime = tEnd - tStart                    # calculate the time the loop took to process
        fps= ((.9 * fps) + (.1 * loopTime))         # generate a regularized fps             
        fps= int(fps)                               # change from a float to an integer 

# ***************************************************
#
# ***************************************************
#       put the frames per second fps value on the screen in the upper left in BLUE

        cv2.putText(unknownFace,str('{0:0.2f}'.format(fps) +' FPS'), veryUpperLeft,myFont,fontH1,myBlue,fontT1)

# ***************************************************

# ***************************************************
#
    cv2.imshow('My Faces',unknownFace)	# display the webcam frame called unknownFace to the screen
                                                		# with its associated framed faces and names

    if cv2.waitKey(1) & 0xff ==ord('q'):        	# check to see if the 'q' key has been pressed
        break                                   		# if so then break the while True: loop

# ***************************************************
#
cam.release()                                   	# release the camera to other applications
cv2.destroyAllWindows()                    # and destroy all windows - GoodHousekeeping practice

os.system('cls')                                   # IMPORTANT NOTE
                                                # Use the Operating system cls command to clear
                                                # the terminal
                                                # DO NOT USE THIS COMMAND IF YOU WANT THE TERMINAL
                                                # TO SAVE ANY DIAGNOSTIC OUTPUT FROM YOUR PROGRAM!
# ***************************************************
#                       END OF PROGRAM  


