'''Very Good  has two .py files and at the end is a .json file
https://www.youtube.com/watch?v=m4pfG47i74E
I have a copy of all programs in the openCV_1 supplemental folder

There was no homework for lesson 27 so I am showing my version of save/load that Paul will address in lesson 28 according to his video description. I am using a feature of PySimpleGUI to save user settings so it will be different than what Paul will show.
Link to Paul McWhorter's lesson 27   

 • AI for Everyone LESSON 27: Improved G...  
Link to PySimpleGUI webpage https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFh1b0tfcDJ1RmZYNDdwSmt1SE93eFBKQ2lwd3xBQ3Jtc0ttLWhCZVl4YThQMEcwT3NsTE9QMlFaSVVVYmNsZVEwYVdaTEdUblAzN2M1Y3Z1czRRUUotbzRULWI0akY2aFVEWjRwVWpHeWhIeXZlb295NC1uaC10dG54NC1sbmpVMkM3eTV0UUNCNWlxTXd5RzhlQQ&q=https%3A%2F%2Fpysimplegui.readthedocs.io%2Fen%2Flatest%2F&v=m4pfG47i74E
Code on GitHub https://github.com/kcl1s/Hand-Gestures
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbnlCZGkwbmN0X2p4TTl0aDNyZTd1aVphbFpLZ3xBQ3Jtc0traHRUNDhza3FFTGF4U1RHY1pnNjZ3T29admNLUUpncllTalRxOHlUNk9OY2YxVDUySHBYSWp3OTd4ZzlkNWh5SXpqY2p6aUhva0Y5OXpYTWJlMTNDbWU4R3UtZTVvSmxZaC1KQ2ZqLWFkTFRwTmJVTQ&q=https%3A%2F%2Fgithub.com%2Fkcl1s%2FHand-Gestures&v=m4pfG47i74E

https://github.com/kcl1s/Hand-Gestures
discusses scalling factors:
https://www.youtube.com/watch?v=Z-uRjDnW0PU


'''
import cv2
print(cv2.__version__)
import numpy as np
import gesturehelp as cvh
import PySimpleGUI as sg
keyDist=((0,4),(0,8),(0,12),(0,16),(0,20),(4,8),(4,12),(4,16),(4,20),(8,12),
            (8,16),(8,20),(12,16),(12,20),(16,20),(0,17))
curE=[10000]*10
tolThresh=400
knownG= False
width=640
height=480
cvh.TrackFPS.start(.05)
cvh.mpHand.start()
sg.user_settings_filename(path='.')     # The settings file will be in the program folder programName.json
DMlist=sg.user_settings_get_entry('-DMs-') #If setting json file available import else use blank
if DMlist != None:
    DMs=np.array(DMlist)
else:
    DMs=np.zeros([10,16],dtype='int')
gNames=sg.user_settings_get_entry('-gNames-',[])
if gNames == []:
    gNames=['']*10

def jointDistance(hand):
    global keyDist
    global DMs
    for x in range(len(keyDist)):
        DMs[0][x]=int(np.sqrt((hand[keyDist[x][0]][0]-hand[keyDist[x][1]][0])**2 + 
                              (hand[keyDist[x][0]][1]-hand[keyDist[x][1]][1])**2))

def findError(knownGesture):
    global DMs
    error=0
    for i in range(15):
        error+= int(abs(DMs[0][i]*100/DMs[0][15] - DMs[knownGesture][i]*100/DMs[knownGesture][15]))
    return error

sg.theme('DarkGreen5')
Icam= sg.Image(filename='',size=(width,height),pad=0,enable_events= True, k='Icam')
Dlayout= [[sg.T('Unknown',font='Times 32',k='Tcur')],[sg.T('Gesture Names     (Check to Train)   ', font=16)]]
Dlayout+= [[sg.In(default_text =gNames[i],s=20,font=16,k='gesture'+str(i)),
            sg.CB('     ',font=16,enable_events=True,k='train'+str(i))] for i in range(1,10)]
layout= [[Icam,sg.Column(Dlayout)],
        [sg.Quit(font=16,size=6)]]
window=sg.Window('cv Gestures', layout,grab_anywhere_using_control = True,finalize=True)
for i in range(1,10): 
    if gNames[i] != '':
        window['train'+str(i)].update(value=True)
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)

while True:
    event, values = window.read(10)
    ignore,  frame = cam.read()
    hand,handLR=cvh.mpHand.getLM(frame,True)
    if hand:
        jointDistance(hand[0])
        if event.startswith('train') and values[event]:
            DMs[int(event[-1:])]=DMs[0]
            print (DMs[int(event[-1:])])
        else:
            for i in range(1,10):
                if values['train'+str(i)]:
                    curE[i]=findError(i)
                    window['train'+str(i)].update(text=curE[i])
                else:
                    window['train'+str(i)].update(text='')
            if min(curE)<tolThresh:
                inNum=curE.index(min(curE))
                inText=values['gesture'+str(inNum)]
            else:
                inText='Unknown' 
            window['Tcur'].update(value=inText)
            
    cv2.putText(frame,str(int(cvh.TrackFPS.getFPS())).rjust(3)+str(' FPS'),(0,50),
                cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)    
    if event in (sg.WIN_CLOSED, 'Quit'):
        for i in range(1,10):
            gNames[i]=window['gesture'+str(i)].get()
        sg.user_settings_set_entry('-gNames-',gNames)
        sg.user_settings_set_entry('-DMs-',DMs.tolist())
        break
    window['Icam'].update(data=cv2.imencode('.ppm', frame)[1].tobytes())
    
cam.release()
window.close()

'''
import mediapipe as mp
import cv2
import time

class mpHand:
    def start():
        mpHand.Hands=mp.solutions.hands.Hands(False,1,.5,.5)
        mpHand.Draw=mp.solutions.drawing_utils

    def getLM(img,doDraw):
        frame=img
        h,w,c=frame.shape
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=mpHand.Hands.process(frameRGB)
        myHands=[]
        handsType=[]
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handsType.append(hand.classification[0].label)
            for HLMs in results.multi_hand_landmarks:
                myHand=[]
                if doDraw==True:
                    mpHand.Draw.draw_landmarks(frame,HLMs,mp.solutions.hands.HAND_CONNECTIONS)
                for LM in HLMs.landmark:
                    myHand.append((int(LM.x*w),int(LM.y*h)))
                myHands.append(myHand)              
        return myHands,handsType    #returns 21 x,y tuples and Left,Right for each hand
                                    #https://google.github.io/mediapipe/solutions/hands.html

class TrackFPS:           #Weighted average (low pass) filter for frames per second
    def start(dataWeight):
        TrackFPS.dw=dataWeight
        TrackFPS.state=0
    
    def getFPS():
        if TrackFPS.state==0:
            TrackFPS.average=0
            TrackFPS.tlast=time.time()
            TrackFPS.state = 1
        elif TrackFPS.state==1:
            TrackFPS.tDelta=time.time()-TrackFPS.tlast
            TrackFPS.average=1/TrackFPS.tDelta
            TrackFPS.tlast=time.time()
            TrackFPS.state = 2            
        else:
            TrackFPS.tDelta=time.time()-TrackFPS.tlast
            TrackFPS.fps=1/TrackFPS.tDelta
            TrackFPS.average=(TrackFPS.dw * TrackFPS.fps)+((1 - TrackFPS.dw) * TrackFPS.average)
            TrackFPS.tlast=time.time()
        return TrackFPS.average



'''

'''
thiis is a .json file
{"-gNames-": ["", "fist", "peace", "3 fingers", "guns up", "1 finger", "5 fingers", "", "", ""], "-DMs-": [[37, 50, 46, 37, 29, 18, 18, 15, 18, 6, 15, 23, 10, 18, 8, 21], [158, 96, 81, 69, 70, 61, 82, 107, 124, 25, 51, 70, 26, 45, 19, 94], [136, 243, 244, 72, 55, 108, 116, 66, 90, 79, 174, 198, 172, 190, 25, 93], [97, 232, 245, 205, 58, 139, 148, 115, 40, 65, 115, 178, 68, 187, 147, 97], [163, 241, 51, 41, 45, 217, 131, 152, 173, 192, 199, 205, 21, 41, 20, 100], [114, 247, 85, 55, 51, 135, 29, 62, 80, 162, 192, 204, 34, 53, 22, 95], [161, 242, 256, 243, 209, 185, 233, 251, 265, 61, 101, 160, 46, 117, 73, 115], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}

'''