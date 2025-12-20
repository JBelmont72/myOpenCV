'''lesson 16 raspberry pi
/Users/judsonbelmont/Documents/SharedFolders/OpenCV/OpenCV_1/Pi_Lessons/Lesson_16_ADC.py

read values through the ADC0834 analog to digital  chip, and then apply a PWM signal to control the brightness of the LED.
'''
import RPi.GPIO as GPIO
import ADC0834
import time

GPIO.setmode(GPIO.BCM)

LEDPin=23
GPIO.setup(LEDPin,GPIO.OUT)
myPWM=GPIO.PWM(LEDPin,1000)
myPWM.start(0)
ADC0834.setup()
try:
    while True:
        analogVal=ADC0834.getResult(0)
        print('Raw= ',analogVal, 'Vol= ', analogVal/255*5)
        DC=analogVal*100/255
        myPWM.ChangeDutyCycle(DC)
        time.sleep(.2)
        
except KeyboardInterrupt:
    GPIO.cleanup()
    print('GPIO Good to Go')