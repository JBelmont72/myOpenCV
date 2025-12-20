'''
use lesson 15 with adc8304

linear equation   the adc reads from 0 to 255 and we want the Percent duty cycle to be 2% for 0 degrees and 255 we want 180 degrees which will be 100%
when all way to right  we put in adc reading of 0 and the servo is a 2% .
when the servo is allthe way to left, the adc reading is 255 and the  output to the servo  is 12%

(0,2)       (255 adc reading , 12 % servo persent duty cycle)m=(12-2)/(255-0) m=10/255

y-2 =(10/255)(x-0)
y =10/255 time xVal +2
y is the pwm val and the x  is the analog val from the potentiometer 
pwmVal=(10/255)*(analogVal)+2
pin 3 is channel zero
'''

import RPi.GPIO as GPIO
import ADC0834
from time import sleep
GPIO.setmode(GPIO.BCM)
pwmPin=18. # 4 a good choice also
GPIO.setup(pwmPin.GPIO.OUT)
pwm=GPIO.PWM(pwmPin,50)
pwm.start()
ADC0834.setup()
try:
    while True:
        analogVal = ADC0834.getResult(0)    # 0 is the channel
        pwmPercent = (10/255) * analogVal +2
        print(pwmPercent)
        pwm.ChangeDutyCycle(pwmPercent)
        sleep(.1) 
except KeyboardInterrupt:
    GPIO.cleanup()
    print('all cleaned, ready to proceed')