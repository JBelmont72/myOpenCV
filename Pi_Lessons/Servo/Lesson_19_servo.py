'''
What is the period of time for one cycle.  For the servo, the period is 20 ms, or 0.02 seconds
the period of the servo is 20 millseconds  .020 seconds, therefore it goes through 1/0.020  or 50 cycles per second
freq =50
1 % OF THE duty cycle (which is 0.020 seconds) is .00020 seconds  or 0.2 milliseconds that the voltage is high.
10% would be 2 milliseconds high and 12 % would be 2.4 milliseconds out of every  20 milliscond period!


1 % of .020 seconds or 20 milliseonds is .2 milliseconds, 2% is .4 milliseconds which is the low range (about 0 degrees ) for the servo.
12% of 0.20 milliseconds is 2.4 milliseconds  out of a 20 millisecond cycle(at 50 hertz)
around 2% is 0 degrees and around 12-15 % is 180 degrees.
The `pwm.ChangeDutyCycle(pwmPercent)` function in the RPi.GPIO library is used to change the duty cycle of a PWM signal, where `pwmPercent` is a value between 0 and 100 that represents the percentage of time the signal is high during one cycle. This allows you to control the brightness of an LED or the speed of a motor by adjusting how long the signal stays on versus off.
PWM in RPi.GPIO Library
Overview of PWM
Pulse Width Modulation (PWM) is a technique used to control the brightness of LEDs and the speed of motors by varying the duty cycle of a signal. The RPi.GPIO library allows you to generate PWM signals on Raspberry Pi GPIO pins.
Using pwm.ChangeDutyCycle(pwmPercent)
The ChangeDutyCycle method is used to adjust the duty cycle of an existing PWM signal. The duty cycle is the percentage of one cycle in which the signal is high (on) versus low (off).
Parameters
pwmPercent: This is a float value between 0 and 100.
0% means the signal is always off.
100% means the signal is always on.
Values in between adjust the brightness or speed proportionally.
Example Usage
Initialize PWM: First, create a PWM object on a specific GPIO pin with a defined frequency.

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_number, GPIO.OUT)
pwm = GPIO.PWM(pin_number, frequency)
pwm.start(initial_duty_cycle)

 Change Duty Cycle: Use ChangeDutyCycle to modify the brightness or speed.

pwm.ChangeDutyCycle(pwmPercent)
Stop PWM: When done, stop the PWM signal.

pwm.stop()
GPIO.cleanup()
Important Notes
The duty cycle can be changed at any time while the PWM is running.
Ensure that the PWM object is started before calling ChangeDutyCycle.
Always clean up GPIO settings after use to avoid warnings or errors in future runs.
'''
import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
pwmPin=18
GPIO.setup(pwmPin.GPIO.OUT)
pwm=GPIO.PWM(pwmPin,50)
pwm.start()
try:
    while True:
        pwmPercent= float(input('PWM %'))
        pwm.ChangeDutyCycle(pwmPercent)
        sleep(.1) # not needed 
except KeyboardInterrupt:
    GPIO.cleanup()
    print('all cleaned, ready to proceed')
'''   
# step 1: Initialize PWM: First, create a PWM object on a specific GPIO pin with a defined frequency   
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(pin_number, GPIO.OUT)
pwm = GPIO.PWM(pin_number, frequency)
pwm.start(initial_duty_cycle)

# 2.Change Duty Cycle: Use ChangeDutyCycle to modify the brightness or speed
pwm.ChangeDutyCycle(pwmPercent)
# 3. stop the pwm signal
pwm.stop()
GPIO.cleanup()
'''
'''
import RPi.GPIO as gp  
from time import sleep  
gp.setmode(gp.BOARD)  
gp.setup(12,gp.OUT)  
pwm=gp.PWM(12,50)  
pwm.start(0)  
for i in range(0,181):  #essentially goes from 1/18+2 which is ~2.05 to 180/18 =2 which is = 12
   sig=(i/18)+2  
   pwm.ChangeDutyCycle(sig)  
   sleep(0.03)  
for i in range(180,-1,-1):  
   sig=(i/18)+2  
   pwm.ChangeDutyCycle(sig)  
   sleep(0.03)  
pwm.stop()  
gp.cleanup()
'''