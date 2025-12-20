'''
Mixed Matrix Arts has a good tutorial on servos with Raspberry pi. https://www.youtube.com/watch?v=g1eMMIUmh6E

'''
from gpiozero import Servo
from time import sleep

# Define the GPIO pin (using BCM numbering)
# You might need to adjust min_pulse_width and max_pulse_width for your specific servo
servo = Servo(17) 

try:
    while True:
        print("Min position")
        servo.min()
        sleep(1)
        print("Mid position")
        servo.mid()
        sleep(0.5)
        print("Max position")
        servo.max()
        sleep(1)
except KeyboardInterrupt:
    servo.close()


import RPi.GPIO as GPIO
import time
import signal
import atexit

# Ensure GPIO pins are cleaned up on program exit
atexit.register(GPIO.cleanup)

# Set the pin numbering mode to use the physical pin layout (BOARD numbering)
# You can also use GPIO.BCM to refer to the Broadcom pin numbers
GPIO.setmode(GPIO.BOARD) 

# Define the pin connected to the servo signal wire
# Pin 11 is used in BOARD numbering here, which corresponds to BCM GPIO 17
servo_pin = 11

# Set up the pin as an output
GPIO.setup(servo_pin, GPIO.OUT)

# Set up the pin for PWM
# Servos typically require a frequency of 50 Hz
pwm = GPIO.PWM(servo_pin, 50) 

# Start PWM with a duty cycle of 0 (servo is off/idle)
pwm.start(0)
print("Waiting for 2 seconds...")
time.sleep(2)

try:
    # Function to set servo angle
    def set_angle(angle):
        # Calculate the duty cycle for the desired angle
        # Typical values are 2.5 (0 deg) to 12.5 (180 deg)
        # This formula is an approximation: duty_cycle = 2.5 + (angle / 18)
        duty_cycle = 2.5 + (angle / 18)
        GPIO.output(servo_pin, True)
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5) # Give the servo time to move
        GPIO.output(servo_pin, False) # Stop sending signal after move
        pwm.ChangeDutyCycle(0) # Stop the PWM pulse to reduce jitter/heat

    # Sweep the servo from 0 to 180 degrees
    print("Rotating 0 to 180 degrees in steps...")
    for angle in range(0, 181, 10):
        set_angle(angle)
        time.sleep(0.1)

    # Sweep the servo back from 180 to 0 degrees
    print("Rotating back 180 to 0 degrees in steps...")
    for angle in range(180, -1, -10):
        set_angle(angle)
        time.sleep(0.1)

    # Return to the center position (90 degrees)
    print("Returning to center (90 degrees)...")
    set_angle(90)
    time.sleep(1)

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    # Clean up everything
    pwm.stop() # Stop the PWM signal
    GPIO.cleanup() # Reset GPIO pins to their default state
    print("GPIO cleaned up")

