'''
Docstring for Pi_Lessons.Dec2025
'''


import sys
import time

import cv2
import numpy as np
import mediapipe as mp

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# my ESP32S3 EYE  camera IP address
url = "http://192.168.1.58/stream"

cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

prev_time = time.time()
frame_count = 0

print("Starting camera loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    frame_count += 1
    if frame_count % 30 == 0:
        now = time.time()
        fps = 30 / (now - prev_time)
        prev_time = now
        print(f"FPS: {fps:.1f}")

    cv2.imshow("MediaPipe Sanity Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
