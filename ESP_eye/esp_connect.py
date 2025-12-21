# import cv2

# cap = cv2.VideoCapture("http://192.168.1.58")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue
#     cv2.imshow("ESP32-S3 Stream", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#http://192.168.1.58/
# File: test_esp_camera.py
# import cv2
# import time
# # my ESP32S3 EYE  camera IP address
# url = "http://192.168.1.58/stream"

# cap = cv2.VideoCapture(url)
# # cap=cv2.VideoCapture(1)
# startTime=time.time()
# fps=20
# new_width=640
# new_height=480
# if not cap.isOpened():
#     print("Failed to connect to camera stream")
#     exit(1)

# print("Connected to camera stream. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     timedelta=time.time() - startTime
#     startTime=time.time()
#     FPS =1/timedelta 
#     fps =.95*fps + 0.05 * FPS
#     # print(frame.shape)
    
#     print(int(fps) )
#     frame=cv2.flip(frame,1)
#     Resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#     print(Resized_frame.shape)
      
#     cv2.putText(Resized_frame,str(int(fps))+' fps',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
#     cv2.imshow("ESP32 Camera", Resized_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#. rame shape  (600, 800, 3) , fps: 17  for svga


import cv2
import time
# my ESP32S3 EYE  camera IP address
url = "http://192.168.1.58/stream"

cap = cv2.VideoCapture(url)
# cap=cv2.VideoCapture(1)
startTime=time.time()
fps=20
new_width=320
new_height=240
# new_width=640
# new_height=480

if not cap.isOpened():
    print("Failed to connect to camera stream")
    exit(1)

print("Connected to camera stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    timedelta=time.time() - startTime
    startTime=time.time()
    FPS =1/timedelta 
    fps =.95*fps + 0.05 * FPS
    # print(frame.shape)
    
    print(int(fps) )
    frame=cv2.flip(frame,1)

    Resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(Resized_frame.shape)
    cv2.putText(Resized_frame,str(int(fps))+' fps',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv2.imshow("ESP32 Camera", Resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()