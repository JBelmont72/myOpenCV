'''


'''
import cv2
import requests
import numpy as np

ESP32_STREAM_URL = "http://192.168.1.58/"   # <-- change to your ESP32 IP

# Haar cascade in the default OpenCV directory
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

def mjpeg_stream(url):
    r = requests.get(url, stream=True)
    bytes_data = b''
    for chunk in r.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield frame

print("Connecting to ESP32-S3 stream...")

for frame in mjpeg_stream(ESP32_STREAM_URL):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=6,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("ESP32-S3 Face Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
