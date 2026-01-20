'''
Docstring for ESP_eye.Servo_control_mp
'''


import cv2
import mediapipe as mp
import threading
import time
cap = cv2.VideoCapture("http://10.0.0.30/stream")

if not cap.isOpened():
    raise RuntimeError("Failed to open stream")
DEAD_ZONE = 20  # pixels
latest_frame = None
lock = threading.Lock()

def capture_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_frame = frame

# START THREAD ONCE
threading.Thread(target=capture_loop, daemon=True).start()

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

last_faces = []
frame_id = 0
startTime=time.time()
FPS=12
while True:
    with lock:
        if latest_frame is None:
            continue
        Frame = latest_frame.copy()
        frame=cv2.flip(Frame,1)

    frame_id += 1
    timeDiff = time.time() -startTime
    fp=1/timeDiff
    fps =int(FPS * 0.90 + 0.1 *fp)
    print(str(fps)+'  fps')
    # Run MediaPipe every other frame
    if frame_id % 2 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb=cv2.flip(Rgb,1)
        results = detector.process(rgb)

        last_faces = []
        if results.detections:
            h, w, _ = frame.shape
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                last_faces.append((
                    int(box.xmin * w),
                    int(box.ymin * h),
                    int(box.width * w),
                    int(box.height * h)
                ))

    for (x, y, bw, bh) in last_faces:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        print(f'x y w h : {x} ,{y}, {x+bw} ,{y+bh}')
        
        face_cx = x + bw // 2
        face_cy = y + bh // 2
        frame_cx = frame.shape[1] // 2
        frame_cy = frame.shape[0] // 2
        print(frame.shape)
        print(f'face_cx: {face_cx} face_cy: {face_cy} frame_cx: {frame_cx} frame_cy: {frame_cy}')
        err_x = face_cx - frame_cx   # pan
        err_y = face_cy - frame_cy   # tilt
        if abs(err_x) < DEAD_ZONE:
            err_x = 0
        if abs(err_y) < DEAD_ZONE:
            err_y = 0
        print(f'Err_x {err_x}   Err_y {err_y}')
        PAN_GAIN  = 0.05
        TILT_GAIN = 0.05

        pan_angle  += int(err_x * PAN_GAIN)
        tilt_angle -= int(err_y * TILT_GAIN)  # inverted 
        sock.sendall(f"P:{pan_angle},T:{tilt_angle}\n".encode())  
    cv2.imshow("ESP32-S3 MediaPipe ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



