



import cv2
import numpy as np
import requests
import time
# Replace with your actual video stream URL
url = 'http://192.168.1.58/stream'

# Create a video stream object
cap = cv2.VideoCapture(url)
fps=15
startTime=time.time()
while True:
    # Read frame by frame from the video stream
    ret, frame = cap.read()
    timeDiff=time.time() - startTime
    startTime=time.time()
    FPS =1/timeDiff 
    FPS =.95*fps + 0.05 * FPS
    print(int(FPS))
    if not ret:
        break  # Break the loop if there are no frames

    # Encode the frame to JPEG format to get the jpeg_bytes
    success, jpeg_bytes = cv2.imencode('.jpg', frame)
    if not success:
        break
    
    # Decode jpeg_bytes back to a frame for processing
    frame_decoded = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    # Downscale for tracking
    frame_small = cv2.resize(frame_decoded, (320, 240))

    # Convert to HSV for color tracking
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

    # (Optional) Display the frame
    cv2.imshow('Frame', hsv)  # Or any other way you want to display it
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
To access the `jpeg_bytes` for your OpenCV program when receiving a video stream via a URL, you'll need to ensure you are correctly reading the bytes from the stream and decoding them into images that OpenCV can use. Here’s how you can accomplish this step-by-step:

## Accessing `jpeg_bytes` from a Video Stream

### Step 1: Read the Video Stream

You can use libraries like `requests` to fetch the stream or directly use OpenCV for reading from the URL. Assuming you're using an appropriate method to fetch the JPEG data, here’s one standard way to do it.

### Step 2: Convert the Stream to `jpeg_bytes`

Here's a code snippet to illustrate how you might fetch the image bytes from a stream:

```python
import cv2
import numpy as np
import requests

# Replace with your actual video stream URL
url = 'http://example.com/path/to/video/stream'

# Create a video stream object
cap = cv2.VideoCapture(url)

while True:
    # Read frame by frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no frames

    # Encode the frame to JPEG format to get the jpeg_bytes
    success, jpeg_bytes = cv2.imencode('.jpg', frame)
    if not success:
        break
    
    # Decode jpeg_bytes back to a frame for processing
    frame_decoded = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)

    # Downscale for tracking
    frame_small = cv2.resize(frame_decoded, (320, 240))

    # Convert to HSV for color tracking
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

    # (Optional) Display the frame
    cv2.imshow('Frame', hsv)  # Or any other way you want to display it
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Explanation

- **`cap.read()`:** Reads each frame from the video stream. This returns a boolean `ret` that indicates if the frame was successfully read, and `frame` which contains the actual image data.
- **`cv2.imencode('.jpg', frame)`:** This function encodes the image as JPEG format, giving you the `jpeg_bytes`.
- **`cv2.imdecode()`:** Converts `jpeg_bytes` back into a format that OpenCV can work with (BGR format).
- **Downscaling and Color Conversion:** This occurs as you described in your code.

### Step 3: Handling Errors

Make sure to manage exceptions such as network issues or invalid URLs by adding error handling in your code to ensure robustness.

This setup allows you to continuously read your video stream, process each frame, and access the `jpeg_bytes` as required. Adjust the URL in the example to your actual video stream URL.


'''