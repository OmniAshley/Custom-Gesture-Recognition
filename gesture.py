import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.abspath(""), "gesture_recognizer.task").replace("\\", "/")

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    result_callback=print_result)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Get initial time
start_time = time.time()

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Compute timestamp in ms
        frame_timestamp_ms = int((time.time() - start_time) * 1000)

        # Send image into recognizer
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Show live video
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
