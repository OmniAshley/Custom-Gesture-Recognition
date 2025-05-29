#----------------------------------------------------------#
# Name: Custom Gesture Recognition Algorithm
# Purpose: To create a custom gesture recognition algorithm
#          to control a PC without mouse & keyboard.
# Author: Ashley Beebakee (https://github.com/OmniAshley)
# Date Created: 17/05/2025
# Last Updated: 21/05/2025
# Python Version: 3.10.6
#----------------------------------------------------------#

# Required libraries for gesture_recognizer.task
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Additional libraries for os, webcam and time components
import os
import cv2
import time

"""
N.B. The model was trained on approx. 30K real-world images
It has 21 landmarks as follows:
0. WRIST
1. THUMB_CMC  5. INDEX_FINGER_MCP   9. MIDDLE_FINGER_MCP  13.RING_FINGER_MCP  17. PINKY_MCP
2. THUMB_MCP  6. INDEX_FINGER_PIP  10. MIDDLE_FINGER_PIP  14.RING_FINGER_PIP  18. PINKY_PIP
3. THUMB_IP   7. INDEX_FINGER_DIP  11. MIDDLE_FINGER_DIP  15.RING_FINGER_DIP  19. PINKY_DIP
4. THUMB_TIP  8. INDEX_FINGER_TIP  12. MIDDLE_FINGER_TIP  16.RING_FINGER_TIP  20. PINKY_TIP
"""
# Define global variables
hand_landmarks_list = []

# Define hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Wrist
]

# Path to the model defined in the correct format
model_path = os.path.join(os.path.abspath(""), "gesture_recognizer.task").replace("\\", "/")

# Build and task configuration for live stream
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global hand_landmarks_list
    hand_landmarks_list = [] # Resets each frame (detects changes)

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            hand_landmarks_list.append(landmarks)  
    #print('Gesture recognition result: {}'.format(result))

# Configuration for GestureRecognizer task
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
    result_callback=print_result)

# Start capturing video from the webcam with custom resolution
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warm-up frames to stabilise webcam
for _ in range(5):
    cap.read()
    time.sleep(0.1)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Get initial time
start_time = time.time()

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Compute timestamp in ms
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Send image into recognizer
        recognizer.recognize_async(mp_image, timestamp_ms)

        # Draw hand landmarks
        image_height, image_width, _ = frame.shape
        for landmarks in hand_landmarks_list:
            for landmark in landmarks:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw bones by connecting landmarks
            for start_index, end_index in HAND_CONNECTIONS:
                x0 = int(landmarks[start_index].x * image_width)
                y0 = int(landmarks[start_index].y * image_height)
                x1 = int(landmarks[end_index].x * image_width)
                y1 = int(landmarks[end_index].y * image_height)
                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # Show live video
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
