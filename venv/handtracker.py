import cv2
import mediapipe as mp
import time
import os
from pathlib import Path

# ----------------------------------------------------------------------
# 1. MediaPipe imports and configuration
# ----------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ----------------------------------------------------------------------
# 2. Hand bone connections (for drawing the skeleton)
# ----------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17)                # palm
]

# Global variable to store the latest detection result
latest_result = None

# ----------------------------------------------------------------------
# 3. Callback function (runs automatically when new results arrive)
# ----------------------------------------------------------------------
def update_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    # Optional: uncomment to see timestamps in the console
    # print(f"Result received at {timestamp_ms} ms")

# ----------------------------------------------------------------------
# 4. Helper function: draw landmarks and connections
# ----------------------------------------------------------------------
def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    # Convert normalized coordinates (0..1) to pixel positions
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Draw bones (lines)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 0), 2)

    # Draw landmark points
    for i, pt in enumerate(points):
        # Wrist (index 0) = red, others = blue
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(frame, pt, 5, color, -1)

# ----------------------------------------------------------------------
# 5. Set up the HandLandmarker in LIVE_STREAM mode
# ----------------------------------------------------------------------
# Load model path from environment variable or use relative path
script_dir = Path(__file__).parent.parent  # Go up from venv to HandTracker directory
MODEL_PATH = os.getenv('HAND_LANDMARKER_MODEL', str(script_dir / 'hand_landmarker.task'))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,   # asynchronous mode
    num_hands=2,
    result_callback=update_result                 # our callback function
)

# ----------------------------------------------------------------------
# 6. Start webcam and run hand tracking (LIVE_STREAM)
# ----------------------------------------------------------------------
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Mirror the image (natural mirror view)
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send frame asynchronously
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # If we have received at least one detection result, draw it
        if latest_result is not None:
            # Draw all detected hands
            if latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

                # Correct handedness labels because the frame is flipped
                for i, handedness_list in enumerate(latest_result.handedness):
                    original_label = handedness_list[0].display_name  # "Left" or "Right"
                    # Swap due to horizontal flip
                    corrected_label = "Left" if original_label == "Right" else "Right"
                    cv2.putText(frame, corrected_label, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show the resulting frame
        cv2.imshow('Hand Tracker (LIVE_STREAM mode)', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()