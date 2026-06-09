import cv2
import mediapipe as mp
import time
import os
from pathlib import Path
import pyautogui

# ----------------------------------------------------------------------
# 1. Constants and globals
# ----------------------------------------------------------------------
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False

COOLDOWN = 1.0
last_action_time = 0
latest_result = None
last_palm_pos = None

# ----------------------------------------------------------------------
# 2. MediaPipe imports and configuration
# ----------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ----------------------------------------------------------------------
# 3. Hand bone connections (for drawing the skeleton)
# ----------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ----------------------------------------------------------------------
# 4. Callback — stores latest result from MediaPipe
# ----------------------------------------------------------------------
def on_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# ----------------------------------------------------------------------
# 5. Draw hand skeleton on frame
# ----------------------------------------------------------------------
def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 0), 2)

    for i, pt in enumerate(points):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(frame, pt, 5, color, -1)

# ----------------------------------------------------------------------
# 6. Move mouse relative to previous palm position
# ----------------------------------------------------------------------
def move_mouse(hand_landmarks):
    global last_palm_pos

    ref = hand_landmarks[5]

    if last_palm_pos is None:
        last_palm_pos = (ref.x, ref.y)
        return

    dx = ref.x - last_palm_pos[0]
    dy = ref.y - last_palm_pos[1]

    sensitivity = 3
    pyautogui.moveRel(
        int(dx * SCREEN_W * sensitivity),
        int(dy * SCREEN_H * sensitivity),
        duration=0
    )

    last_palm_pos = (ref.x, ref.y)

# ----------------------------------------------------------------------
# 7. Pause video — clicks at fixed screen position
# ----------------------------------------------------------------------
def pause_vid():
    pyautogui.moveTo(1281, 586)
    pyautogui.click()

# ----------------------------------------------------------------------
# 8. Handle gestures
# ----------------------------------------------------------------------
def handle_gestures(frame, cap):
    global last_action_time, last_palm_pos

    if not latest_result or not latest_result.gestures:
        last_palm_pos = None
        return

    should_exit = False

    for i, gesture_list in enumerate(latest_result.gestures):
        gesture_name = gesture_list[0].category_name
        score = gesture_list[0].score

        # Display gesture label on frame
        cv2.putText(frame, f"{gesture_name} ({score:.0%})",
                    (10, 60 + i * 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Victory → exit program
        if gesture_name == "Victory" and score > 0.5:
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Pointing up → move mouse
        elif gesture_name == "Pointing_Up" and score > 0.5:
            if i < len(latest_result.hand_landmarks):
                move_mouse(latest_result.hand_landmarks[i])

        # Open palm → pause video (with cooldown)
        elif gesture_name == "Open_Palm" and score > 0.5:
            current_time = time.time()
            print(f"Current time!!!!!!!!!!:{current_time}")
            if current_time - last_action_time > COOLDOWN:
                print(F"LAST ACTION TIME!!!!!!!!!!!!!!!: {last_action_time}")
                pause_vid()
                last_action_time = current_time

        # Any other gesture → reset mouse tracking
        else:
            last_palm_pos = None

# ----------------------------------------------------------------------
# 9. Model path
# ----------------------------------------------------------------------
script_dir = Path(__file__).parent.parent
MODEL_PATH = os.getenv('GESTURE_RECOGNIZER_MODEL', str(script_dir / 'gesture_recognizer.task'))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result
)

# ----------------------------------------------------------------------
# 10. Main loop
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            recognizer.recognize_async(mp_image, int(time.time() * 1000))

            # Draw landmarks
            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            # Handle gestures
            handle_gestures(frame, cap)

            cv2.imshow('Hand Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

main()