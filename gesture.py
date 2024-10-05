# importing required libraries
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to get distance between two points
def get_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Function to control volume based on hand position and movement
def control_volume(hand_landmarks, previous_vertical_position, sensitivity, scaling_factor):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = get_distance(thumb_tip, index_tip)
    volume = int(np.interp(distance, [0.1, 0.25], [0, 100]))  # Adjusted distance range

    # Detect hand movement direction
    current_vertical_position = thumb_tip.y
    if previous_vertical_position is not None:
        delta_y = current_vertical_position - previous_vertical_position
        if abs(delta_y) > sensitivity:  # Threshold for hand movement (adjustable sensitivity)
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # Corrected attribute
            volume_object = cast(interface, POINTER(IAudioEndpointVolume))
            
            if delta_y < 0:
                # Move hand up to increase volume
                current_volume = volume_object.GetMasterVolumeLevelScalar()
                new_volume = min(1.0, current_volume + scaling_factor * (volume / 100.0))
                volume_object.SetMasterVolumeLevelScalar(new_volume, None)
            else:
                # Move hand down to decrease volume
                current_volume = volume_object.GetMasterVolumeLevelScalar()
                new_volume = max(0.0, current_volume - scaling_factor * (volume / 100.0))
                volume_object.SetMasterVolumeLevelScalar(new_volume, None)
    return current_vertical_position

# Main function
def main():
    prev_y = None  # Previous vertical position of the hand
    sensitivity = 0.02  # Adjust this value for sensitivity
    scaling_factor = 0.2  # Adjust this value for fine-tuning volume change rate
    try:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255))
                        )  # Use landmark_drawing_spec to draw landmarks in white color
                        prev_y = control_volume(hand_landmarks, prev_y, sensitivity, scaling_factor)
                cv2.imshow('Hand Gesture Volume Control', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
    except ImportError as e:
        print(f"Error: {e}. Please make sure to install the required libraries (cv2, mediapipe, numpy, pycaw).")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
