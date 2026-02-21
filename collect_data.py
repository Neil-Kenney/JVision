import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import time

# Config
SEQ_LENGTH = 20
DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Key bindings for labeling
GESTURE_LABELS = {
    ord('l'): "swipe_left",
    ord('r'): "swipe_right",
    ord('u'): "swipe_up",
    ord('d'): "swipe_down",
    ord('n'): "no_gesture"
}

# Feature Extraction
def extract_features(hand_landmarks):
    """Flatten 21 Mediapipe landmarks into a 63D feature vector."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


# MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQ_LENGTH)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    print("=== DATA COLLECTION MODE ===")
    print("Press:")
    print("  L = swipe_left")
    print("  R = swipe_right")
    print("  U = swipe_up")
    print("  D = swipe_down")
    print("  N = no_gesture")
    print("ESC to exit")
    print("=============================")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.resize(frame, (1280, 720))


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features
            features = extract_features(hand_landmarks)
            sequence.append(features)

            # Show sequence progress
            cv2.putText(frame, f"Seq: {len(sequence)}/{SEQ_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # If sequence full → ready to save
            if len(sequence) == SEQ_LENGTH:
                seq_array = np.array(sequence)

                key = cv2.waitKey(1)
                if key in GESTURE_LABELS:
                    label = GESTURE_LABELS[key]
                    filename = f"{label}_{int(time.time()*1000)}.npy"
                    save_path = os.path.join(DATA_DIR, filename)
                    np.save(save_path, seq_array)
                    print(f"[SAVED] {save_path}")

        cv2.imshow("Collecting Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
