from collections import deque
import numpy as np
import cv2
import mediapipe as mp

# ML ALGO
SEQ_LENGTH = 20
sequence = deque(maxlen=SEQ_LENGTH)

def extract_features(hand_landmarks):
    """Flatten 21 Mediapipe landmarks into a 63D feature vector."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Mediapipe Hands
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # If hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Print all 21 landmark coordinates
                print("\n--- HAND LANDMARKS ---")
                for i, lm in enumerate(hand_landmarks.landmark):
                    print(f"ID {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

                # Extract 63D feature vector
                features = extract_features(hand_landmarks)

                # Add to rolling sequence buffer
                sequence.append(features)

                # Debug overlay
                cv2.putText(frame, f"Seq: {len(sequence)}/{SEQ_LENGTH}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                # When sequence is full, you now have a ready LSTM input
                if len(sequence) == SEQ_LENGTH:
                    seq_array = np.array(sequence)  # shape: (20, 63)
                    print("Sequence ready:", seq_array.shape)

                

                # Example: simple gesture detection
                # Thumb tip = 4, Index tip = 8
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Example gesture: "Pinch"
                if abs(thumb_tip.x - index_tip.x) < 0.03 and abs(thumb_tip.y - index_tip.y) < 0.03:
                    cv2.putText(frame, "PINCH", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Hand Gesture Mediapipe", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
