import cv2
import mediapipe as mp
import subprocess

# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Tracking Variables
# -----------------------------
prev_face_center = None
prev_hand_center = None

no_face_frames = 0
no_hand_frames = 0

FACE_MISSING_THRESHOLD = 10
HAND_MISSING_THRESHOLD = 10

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb)
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    # ============================================================
    # FACE DETECTION + MOVEMENT
    # ============================================================
    if face_results.detections:
        no_face_frames = 0

        for detection in face_results.detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Face center
            face_center = (x + bw // 2, y + bh // 2)

            # Movement detection
            if prev_face_center is not None:
                dx = abs(face_center[0] - prev_face_center[0])
                dy = abs(face_center[1] - prev_face_center[1])
                if dx + dy > 20:
                    print("Face movement detected")

            prev_face_center = face_center

    else:
        no_face_frames += 1

    # ============================================================
    # HAND DETECTION + MOVEMENT + INDEX-FINGER SWIPE
    # ============================================================
    if hand_results.multi_hand_landmarks:
        no_hand_frames = 0

        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Try to use INDEX FINGER TIP (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            # Check if index finger is visible (inside frame)
            index_visible = 0 <= ix < w and 0 <= iy < h

            if index_visible:
                cx, cy = ix, iy
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)  # yellow dot for index
            else:
                # Fallback to wrist (landmark 0)
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)  # blue dot for wrist

            hand_center = (cx, cy)

            if prev_hand_center is not None:
                dx = hand_center[0] - prev_hand_center[0]
                dy = hand_center[1] - prev_hand_center[1]

                # Basic movement
                if abs(dx) + abs(dy) > 20:
                    print("Hand movement detected")

                # -------------------------
                # SWIPE RIGHT (index or wrist)
                # -------------------------
                if dx > 40 and abs(dy) < 30:
                    print("Swipe Right detected")

                # -------------------------
                # SWIPE LEFT → open different YouTube video
                # -------------------------
                if dx < -40 and abs(dy) < 30:
                    print("Swipe Left detected — opening alternate video")
                    subprocess.Popen([
                        EDGE_PATH,
                        "--inprivate",
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                    ])

            prev_hand_center = hand_center

    else:
        no_hand_frames += 1

    # ============================================================
    # TRIGGER WHEN BOTH FACE + HAND ARE GONE
    # ============================================================
    if no_face_frames > FACE_MISSING_THRESHOLD and no_hand_frames > HAND_MISSING_THRESHOLD:
        print("No face and no hand — opening YouTube in InPrivate mode")

        subprocess.Popen([
            EDGE_PATH,
            "--inprivate",
            "https://www.youtube.com/watch?v=VLP_tnnDGSQ"
        ])
        break

    # ============================================================
    # Display
    # ============================================================
    cv2.imshow("Face + Hand + Gestures", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
