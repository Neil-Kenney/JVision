import cv2
import mediapipe as mp
import time
import numpy as np

# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.75
)

# -----------------------------
# Camera
# -----------------------------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Camera did not open. Check permissions and close apps.")

# -----------------------------
# Tracking Variables
# -----------------------------
prev_face_center = None
face_start_time = None
face_visible = False
no_face_frames = 0

# -----------------------------
# Time-in-frame tracking
# -----------------------------
BUFFER_FRAMES = 20    # require N consec frames to confirm presence/absence

present_frames = 0
absent_frames = 0

total_present_seconds = 0.0     # accumulated time present
total_away_seconds = 0.0        # accumulated time away
last_frame_time = time.time()   # for dt accumulation
AWAY_SECONDS_THRESHOLD = 5

# -----------------------------
# Break popup state
# -----------------------------
BREAK_WINDOW = "Take a Break"
BREAK_SECONDS = 20          # show popup once total seconds reach this
break_active = False        # whether popup is currently showing

# break_window = BreakWindow()

# -----------------------------
# Main Loop
# -----------------------------
while True:
    # ret, frame = cap.read()
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if not ret: # error check
    #     break
    ret, frame = cap.read()
    if not ret or frame is None:
        continue  # skip until we get a real frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb)

    h, w, _ = frame.shape

    # ============================================================
    # PRESENCE + TIMER LOGIC
    # ============================================================
    now = time.time()
    dt = now - last_frame_time
    last_frame_time = now

    face_present = bool(face_results.detections)

    if break_active:
        # Create a simple "popup" image
        popup = 255 * (np.ones((220, 520, 3), dtype=np.uint8))

        cv2.putText(popup, "Time for a break!", (35, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(popup, "Step away from the camera.", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(popup, "Press B to dismiss.", (35, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        to_display = popup

        # cv2.imshow(BREAK_WINDOW, popup)
        # if not break_window.active():
        #     break_window.show()
        #     break_window.root.update()

        if face_present:
            if absent_frames >= BUFFER_FRAMES:
                face_present = False
            present_frames += 1
            absent_frames = 0
        else:
            if present_frames >= BUFFER_FRAMES:
                face_present = True
                total_away_seconds = 0.0  # reset away timer if we just 
                                          # transitioned to away
            total_away_seconds += dt
            absent_frames += 1
            present_frames = 0

        if total_away_seconds >= 5: # auto-dismiss break after 5 seconds away
            break_active = False
            # cv2.destroyWindow(BREAK_WINDOW)
            # break_window.close()
            total_present_seconds = 0.0


    else: # break_active = false
        to_display = frame
        if face_present:
            if present_frames >= BUFFER_FRAMES:
                face_present = True
                total_away_seconds = 0.0  # reset away timer if we just
                                          # transitioned to present
            total_present_seconds += dt
            present_frames += 1
            absent_frames = 0
        else:
            if absent_frames >= BUFFER_FRAMES:
                face_present = False
            total_away_seconds += dt
            absent_frames += 1
            present_frames = 0
        
        if total_away_seconds >= 5: # todo: make const AWAY_SECONDS_THRESHOLD
            total_present_seconds = 0.0
    
    
    # ============================================================
    # AUTO-DISMISS BREAK WINDOW IF USER LEAVES
    # ============================================================
    # if break_active and not face_present:
    #     break_active = False
    #     cv2.destroyWindow(BREAK_WINDOW)


    # ============================================================
    # BREAK POPUP TRIGGER (TOTAL TIME)
    # ============================================================
    if total_present_seconds >= BREAK_SECONDS:
        break_active = True
        # cv2.namedWindow(BREAK_WINDOW, cv2.WINDOW_AUTOSIZE)
        # break_window.show()
        # break_window.root.update()

    # ============================================================
    # FACE DETECTION + MOVEMENT
    # ============================================================
    if face_results.detections:

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

            prev_face_center = face_center

    # ============================================================
    # Display
    # ============================================================
    status = "PRESENT" if face_present else "ABSENT"

    cv2.putText(frame, f"Status: {status}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Total: {total_present_seconds:0.1f}s", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # cv2.imshow("Face + Hand + Gestures", frame)
    cv2.imshow("Face + Hand + Gestures", to_display)

    key = cv2.waitKey(1) & 0xFF

    # ESC quits program
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()