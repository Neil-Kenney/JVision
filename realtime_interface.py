import cv2
import mediapipe as mp
import numpy as np
from collections import deque, defaultdict
import time
import threading
import http.server
import pyautogui
import tkinter as tk
import math
from tensorflow.keras.models import load_model


# ============================================================
# CONFIG
# ============================================================

# Pinch toggle
PINCH_INDEX_MAX = 0.050
PINCH_OTHERS_MIN = 0.080
PINCH_DEBOUNCE = 0.55
PINCH_HOLD_FRAMES = 4
WRIST_STATIONARY_MAX = 0.015

# LSTM model config (sequence of 20 frames × 63 features)
SEQ_LENGTH = 20
FEATURE_SIZE = 63
LR_CONFIDENCE_THRESHOLD = 0.85
MIN_LR_DISPLACEMENT = 0.02  # minimal wrist displacement before we even consider LR model

# Velocity-based swipe detection
# PRESENTATION MODE (used mainly for vertical)
VEL_WINDOW = 9
VEL_THRESHOLD_HORIZONTAL = 0.028  # still used for gating LR model via motion
# ACTION MODE (armed) – vertical only
VEL_THRESHOLD_VERTICAL_UP = 0.035
VEL_THRESHOLD_VERTICAL_DOWN = 0.065
MIN_VERTICAL_DISPLACEMENT = 0.06

# FILE MODE (B leaning toward C) – vertical only
FILEMODE_VEL_THRESHOLD = 0.05
FILEMODE_MIN_DISPLACEMENT = 0.025
FILEMODE_FREEZE = 0.35
filemode_freeze_until = 0

# Cooldowns
SWIPE_COOLDOWN = 1
FREEZE_TIME = 0.4

# Hand entry ignore
HAND_ENTRY_IGNORE_FRAMES = 1
hand_entry_counter = 0

# Two-hand lockout
TWO_HAND_THRESHOLD = 5
ONE_HAND_THRESHOLD = 5
twohand_exit_grace = 0
TWOHAND_EXIT_GRACE_FRAMES = 6   # tune as needed


# State
last_pinch_toggle = 0
pinch_hold_counter = 0
wrist_history = deque(maxlen=PINCH_HOLD_FRAMES)
wrist_buffer = deque(maxlen=20)
landmark_seq = deque(maxlen=SEQ_LENGTH)  # for LSTM model input
hand_present_frames = 0
hand_absent_frames = 0
two_hand_frames = 0
one_hand_frames = 0
two_hands_active = False
last_swipe_time = 0
freeze_until = 0.0

DEBUG_GESTURES = False


current_index = 0
file_mode = False
armed = False

# ============================================================
# NETWORK FILE SEND
# ============================================================

HOST_IP = "192.168.137.1"
HOST_PORT = 8765
SERVER_TIMEOUT = 20

active_server = None
bt_status_message = ""
bt_status_time = 0

def set_bt_status(msg):
    global bt_status_message, bt_status_time
    bt_status_message = msg
    bt_status_time = time.time()
    print("[BT]", msg)

class SingleFileHandler(http.server.BaseHTTPRequestHandler):
    filepath = None
    filename = None

    def do_GET(self):
        if self.path == "/file":
            try:
                with open(self.filepath, "rb") as f:
                    data = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{self.filename}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                set_bt_status("File sent successfully!")

                def _shutdown():
                    global active_server
                    try:
                        self.server.shutdown()
                    except:
                        pass
                    active_server = None

                threading.Thread(target=_shutdown, daemon=True).start()

            except Exception as e:
                print("[NET] send error:", e)
                self.send_response(500)
                self.end_headers()

        elif self.path == "/":
            html = f"""
            <html><body style="font-family:sans-serif;text-align:center;padding:50px">
            <h2>Incoming file: {self.filename}</h2>
            <p>Someone wants to send you a file.</p>
            <a href="/file" style="background:#4CAF50;color:white;padding:15px 30px;
            text-decoration:none;border-radius:8px;font-size:18px">Accept & Download</a>
            </body></html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass

def send_document(doc):
    def _send():
        global active_server

        filepath = doc["path"]
        filename = filepath.replace("\\", "/").split("/")[-1]

        SingleFileHandler.filepath = filepath
        SingleFileHandler.filename = filename

        try:
            http.server.HTTPServer.allow_reuse_address = True
            server = http.server.HTTPServer((HOST_IP, HOST_PORT), SingleFileHandler)
            active_server = server
            url = f"http://{HOST_IP}:{HOST_PORT}/"
            set_bt_status(f"Ready — tell receiver to open: {url}")
            print("[NET] Serving at", url)

            server.timeout = SERVER_TIMEOUT
            deadline = time.time() + SERVER_TIMEOUT

            while time.time() < deadline:
                server.handle_request()

            active_server = None

        except Exception as e:
            active_server = None
            set_bt_status(f"Server error: {e}")

    threading.Thread(target=_send, daemon=True).start()
    set_bt_status(f"Starting file server for {doc['name']}...")

# ============================================================
# UI
# ============================================================

def draw_doc_ui(frame):
    x, y = 20, 50
    h = 30

    for i, doc in enumerate(documents):
        color = (255, 255, 255)
        if i == current_index:
            color = (0, 255, 255)
        if i == current_index and armed:
            color = (0, 255, 0)

        cv2.putText(frame, doc["name"], (x, y + i*h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    mode_text = "FILE MODE" if file_mode else "PRESENTATION MODE"
    cv2.putText(frame, mode_text, (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    if armed and not file_mode:
        cv2.putText(frame, "ARMED (up=send, down=cancel)",
                    (x, y + (len(documents)+2)*h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if bt_status_message and (time.time() - bt_status_time) < 5:
        cv2.putText(frame, bt_status_message,
                    (x, y + (len(documents)+4)*h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

# ============================================================
# OVERLAY WINDOW
# ============================================================

_overlay_root = None
_overlay_label = None

def _overlay_thread():
    global _overlay_root, _overlay_label
    _overlay_root = tk.Tk()
    _overlay_root.title("Mode")
    _overlay_root.attributes("-topmost", True)
    _overlay_root.resizable(False, False)
    _overlay_root.overrideredirect(True)

    _overlay_label = tk.Label(_overlay_root, text="", font=("Segoe UI", 14, "bold"),
                              bg="#222222", fg="#00FFAA", padx=12, pady=6)
    _overlay_label.pack()

    screen_w = _overlay_root.winfo_screenwidth()
    win_w = 220
    win_h = 50
    x = screen_w - win_w - 20
    y = 40
    _overlay_root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    _overlay_root.mainloop()

def start_overlay():
    threading.Thread(target=_overlay_thread, daemon=True).start()

def update_overlay(text, color="#00AAFF"):
    if _overlay_label:
        try:
            _overlay_label.config(text=text, fg=color)
        except:
            pass

# ============================================================
# PINCH METRICS
# ============================================================

def pinch_metrics(hand):
    lm = hand.landmark
    def d(a, b):
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
    return d(lm[4], lm[8]), d(lm[4], lm[12]), d(lm[4], lm[16])

# ============================================================
# FEATURE EXTRACTION FOR LSTM
# ============================================================

def extract_features_from_hand(hand):
    # 21 landmarks × (x, y, z) → 63 features
    feats = []
    for lm in hand.landmark:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=np.float32)

def predict_lr_gesture(model, labels, seq):
    # seq: deque/list of length SEQ_LENGTH, each 63-dim
    arr = np.array(seq, dtype=np.float32).reshape(1, SEQ_LENGTH, FEATURE_SIZE)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argmax(probs)
    label = labels[idx]
    conf = probs[idx]
    return label, conf

# ============================================================
# MAIN
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load trained gesture model + labels
gesture_model = load_model("gesture_model.h5")
gesture_labels = np.load("gesture_labels.npy")

# ============================================================
# FILE DOCKER INTEGRATION
# ============================================================
from file_docker import choose_files_popup

documents = choose_files_popup()
current_index = 0


cap = cv2.VideoCapture(0)

# ============================================================
# FIX: WAIT FOR CAMERA BEFORE STARTING OVERLAY
# ============================================================

ok, frame = cap.read()
while not ok or frame is None or frame.size == 0:
    ok, frame = cap.read()

start_overlay()
update_overlay("PRESENTATION", "#00AAFF")

# ============================================================
# MEDIAPIPE HANDS LOOP
# ============================================================

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.time()

        # -----------------------------
        # HAND PRESENCE + ENTRY IGNORE
        # -----------------------------
        if results.multi_hand_landmarks:
            if hand_present_frames == 0:
                hand_entry_counter = HAND_ENTRY_IGNORE_FRAMES
                wrist_buffer.clear()
                landmark_seq.clear()

            hand_present_frames += 1
            hand_absent_frames = 0

            if hand_entry_counter > 0:
                hand_entry_counter -= 1

        else:
            hand_absent_frames += 1
            if hand_absent_frames > 3:
                hand_present_frames = 0
                wrist_buffer.clear()
                landmark_seq.clear()
                hand_entry_counter = 0

            draw_doc_ui(frame)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # -----------------------------
        # TWO-HAND LOCKOUT
        # -----------------------------
        count = len(results.multi_hand_landmarks)

        if count > 1:
            two_hand_frames += 1
            one_hand_frames = 0
        else:
            one_hand_frames += 1
            two_hand_frames = 0

        if two_hand_frames >= TWO_HAND_THRESHOLD:
            two_hands_active = True

        if one_hand_frames >= ONE_HAND_THRESHOLD and two_hands_active:
            two_hands_active = False
            twohand_exit_grace = TWOHAND_EXIT_GRACE_FRAMES  # <<< NEW

        # If currently in two-hand lockout
        if two_hands_active:
            draw_doc_ui(frame)
            cv2.putText(frame, "Two hands detected - paused", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # Grace period AFTER leaving two-hand mode
        if twohand_exit_grace > 0:
            twohand_exit_grace -= 1
            draw_doc_ui(frame)
            cv2.putText(frame, "Stabilizing...", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # -----------------------------
        # ONE HAND LOGIC
        # -----------------------------
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # PINCH TOGGLE
        d_i, d_m, d_r = pinch_metrics(hand)
        wrist = hand.landmark[0]

        wrist_history.append((wrist.x, wrist.y))
        wrist_movement = 0
        if len(wrist_history) >= 2:
            xs = [p[0] for p in wrist_history]
            ys = [p[1] for p in wrist_history]
            wrist_movement = (max(xs) - min(xs)) + (max(ys) - min(ys))

        is_pinch = (d_i < PINCH_INDEX_MAX and d_m > PINCH_OTHERS_MIN and d_r > PINCH_OTHERS_MIN)

        if is_pinch:
            pinch_hold_counter += 1
        else:
            pinch_hold_counter = 0

        if pinch_hold_counter >= PINCH_HOLD_FRAMES and wrist_movement <= WRIST_STATIONARY_MAX:
            if now - last_pinch_toggle > PINCH_DEBOUNCE:
                file_mode = not file_mode
                last_pinch_toggle = now
                pinch_hold_counter = 0
                wrist_history.clear()

                if file_mode:
                    armed = False
                    update_overlay("FILE MODE", "#00FF00")
                else:
                    armed = True
                    update_overlay("PRESENTATION", "#00AAFF")

        # -----------------------------
        # IGNORE SWIPES DURING HAND ENTRY
        # -----------------------------
        if hand_entry_counter > 0:
            # Still collect landmarks so model buffer fills after entry
            feats = extract_features_from_hand(hand)
            landmark_seq.append(feats)

            draw_doc_ui(frame)
            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # -----------------------------
        # UPDATE LANDMARK SEQUENCE FOR MODEL
        # -----------------------------
        feats = extract_features_from_hand(hand)
        landmark_seq.append(feats)

        # -----------------------------
        # VELOCITY-BASED SWIPE DETECTION (VERTICAL FOCUS)
        # -----------------------------
        wrist_buffer.append((wrist.x, wrist.y, now))

        if len(wrist_buffer) >= 3:

            # Micro-smoothing
            buf = list(wrist_buffer)
            smoothed = []
            for i in range(len(buf)):
                window = buf[max(0, i-1):min(len(buf), i+2)]
                avg_x = sum(p[0] for p in window) / len(window)
                avg_y = sum(p[1] for p in window) / len(window)
                smoothed.append((avg_x, avg_y, buf[i][2]))

            wrist_buffer = deque(smoothed, maxlen=20)

            if len(wrist_buffer) >= VEL_WINDOW:
                x0, y0, t0 = wrist_buffer[-VEL_WINDOW]
                x1, y1, t1 = wrist_buffer[-1]

                dx = x1 - x0
                dy = y1 - y0
                dt = max(0.001, t1 - t0)

                vx = dx / dt
                vy = dy / dt
                v = math.sqrt(vx*vx + vy*vy)

                angle = math.degrees(math.atan2(vy, vx))
                dist = math.sqrt(dx*dx + dy*dy)

                # ====================================================
                # 1) VERTICAL GESTURES (VELOCITY-BASED)
                # ====================================================

                vertical_gesture = None

                # Only consider vertical if displacement is big enough
                if dist > MIN_VERTICAL_DISPLACEMENT and now - last_swipe_time > SWIPE_COOLDOWN and now >= freeze_until:
                    # Up vs down thresholds
                    if 60 < angle < 120:
                        # Down
                        if v > VEL_THRESHOLD_VERTICAL_DOWN:
                            vertical_gesture = "swipe_down"
                    elif -120 < angle < -60:
                        # Up
                        if v > VEL_THRESHOLD_VERTICAL_UP:
                            vertical_gesture = "swipe_up"

                if vertical_gesture:
                    last_swipe_time = now
                    wrist_buffer.clear()

                    # FILE MODE → ONLY vertical allowed (selection)
                    if file_mode:
                        if now < filemode_freeze_until:
                            # still in freeze window after last selection
                            pass
                        else:
                            if vertical_gesture == "swipe_up":
                                current_index = (current_index - 1) % len(documents)
                            elif vertical_gesture == "swipe_down":
                                current_index = (current_index + 1) % len(documents)
                            filemode_freeze_until = now + FILEMODE_FREEZE

                    # ARMED MODE → ONLY vertical allowed (send/cancel)
                    elif armed:
                        if vertical_gesture == "swipe_up":
                            doc = documents[current_index]
                            send_document(doc)
                            armed = False
                            freeze_until = now + FREEZE_TIME
                        elif vertical_gesture == "swipe_down":
                            armed = False

                    # PRESENTATION MODE → vertical currently unused (but reserved)
                    else:
                        # You can hook vertical actions here if desired
                        pass

                # ====================================================
                # 2) HORIZONTAL GESTURES (MODEL-BASED ONLY)
                # ====================================================

                # Only in PRESENTATION mode (not file_mode, not armed)
                if (not file_mode) and (not armed):
                    # Require some horizontal dominance and displacement
                    if abs(vx) > abs(vy) and dist > MIN_LR_DISPLACEMENT and now - last_swipe_time > SWIPE_COOLDOWN and now >= freeze_until:
                        # Only run model if we have a full sequence
                        if len(landmark_seq) == SEQ_LENGTH:
                            label, conf = predict_lr_gesture(gesture_model, gesture_labels, landmark_seq)

                            if conf >= LR_CONFIDENCE_THRESHOLD and label in ("swipe_left", "swipe_right"):
                                last_swipe_time = now
                                wrist_buffer.clear()
                                landmark_seq.clear()

                                if label == "swipe_left":
                                    print("right swipe (ML)")
                                    pyautogui.press("right")
                                elif label == "swipe_right":
                                    print("left swipe (ML)")
                                    pyautogui.press("left")

        # -----------------------------
        # DRAW UI
        # -----------------------------
        draw_doc_ui(frame)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
