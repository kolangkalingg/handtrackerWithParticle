import cv2
import mediapipe as mp
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import math
import base64

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")

# KONFIGURASI MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_sibi_gesture(lm):
    tips = [4, 8, 12, 16, 20]
    wrist = lm[0]
    middle_mcp = lm[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    
    orientation = "UP"
    if abs(dx) > abs(dy): orientation = "SIDE"
    elif dy > 0: orientation = "DOWN"
    
    fingers = []
    # Cek Jempol
    if get_dist(lm[4], lm[5]) > 0.05: fingers.append(1)
    else: fingers.append(0)

    # Cek 4 Jari Lain
    for i in range(1, 5):
        if orientation == "UP": isOpen = lm[tips[i]].y < lm[tips[i]-2].y
        elif orientation == "DOWN": isOpen = lm[tips[i]].y > lm[tips[i]-2].y
        else: isOpen = get_dist(lm[tips[i]], wrist) > get_dist(lm[tips[i]-2], wrist)
        fingers.append(1 if isOpen else 0)
    
    total_fingers = fingers.count(1)

    # --- DETEKSI METAL (Start Game) ---
    if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0:
        return "METAL"

    # --- LOGIKA SIBI ---
    if orientation == "SIDE":
        if fingers[1] == 1 and fingers[2] == 0: return "G"
        if fingers[1] == 1 and fingers[2] == 1: return "H"
        if fingers[1] == 1 and fingers[0] == 0: return "Z"
        return "UNKNOWN"

    if orientation == "DOWN":
        if fingers[1] == 1 and fingers[2] == 1: return "P"
        if fingers[1] == 1: return "Q"
        return "UNKNOWN"

    if total_fingers == 0 or (total_fingers == 1 and fingers[0] == 1):
        if fingers[0] == 1 and lm[4].y < lm[6].y: return "A" 
        if lm[4].x > lm[5].x and lm[4].x < lm[17].x: return "S" # Kepalan (Jempol silang)
        if get_dist(lm[8], lm[0]) < 0.1: return "E" # Kepalan (Cakar/Kuku)
        
        th_y = lm[4].y
        if th_y < lm[13].y: return "T"
        if th_y < lm[17].y: return "N"
        return "M"

    if fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 1]: return "B"
    
    if fingers == [1, 1, 1, 1, 1] or fingers == [1, 1, 0, 0, 0]:
        dist_idx_thumb = get_dist(lm[4], lm[8])
        if dist_idx_thumb < 0.05: return "O"
        if dist_idx_thumb < 0.15: return "C"

    if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]: return "D"
    
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
        if lm[8].x > lm[12].x: return "R"
        return "F"

    if fingers[4] == 1 and fingers[1] == 0:
        if lm[20].x > lm[4].x: return "I"
        return "J"

    if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1: return "K"
    if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0: return "L"

    # --- PERBAIKAN U & V ---
    if fingers[1]==1 and fingers[2]==1 and fingers[3]==0:
        # Perlebar threshold jarak untuk V, agar U (rapat) lebih mudah
        # Jika jarak ujung telunjuk & tengah > 0.065, dianggap V. Jika rapat, U.
        if get_dist(lm[8], lm[12]) > 0.065: return "V"
        return "U"
    
    if fingers[1]==1 and fingers[2]==1 and fingers[3]==1: return "W"
    if fingers[0]==1 and fingers[4]==1: return "Y"

    return "NONE"

def camera_loop():
    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        data = {"gesture": "NO_HAND", "x": 0.5, "y": 0.5, "image": frame_base64}

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            gesture = detect_sibi_gesture(lm)
            cx = lm[9].x
            cy = lm[9].y
            data["gesture"] = gesture
            data["x"] = cx
            data["y"] = cy

        socketio.emit("update_data", data)
        if cv2.waitKey(1) & 0xFF == 27: break

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)