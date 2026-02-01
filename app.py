import cv2
import mediapipe as mp
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*"
)

@app.route("/")
def index():
    return render_template("index.html")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def camera_loop():
    print("Camera loop started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "NONE"

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            fingers = [
                lm[8].y < lm[6].y,   # index
                lm[12].y < lm[10].y, # middle
                lm[16].y < lm[14].y, # ring
                lm[20].y < lm[18].y  # pinky
            ]

            if all(fingers):
                gesture = "OPEN_PALM"   # ✋
            elif not any(fingers):
                gesture = "FIST"        # ✊
            elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                gesture = "PEACE"       # ✌️
            elif fingers[0] and fingers[3] and not fingers[1] and not fingers[2]:
                gesture = "LOVE"        # 🤟

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        socketio.emit("gesture", {"value": gesture})
        print("sent:", gesture)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    socketio.run(app, host="127.0.0.1", port=5000, debug=False)
