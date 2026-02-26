import cv2, time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
from deepface import DeepFace
from numpy.linalg import norm
from math import hypot
import csv
from datetime import datetime
from ultralytics import YOLO

# -------------------------
# Load Database
# -------------------------
database = {}
attendance_cooldown = 60*30  # 30 minutes cooldown for attendance logging
last_logged = {}

for file in os.listdir():
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(file, "rb") as f:
            database[name] = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def draw_face_box(frame, landmarks, w, h, color):
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

def log_attendance(name, score, status):
    global last_logged
    current_time = time.time()

    if name in last_logged:
        if current_time - last_logged[name] < attendance_cooldown:
            return

    last_logged[name] = current_time
    file_exists = os.path.isfile("attendance.csv")

    with open("attendance.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Score", "Status"])

        now = datetime.now()
        writer.writerow([
            name,
            now.strftime("%d-%m-%Y"),
            now.strftime("%H:%M:%S"),
            round(score, 2),
            status
        ])


# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("yolov8n.pt")   # lightweight & fast

DEVICE_CLASSES = ["cell phone", "laptop"]

# -------------------------
# MediaPipe Setup
# -------------------------
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.20
THRESHOLD = 0.7
lock_time = 5  # seconds

def get_landmark_xy(landmarks, index, w, h):
    lm = landmarks[index]
    return np.array([int(lm.x * w), int(lm.y * h)])

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    p1 = get_landmark_xy(landmarks, eye_indices[0], w, h)
    p2 = get_landmark_xy(landmarks, eye_indices[1], w, h)
    p3 = get_landmark_xy(landmarks, eye_indices[2], w, h)
    p4 = get_landmark_xy(landmarks, eye_indices[3], w, h)
    p5 = get_landmark_xy(landmarks, eye_indices[4], w, h)
    p6 = get_landmark_xy(landmarks, eye_indices[5], w, h)

    vertical1 = hypot(*(p2 - p6))
    vertical2 = hypot(*(p3 - p5))
    horizontal = hypot(*(p1 - p4))

    return (vertical1 + vertical2) / (2.0 * horizontal)

# -------------------------
# Main Logic
# -------------------------
cap = cv2.VideoCapture(0)

blink_count = 0
blink_detected = False
verified = False
identity_verified = False
recognition_done = False
last_verified_time = 0

frame_counter = 0
device_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    device_detected = False
    label = ""

    # -------------------------
    # DEVICE DETECTION (Every 3 Frames)
    # -------------------------
    if frame_counter % 3 == 0:
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected_label = model.names[cls_id]

                if detected_label in DEVICE_CLASSES:
                    device_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, detected_label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,0,255),
                                2)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    current_time = time.time()

    # -------------------------
    # IF DEVICE FOUND → BLOCK EVERYTHING
    # -------------------------
    if device_detected:
        cv2.putText(frame,
                    "Remove electronic devices from camera view",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

        # Reset verification if device appears
        verified = False
        identity_verified = False
        recognition_done = False
        blink_count = 0
        blink_detected = False

    # -------------------------
    # NORMAL FLOW (Only If Safe)
    # -------------------------
    elif result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # Draw box based on verification
        if not verified:
            draw_face_box(frame, landmarks, w, h, (0, 0, 255))
        else:
            if identity_verified:
                draw_face_box(frame, landmarks, w, h, (0, 255, 0))
            else:
                draw_face_box(frame, landmarks, w, h, (0, 0, 255))

        # -------------------------
        # STEP 1: LIVENESS
        # -------------------------
        if not verified:
            ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)

            if ear < EAR_THRESHOLD:
                blink_detected = True
            else:
                if blink_detected:
                    blink_count += 1
                    blink_detected = False

            cv2.putText(frame, f"Blinks: {blink_count}", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, "Blink 2 Times to Verify", (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if blink_count >= 2:
                verified = True
                last_verified_time = current_time

        # -------------------------
        # STEP 2: RECOGNITION (RUN ONCE)
        # -------------------------
        else:
            if not recognition_done:
                try:
                    embedding = DeepFace.represent(
                        img_path=frame,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]

                    best_match = "Unknown"
                    highest_similarity = 0

                    for name, db_embedding in database.items():
                        similarity = cosine_similarity(embedding, db_embedding)
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            best_match = name

                    if highest_similarity > THRESHOLD:
                        identity_verified = True
                        confidence = round(highest_similarity * 100, 1)
                        label = f"{best_match} - {confidence}%"
                        log_attendance(best_match, highest_similarity, "Verified")
                    else:
                        identity_verified = False
                        label = "Unknown"
                       # log_attendance("Unknown", highest_similarity, "Unknown")

                    recognition_done = True

                except:
                    label = "Processing..."

            cv2.putText(frame, label, (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Reset after lock time
            if current_time - last_verified_time > lock_time:
                verified = False
                identity_verified = False
                recognition_done = False
                blink_count = 0
                blink_detected = False

    cv2.imshow("Secure Face Authentication", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()