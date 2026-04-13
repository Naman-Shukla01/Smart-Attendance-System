import csv
import os
import pickle
import threading
import time
import tkinter as tk
from datetime import datetime
from math import hypot
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy.linalg import norm
from ultralytics import YOLO


# -------------------------
# Configuration
# -------------------------
ATTENDANCE_COOLDOWN = 60 * 30
DEVICE_CLASSES = ["cell phone", "laptop"]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.20
THRESHOLD = 0.7
LOCK_TIME = 5

# -------------------------
# GUI Dropdown Options
# -------------------------
# Edit these lists as needed.
TEACHER_OPTIONS = ["KKR", "Teacher_2", "Teacher_3"]
CLASS_OPTIONS = ["IT61", "IT62", "IT63"]


# -------------------------
# Load Database
# -------------------------
def load_database():
    database = {}
    for file_name in os.listdir():
        if file_name.endswith(".pkl"):
            name = file_name.replace(".pkl", "")
            with open(file_name, "rb") as file:
                database[name] = pickle.load(file)
    return database


# -------------------------
# Utility Functions
# -------------------------
def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))


def draw_face_box(frame, landmarks, width, height, color):
    x_coords = [int(landmark.x * width) for landmark in landmarks]
    y_coords = [int(landmark.y * height) for landmark in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


def get_landmark_xy(landmarks, index, width, height):
    landmark = landmarks[index]
    return np.array([int(landmark.x * width), int(landmark.y * height)])


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    p1 = get_landmark_xy(landmarks, eye_indices[0], width, height)
    p2 = get_landmark_xy(landmarks, eye_indices[1], width, height)
    p3 = get_landmark_xy(landmarks, eye_indices[2], width, height)
    p4 = get_landmark_xy(landmarks, eye_indices[3], width, height)
    p5 = get_landmark_xy(landmarks, eye_indices[4], width, height)
    p6 = get_landmark_xy(landmarks, eye_indices[5], width, height)

    vertical1 = hypot(*(p2 - p6))
    vertical2 = hypot(*(p3 - p5))
    horizontal = hypot(*(p1 - p4))

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# -------------------------
# GUI + Recognition App
# -------------------------
class AttendanceApp:
    def __init__(self):
        # ---------------------------
        # Load YOLO Model
        # ---------------------------
        self.database = load_database()
        self.model = YOLO("yolov8n.pt")

        # -------------------------
        # MediaPipe Setup
        # -------------------------
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.running = False
        self.worker = None
        self.last_logged = {}
        self.present_people = set()
        self.unknown_count = 0
        self.attendance_filename = ""
        self.teacher_name = ""
        self.subject_name = ""
        self.class_name = ""
        self.batch_name = ""
        self.today_date = ""

        self.root = tk.Tk()
        self.root.title("Smart Face Attendance System")
        self.root.geometry("500x460")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_gui()

    # -------------------------
    # GUI Layout
    # -------------------------
    def build_gui(self):
        container = tk.Frame(self.root, padx=20, pady=20)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Teacher Name").pack(anchor="w")
        self.teacher_combo = ttk.Combobox(container, values=TEACHER_OPTIONS)
        self.teacher_combo.pack(fill="x", pady=(0, 10))
        if TEACHER_OPTIONS:
            self.teacher_combo.set(TEACHER_OPTIONS[0])

        tk.Label(container, text="Subject").pack(anchor="w")
        self.subject_entry = tk.Entry(container)
        self.subject_entry.pack(fill="x", pady=(0, 10))

        tk.Label(container, text="Class").pack(anchor="w")
        self.class_combo = ttk.Combobox(container, values=CLASS_OPTIONS)
        self.class_combo.pack(fill="x", pady=(0, 10))
        if CLASS_OPTIONS:
            self.class_combo.set(CLASS_OPTIONS[0])

        tk.Label(container, text="Batch").pack(anchor="w")
        self.batch_entry = tk.Entry(container)
        self.batch_entry.pack(fill="x", pady=(0, 16))

        register_actions = tk.Frame(container)
        register_actions.pack(fill="x", pady=(0, 16))
        tk.Button(
            register_actions, text="Register New User", command=self.open_register_window
        ).pack(side="left")

        controls = tk.Frame(container)
        controls.pack(fill="x", pady=(0, 16))
        tk.Button(controls, text="Start System", command=self.start_system).pack(
            side="left"
        )
        tk.Button(controls, text="Stop System", command=self.stop_system).pack(
            side="left", padx=(10, 0)
        )

        self.present_label = tk.Label(container, text="Present: 0", font=("Arial", 12))
        self.present_label.pack(anchor="w", pady=4)

        self.unknown_label = tk.Label(container, text="Unknown: 0", font=("Arial", 12))
        self.unknown_label.pack(anchor="w", pady=4)

        self.status_label = tk.Label(
            container, text="Status: Waiting", font=("Arial", 12)
        )
        self.status_label.pack(anchor="w", pady=4)

        self.file_label = tk.Label(
            container,
            text="Attendance File: Not started",
            wraplength=440,
            justify="left",
        )
        self.file_label.pack(anchor="w", pady=(12, 0))

    # -------------------------
    # Registration Flow
    # -------------------------
    def open_register_window(self):
        if hasattr(self, "register_window") and self.register_window.winfo_exists():
            self.register_window.focus()
            return

        self.register_window = tk.Toplevel(self.root)
        self.register_window.title("Register New User")
        self.register_window.geometry("400x160")

        wrapper = tk.Frame(self.register_window, padx=20, pady=20)
        wrapper.pack(fill="both", expand=True)

        tk.Label(wrapper, text="Enter Name").pack(anchor="w")
        self.register_name_entry = tk.Entry(wrapper)
        self.register_name_entry.pack(fill="x", pady=(0, 12))

        tk.Button(
            wrapper, text="Start Registration", command=self.register_user
        ).pack(anchor="w")

    def register_user(self):
        name = self.register_name_entry.get().strip().replace(" ", "_")
        if not name:
            messagebox.showerror("Error", "Please enter a name to register.")
            return

        file_path = f"{name}.pkl"
        if os.path.exists(file_path):
            overwrite = messagebox.askyesno(
                "Confirm Overwrite",
                f"{name} already exists. Overwrite?",
            )
            if not overwrite:
                return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        captured_frame = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Camera Error", "Failed to read from camera.")
                    return

                cv2.putText(
                    frame,
                    "Press 'S' to capture, 'Q' to cancel",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Register Face", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    captured_frame = frame.copy()
                    break
                if key == ord("q"):
                    return
        finally:
            cap.release()
            cv2.destroyWindow("Register Face")

        if captured_frame is None:
            return

        try:
            embedding = DeepFace.represent(
                img_path=captured_frame,
                model_name="Facenet",
                enforce_detection=False,
            )[0]["embedding"]
        except Exception:
            messagebox.showerror("Error", "Failed to generate face embedding.")
            return

        with open(file_path, "wb") as file:
            pickle.dump(embedding, file)

        self.database[name] = embedding
        self.update_gui(status=f"Registered: {name}")
        messagebox.showinfo("Success", f"{name} registered successfully.")
        if hasattr(self, "register_window") and self.register_window.winfo_exists():
            self.register_window.destroy()

    # -------------------------
    # GUI Update Helper
    # -------------------------
    def update_gui(self, status=None):
        def apply_updates():
            self.present_label.config(text=f"Present: {len(self.present_people)}")
            self.unknown_label.config(text=f"Unknown: {self.unknown_count}")
            if status is not None:
                self.status_label.config(text=f"Status: {status}")
            if self.attendance_filename:
                self.file_label.config(
                    text=f"Attendance File: {self.attendance_filename}"
                )

        self.root.after(0, apply_updates)

    # -------------------------
    # SESSION DETAILS INPUT
    # -------------------------

    def build_attendance_file(self):
        self.teacher_name = self.teacher_combo.get().strip().replace(" ", "_")
        self.subject_name = self.subject_entry.get().strip().replace(" ", "_")
        self.class_name = self.class_combo.get().strip().replace(" ", "_")
        self.batch_name = self.batch_entry.get().strip().replace(" ", "_")

        if not all(
            [self.teacher_name, self.subject_name, self.class_name, self.batch_name]
        ):
            messagebox.showerror("Error", "Teacher, subject, class and batch are required.")
            return False

        self.today_date = datetime.now().strftime("%d-%m-%Y")
        main_folder = "Attendance_Records"
        batch_folder = os.path.join(main_folder, self.batch_name)
        date_folder = os.path.join(batch_folder, self.today_date)
        os.makedirs(date_folder, exist_ok=True)

        self.attendance_filename = os.path.join(
            date_folder,
            f"Attendance_{self.class_name}_{self.subject_name}_{self.teacher_name}.csv",
        )
        self.present_people = set()
        self.unknown_count = 0
        self.last_logged = {}
        return True

    # -------------------------
    # Attendance Logging
    # -------------------------
    def log_attendance(self, name, score, status):
        current_time = time.time()
        if name in self.last_logged:
            if current_time - self.last_logged[name] < ATTENDANCE_COOLDOWN:
                return

        self.last_logged[name] = current_time
        file_exists = os.path.isfile(self.attendance_filename)

        with open(self.attendance_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Class", self.class_name])
                writer.writerow(["Subject", self.subject_name])
                writer.writerow(["Teacher", self.teacher_name])
                writer.writerow(["Date", self.today_date])
                writer.writerow([])
                writer.writerow(["Name", "Time", "Score", "Status"])

            now = datetime.now()
            writer.writerow([name, now.strftime("%H:%M:%S"), round(score, 2), status])

    # -------------------------
    # Recognition State Reset
    # -------------------------
    def reset_state(self):
        return {
            "blink_count": 0,
            "blink_detected": False,
            "verified": False,
            "identity_verified": False,
            "recognition_done": False,
            "last_verified_time": 0.0,
            "frame_counter": 0,
            "label": "",
        }

    # -------------------------
    # Start / Stop Controls
    # -------------------------
    def start_system(self):
        if self.running:
            messagebox.showinfo("Running", "Recognition is already running.")
            return

        if not self.build_attendance_file():
            return

        self.running = True
        self.update_gui(status="Starting camera")
        self.worker = threading.Thread(target=self.run_camera, daemon=True)
        self.worker.start()

    def stop_system(self):
        self.running = False
        self.update_gui(status="Stopped")

    def on_close(self):
        self.stop_system()
        self.root.after(200, self.root.destroy)

    # -------------------------
    # Main Logic
    # -------------------------
    def run_camera(self):
        state = self.reset_state()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.running = False
            self.root.after(
                0, lambda: messagebox.showerror("Camera Error", "Could not open camera.")
            )
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_gui(status="Camera read failed")
                    break

                state["frame_counter"] += 1
                device_detected = False
                status_text = "Waiting for face"

                # -------------------------
                # DEVICE DETECTION (Every 3 Frames)
                # -------------------------
                if state["frame_counter"] % 3 == 0:
                    results = self.model(frame, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            detected_label = self.model.names[class_id]
                            if detected_label in DEVICE_CLASSES:
                                device_detected = True
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(
                                    frame,
                                    detected_label,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2,
                                )

                height, width, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)
                current_time = time.time()

                # -------------------------
                # IF DEVICE FOUND -> BLOCK EVERYTHING
                # -------------------------
                if device_detected:
                    cv2.putText(
                        frame,
                        "Remove electronic devices from camera view",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    status_text = "Device detected"
                    state = self.reset_state()

                # -------------------------
                # NORMAL FLOW (Only If Safe)
                # -------------------------
                elif result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    box_color = (0, 255, 0) if state["identity_verified"] else (0, 0, 255)
                    draw_face_box(frame, landmarks, width, height, box_color)

                    # -------------------------
                    # STEP 1: LIVENESS
                    # -------------------------
                    if not state["verified"]:
                        ear = eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
                        if ear < EAR_THRESHOLD:
                            state["blink_detected"] = True
                        elif state["blink_detected"]:
                            state["blink_count"] += 1
                            state["blink_detected"] = False

                        cv2.putText(
                            frame,
                            f"Blinks: {state['blink_count']}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "Blink 2 Times to Verify",
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                        status_text = "Blink 2 times to verify"

                        if state["blink_count"] >= 2:
                            state["verified"] = True
                            state["last_verified_time"] = current_time
                            status_text = "Liveness verified"

                    # -------------------------
                    # STEP 2: RECOGNITION (RUN ONCE)
                    # -------------------------
                    else:
                        if not state["recognition_done"]:
                            try:
                                embedding = DeepFace.represent(
                                    img_path=frame,
                                    model_name="Facenet",
                                    enforce_detection=False,
                                )[0]["embedding"]

                                best_match = "Unknown"
                                highest_similarity = 0

                                for name, db_embedding in self.database.items():
                                    similarity = cosine_similarity(embedding, db_embedding)
                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        best_match = name

                                if highest_similarity > THRESHOLD:
                                    confidence = round(highest_similarity * 100, 1)
                                    state["identity_verified"] = True
                                    state["label"] = f"{best_match} - {confidence}%"
                                    self.present_people.add(best_match)
                                    self.log_attendance(
                                        best_match, highest_similarity, "Verified"
                                    )
                                    status_text = f"Recognized: {best_match}"
                                else:
                                    state["identity_verified"] = False
                                    state["label"] = "Unknown"
                                    self.unknown_count += 1
                                    status_text = "Unknown face"

                                state["recognition_done"] = True
                            except Exception:
                                state["label"] = "Processing..."
                                status_text = "Processing face"

                        cv2.putText(
                            frame,
                            state["label"],
                            (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )

                        if current_time - state["last_verified_time"] > LOCK_TIME:
                            state = self.reset_state()
                            status_text = "Verification reset"

                self.update_gui(status=status_text)
                cv2.imshow("Secure Face Authentication", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    self.update_gui(status="Stopped from camera window")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False

    # -------------------------
    # App Entry
    # -------------------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AttendanceApp().run()
