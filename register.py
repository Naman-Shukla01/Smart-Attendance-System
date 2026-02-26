import cv2
import pickle
from deepface import DeepFace

name = input("Enter user name: ")

cap = cv2.VideoCapture(0)

print("Press 's' to capture face")

while True:
    ret, frame = cap.read()
    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("i'm in")
        cv2.imwrite("temp.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

# Generate embedding
embedding = DeepFace.represent(
    img_path="temp.jpg",
    model_name="Facenet"
)[0]["embedding"]

# Save embedding
with open(f"{name}.pkl", "wb") as f:
    pickle.dump(embedding, f)

print("User registered successfully!")
