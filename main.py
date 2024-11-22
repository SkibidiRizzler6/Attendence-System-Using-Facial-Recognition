import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import os
import mediapipe as mp
import numpy as np
import mysql.connector
from sklearn.preprocessing import LabelEncoder
import pickle

# MySQL Connection Setup
def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="pass@1234",  # Replace with your MySQL password
        database="face_recognition_system"  # Your database name
    )

def authenticate_admin(username, password):
    db_connection = create_connection()
    cursor = db_connection.cursor()

    try:
        query = "SELECT * FROM admins WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        
        # Fetch the first matching result
        result = cursor.fetchone()

        # Check if the result is valid
        if result is not None:
            return True
        else:
            return False
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.fetchall()  # This ensures all results are fetched
        cursor.close()
        db_connection.close()


# Initialize face detection module and face recognizer
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)

# Log attendance into MySQL
def log_attendance(student_name):
    try:
        student_name = str(student_name)  # Ensure it's a string
        conn = create_connection()
        cursor = conn.cursor()

        # Corrected query to match the column 'student_name'
        query = "INSERT INTO attendance (student_name) VALUES (%s)"
        cursor.execute(query, (student_name,))
        conn.commit()
        print(f"Attendance logged for {student_name}")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()


# Take Attendance
def take_attendance():
    cap = cv2.VideoCapture(0)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_recognizer.yml")

    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min, y_min = int(bbox.xmin * w), int(bbox.ymin * h)
                x_max, y_max = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                face_image = frame[y_min:y_max, x_min:x_max]
                face_image_resized = cv2.resize(face_image, (150, 150))
                face_image_gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)

                # Predict using the trained model
                label, confidence = face_recognizer.predict(face_image_gray)
                name = label_encoder.inverse_transform([label])[0]

                # Show name and confidence
                if confidence < 100:  # If confidence is good, mark attendance
                    cv2.putText(frame, f"Attendance marked for {name}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    # Log attendance to MySQL
                    log_attendance(name)

                    # Close camera and window after attendance is marked
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Exit the function once attendance is logged

        cv2.imshow("Take Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def train_model(student_name):
    cap = cv2.VideoCapture(0)

    # Create folder to save training images if it doesn't exist
    os.makedirs("TrainingImages", exist_ok=True)

    sample_count = 0
    known_faces = []
    labels = []
    label_encoder = LabelEncoder()

    print("Starting training...")

    while sample_count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min, y_min = int(bbox.xmin * w), int(bbox.ymin * h)
                x_max, y_max = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                face_image = frame[y_min:y_max, x_min:x_max]
                face_image = cv2.resize(face_image, (150, 150))
                face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

                # Show face markers and collect training data
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Collect data for training
                known_faces.append(face_image_gray)
                labels.append(student_name)  # Use the entered name as the label for the training image

                sample_count += 1

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    labels = label_encoder.fit_transform(labels)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(known_faces, np.array(labels))

    # Save the trained model and label encoder
    face_recognizer.save("face_recognizer.yml")
    with open("label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    cap.release()
    cv2.destroyAllWindows()

    print("Training complete")

def start_training(student_name):
    threading.Thread(target=train_model, args=(student_name,), daemon=True).start()

def start_attendance():
    threading.Thread(target=take_attendance, daemon=True).start()

def create_gui():
    window = tk.Tk()
    window.geometry("900x600")
    window.title("Soham Mukherjee's Project for 12th Grade")

    # Styling
    window.config(bg="#f0f0f0")
    title_label = tk.Label(window, text="Face Recognition System", font=("Arial", 24, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)

    # Admin Authentication GUI
    def authenticate_admin_gui():
        admin_username = admin_username_entry.get()
        admin_password = admin_password_entry.get()

        if authenticate_admin(admin_username, admin_password):
            messagebox.showinfo("Success", "Admin authenticated successfully!")
            # Hide authentication form and show main buttons
            authentication_frame.pack_forget()
            main_frame.pack(pady=20)
        else:
            messagebox.showerror("Error", "Invalid admin credentials.")

    # Authentication Frame
    authentication_frame = tk.Frame(window, bg="#f0f0f0")
    admin_username_label = tk.Label(authentication_frame, text="Admin Username:", font=("Arial", 14), bg="#f0f0f0")
    admin_username_label.grid(row=0, column=0, padx=10, pady=5)
    admin_username_entry = tk.Entry(authentication_frame, font=("Arial", 14))
    admin_username_entry.grid(row=0, column=1, padx=10, pady=5)

    admin_password_label = tk.Label(authentication_frame, text="Admin Password:", font=("Arial", 14), bg="#f0f0f0")
    admin_password_label.grid(row=1, column=0, padx=10, pady=5)
    admin_password_entry = tk.Entry(authentication_frame, show="*", font=("Arial", 14))
    admin_password_entry.grid(row=1, column=1, padx=10, pady=5)

    authenticate_button = tk.Button(authentication_frame, text="Authenticate", font=("Arial", 14), command=authenticate_admin_gui, bg="#4CAF50", fg="white", relief="flat")
    authenticate_button.grid(row=2, columnspan=2, pady=20)

    authentication_frame.pack(pady=50)

    # Main Action Frame (After authentication)
    main_frame = tk.Frame(window, bg="#f0f0f0")

    student_name_label = tk.Label(main_frame, text="Enter Student's Name:", font=("Arial", 14), bg="#f0f0f0")
    student_name_label.pack(pady=10)

    student_name_entry = tk.Entry(main_frame, font=("Arial", 14))
    student_name_entry.pack(pady=10)

    train_button = tk.Button(main_frame, text="Train Model", font=("Arial", 14), command=lambda: start_training(student_name_entry.get()), bg="#FF5722", fg="white", relief="flat")
    train_button.pack(pady=10)

    attendance_button = tk.Button(main_frame, text="Take Attendance", font=("Arial", 14), command=start_attendance, bg="#03A9F4", fg="white", relief="flat")
    attendance_button.pack(pady=10)

    window.mainloop()

# Run the application
create_gui()
