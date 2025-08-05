import os
import pickle
import traceback
import urllib.request
from multiprocessing import connection
from threading import Timer

import cursor
import cv2
import mediapipe as mp
import psycopg2
import torch
import torch.nn as nn
from PIL import Image

from flask import Flask, render_template, request, json, jsonify
from flask import Response
from psycopg2.extras import RealDictCursor
from datetime import datetime, date, timedelta, time
from flask import request, redirect, url_for, flash
from dotenv import load_dotenv

import pytz
import calendar
from torchvision import models, transforms

from werkzeug.security import generate_password_hash
from ml_utils import extract_features

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load Haar cascade
conn = psycopg2.connect("postgresql://postgres.mggobpvspdsuimokmlwc:EyEcansEEoptical@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres")
cursor = conn.cursor(cursor_factory=RealDictCursor)

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")

# ✅ Set a unique and secret key
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_dev_key')

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

camera = None
camera_timer = None
camera_active = False

EYEWEAR_CLASSES = ['cat_eye', 'geometric', 'half_rim', 'oval', 'rectangle', 'round']

best_match_map = {
    "oval": ["rectangle", "square", "cat_eye", "geometric", "wayfarer"],
    "round": ["rectangle", "square", "browline", "wayfarer"],
    "square": ["round", "oval", "cat_eye"],
    "heart": ["oval", "cat_eye", "round"],
    "diamond": ["oval", "rimless", "round"],
    "oblong": ["tall", "full_rim", "aviator"],
}

def load_eyewear_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EYEWEAR_CLASSES))
    model.load_state_dict(torch.load("eyewear_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model
eyewear_model = load_eyewear_model()


def predict_eyewear_style(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = eyewear_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return EYEWEAR_CLASSES[predicted.item()]


# Load the model
with open("face_shape_model.pkl", "rb") as f:
    model = pickle.load(f)

# Get 3 most recent images in static/Photos
photo_dir = "static/Photos"
image_paths = sorted(
    [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith(".jpg")],
    key=os.path.getmtime,
    reverse=True
)[:3]

# Predict face shape for each image
for path in image_paths:
    print(f"[INFO] Predicting for {path}")
    features = extract_features(path)
    if features:
        shape = model.predict([features])[0]
        print(f"[✅] Predicted face shape: {shape}")
    else:
        print(f"[❌] Could not extract features from: {path}")


def start_camera():
    global camera, camera_active
    if not camera_active:
        print("[CAMERA STATUS] ✅ Camera is ON")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_active = True


def stop_camera():
    global camera, camera_active
    if camera_active:
        camera.release()
        camera = None
        camera_active = False
        print("[CAMERA STATUS] 🔌 Camera is OFF")


def reset_camera_timer():
    global camera_timer
    if camera_timer:
        camera_timer.cancel()
    camera_timer = Timer(2.0, stop_camera)  # 2 seconds after last access
    camera_timer.start()


# Try using Pi Camera, fallback to USB/laptop cam
try:
    camera = cv2.VideoCapture(0)  # default camera (USB/laptop)
    if not camera.isOpened():
        raise Exception("Default camera not available")
except:
    camera = None


def gen_frames():
    global camera, camera_active
    start_camera()

    if camera is None or not camera.isOpened():
        print("[CAMERA STATUS] ❌ Camera not available.")
        return

    if not camera_active:
        print("[CAMERA STATUS] ✅ Camera is ON")
        camera_active = True

    while True:
        reset_camera_timer()
        success, frame = camera.read()
        if not success:
            print("[ERROR] Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        axis_x, axis_y = 110, 140  # Radius of ellipse (horizontal, vertical)

        # Convert BGR to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        alignment_ok = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Use nose tip (landmark index 1)
                nose = face_landmarks.landmark[1]
                nose_x = int(nose.x * frame.shape[1])
                nose_y = int(nose.y * frame.shape[0])

                # Draw all landmarks
                for pt in face_landmarks.landmark:
                    x = int(pt.x * frame.shape[1])
                    y = int(pt.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Check if the nose tip is within the ellipse
                ellipse_eq = ((nose_x - center_x) ** 2) / (axis_x ** 2) + ((nose_y - center_y) ** 2) / (axis_y ** 2)
                if ellipse_eq <= 1.0:
                    alignment_ok = True

        # Draw the oval guide
        oval_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.ellipse(frame, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, oval_color, 2)

        # Message
        message = "✅ Aligned - You may take a photo" if alignment_ok else "🔴 Please align your face in the oval"
        text_color = (0, 255, 0) if alignment_ok else (0, 0, 255)
        cv2.putText(frame, message, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def crop_eyeglass_region(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Adjust the region to likely eyeglass area
        eyeglass_y = y + int(h * 0.25)
        eyeglass_h = int(h * 0.3)
        eyeglass_x = x
        eyeglass_w = w

        cropped = img[eyeglass_y:eyeglass_y + eyeglass_h, eyeglass_x:eyeglass_x + eyeglass_w]

        # Save to static/Cropped/
        cropped_dir = os.path.join("static", "Cropped")
        os.makedirs(cropped_dir, exist_ok=True)

        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        save_path = os.path.join(cropped_dir, f"cropped_{name}.jpg")
        cv2.imwrite(save_path, cropped)

        print(f"[✅] Cropped eyeglass region saved to {save_path}")
        return  # Only one face handled for now


@app.route('/')
def choose_frame():
    photo_dir = os.path.join('static', 'Photos')
    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)

    # List up to 3 recent photos
    image_paths = sorted(
        ['Photos/' + f for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=lambda x: os.path.getmtime(os.path.join('static', x)),
        reverse=True
    )[:3]

    return render_template('choose_frame.html', image_paths=image_paths)


@app.route('/open-camera')
def open_camera():
    return render_template('choose_frame.html')  # Make sure this file exists


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/take_photo')
def take_photo():
    if camera is None:
        return "Camera not available", 500

    ret, frame = camera.read()
    if not ret:
        return "Failed to capture image", 500

    # ✅ Flip the frame horizontally to match live preview
    frame = cv2.flip(frame, 1)

    # Save photo
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"static/Photos/photo_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

    # ✅ Crop the eyeglass region right after saving
    crop_eyeglass_region(filename)

    # Cleanup older images (keep only 3)
    photo_dir = os.path.join('static', 'Photos')
    photos = sorted(
        [f for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=lambda x: os.path.getmtime(os.path.join(photo_dir, x)),
        reverse=True
    )

    for extra_photo in photos[3:]:
        os.remove(os.path.join(photo_dir, extra_photo))

    return redirect(url_for('choose_frame'))


def detect_face_shape(image_path):
    features = extract_features(image_path)
    if features:
        prediction = model.predict([features])[0]
        return prediction
    else:
        return "Unknown"


@app.route('/analyze', methods=['POST'])
def analyze():
    from ml_utils import extract_features
    import pickle
    import numpy as np
    import os
    from run_fit_test import run_fit_test

    # Load ML model and scaler
    with open("face_shape_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Define paths
    photo_dir = os.path.join('static', 'Photos')
    image_paths = sorted(
        [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.endswith('.jpg')],
        key=os.path.getmtime,
        reverse=True
    )[:3]

    feature_list = []
    valid_images = []
    frame_predictions = []

    for path in image_paths:
        features = extract_features(path)
        if not features:
            continue

        feature_list.append(features)
        valid_images.append(path)

        crop_eyeglass_region(path)

        name = os.path.splitext(os.path.basename(path))[0]
        cropped_path = os.path.join('static', 'Cropped', f"cropped_{name}.jpg")

        if os.path.exists(cropped_path):
            frame = predict_eyewear_style(cropped_path)
        else:
            frame = "Unknown"

        frame_predictions.append((os.path.basename(path), frame))

    if not feature_list:
        return "Failed to extract features", 400

    features_scaled = scaler.transform(feature_list)

    # Predict class probabilities and average them
    class_probs = model.predict_proba(features_scaled)
    avg_probs = np.mean(class_probs, axis=0)

    class_labels = model.classes_
    average_prob_dict = {label: round(prob * 100, 2) for label, prob in zip(class_labels, avg_probs)}

    main_predicted_shape = class_labels[np.argmax(avg_probs)]
    main_shape_percent = round(np.max(avg_probs) * 100, 2)

    # Recommendation map
    recommended_frames = best_match_map.get(main_predicted_shape.lower(), [])
    best_match = next(((img, frame) for img, frame in frame_predictions if frame.lower() in recommended_frames), None)

    if best_match is None and frame_predictions:
        best_match = frame_predictions[0]

    # Frame descriptions dictionary
    frame_info = {
        "rectangle": "Rectangle frames complement round face shapes by adding angles.",
        "round": "Round frames are great for square or angular face shapes.",
        "oval": "Oval frames suit most face shapes and offer a balanced look.",
        "geometric": "Geometric frames work well with oval and round face shapes.",
        "cat eye": "Cat eye frames are ideal for heart-shaped faces and add a stylish lift.",
        "half rim": "Half-rim frames can complement various face shapes, particularly oval, round, and heart-shaped faces.",
        "unknown": "The frame could not be recognized. Please ensure a clear photo."
    }

    frame_description = frame_info.get(best_match[1].lower(), "No description available.") if best_match else ""

    # Run fit analysis on the 3 most recent images
    fit_scores = run_fit_test(image_paths)

    return render_template("results.html",
                           uploaded_images=[os.path.basename(p) for p in valid_images],
                           average_features=[round(f, 4) for f in np.mean(feature_list, axis=0)],
                           predicted_shape=main_predicted_shape,
                           main_shape_percent=main_shape_percent,
                           average_face_shape_probs=average_prob_dict,
                           frame_predictions=frame_predictions,
                           best_match=best_match,
                           frame_description=frame_description,
                           fit_scores=fit_scores)


@app.route('/delete_photo', methods=['POST'])
def delete_photo():
    filename = request.form.get('filename')
    if filename:
        file_path = os.path.join(app.static_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return redirect(url_for('choose_frame'))

@app.template_filter('format_time')
def format_time(value):
    return value.strftime('%I:%M %p') if isinstance(value, datetime) else value






@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_type = request.form['user_type']
        first_name = request.form['first_name']
        middle_initial = request.form['middle_initial']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        print("Form received:", request.form.to_dict())

        try:
            if user_type == 'admin':
                role = request.form.get('role')
                contact_info = request.form.get('contact_info')

                print("Inserting admin record...")
                cursor.execute("""
                    INSERT INTO admin (
                        admin_fname, admin_minitial, admin_lname,
                        admin_username, email, password, role, contact_info
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    first_name, middle_initial, last_name,
                    username, email, hashed_password, role, contact_info
                ))
            else:
                print("Inserting user record...")
                cursor.execute("""
                    INSERT INTO users (
                        user_fname, user_minitial, user_lname,
                        user_username, email, password
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    first_name, middle_initial, last_name,
                    username, email, hashed_password
                ))

            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))

        except psycopg2.Error as e:
            conn.rollback()
            print("Database error:", e.pgerror)
            flash(f"Database error: {e.pgerror}", 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')



if __name__ == '__main__':
    app.run(debug=True)
