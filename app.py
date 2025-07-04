from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import base64
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import tempfile
from io import BytesIO
from predict import predict_image,extract_eye_region, map_coordinates_to_label
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
app = Flask(__name__)
from keras.losses import MeanSquaredError

CORS(app)  # Allow requests from mobile app
# CORS(app)  # Allow requests from mobile app
# socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket support

DATASET_DIR = 'dataset'
LOG_FILE = 'upload_log.txt'

# Load the trained model 
 # Update with your best model path
# MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'Model_1_Simple_CNN_200_50_20250604_160555.h5')
# MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'Model_1_Simple_CNN_200_50_20250606_113059.h5')
MODEL_PATH =  'tanh_Model_1_Simple_CNN_200_50_20250625_040706.h5'

MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()},compile=False)

# Grid configuration
GRID_CONFIG = {
    'rows': 9,
    'cols': 9,
}
IMAGE_SIZE = (200, 50)  # Update this to match your model's input size

# Mediapipe setup for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 384, 398, 466]

def extract_eye_region(image, landmarks, eye_landmarks, padding=30):
    h, w, _ = image.shape
    eye_points = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_landmarks]
    x_coords, y_coords = zip(*eye_points)
    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)
    return image[y_min:y_max, x_min:x_max] if (x_max - x_min > 0 and y_max - y_min > 0) else None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64-encoded image from the request
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{now}] {request.method} /predict"

        with open("predict.log", "a") as f:
            f.write(log_entry + "\n")
        data = request.get_json()
        image_data = data.get('image')
        os.makedirs('temp', exist_ok=True)

        # Generate unique filename
        filename = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jpg'
        filepath = os.path.join('temp/', filename)

        # Decode and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Decode the base64 image
        result = predict_image(filepath,MODEL)
        print(result)
        label = result['predicted_label']
        # delete the temporary file
        # os.remove(filepath)
        
        # Return the prediction
        return jsonify({
            'predictedIndex': label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    print("Request received:", request.json)
    try:
        data = request.get_json()
        image_data = data.get('image')
        x = data.get('x')
        y = data.get('y')
        index = data.get('index')
        if not image_data or not x or not y:
            return jsonify({'status': 'fail', 'message': 'Missing image or label'}), 400

        # Prepare directory for label
        save_dir = os.path.join(DATASET_DIR, 'images')
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename
        filename = f"{index}_{x}_{y}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(save_dir, filename)

        # Decode and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Log the upload
        log_entry = f"{datetime.now()} - coordinates: {x}, {y}, File: {filepath}\n"
        with open(LOG_FILE, 'a') as log:
            log.write(log_entry)

        print(f"Saved image for coordinates '{x},{y}' to {filepath}")
        return jsonify({'status': 'success', 'message': f'Saved to {filepath}'})

    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        with open(LOG_FILE, 'a') as log:
            log.write(f"{datetime.now()} - {error_msg}\n")
        return jsonify({'status': 'error', 'message': error_msg}), 500


@app.route('/predictVideo', methods=['POST'])
def predict_video():
    try:
        data = request.get_json()
        video_data = data.get('video')
        if not video_data:
            return jsonify({'status': 'fail', 'message': 'Missing video data'}), 400

        # Decode the base64 video
        video_bytes = base64.b64decode(video_data)
        video_path = os.path.join('temp', f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4")
        
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        # Process the video
        cap = cv2.VideoCapture(video_path)
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = face_mesh.process(image_rgb)

            if not results_mp.multi_face_landmarks:
                continue

            landmarks = results_mp.multi_face_landmarks[0].landmark
            left_eye_region = extract_eye_region(frame, landmarks, LEFT_EYE_LANDMARKS)
            right_eye_region = extract_eye_region(frame, landmarks, RIGHT_EYE_LANDMARKS)

            if left_eye_region is None or right_eye_region is None:
                continue

            # Predict coordinates for both eyes
            left_eye_coords = MODEL.predict(np.expand_dims(left_eye_region, axis=0))
            right_eye_coords = MODEL.predict(np.expand_dims(right_eye_region, axis=0))

            x_coord = (left_eye_coords[0][0] + right_eye_coords[0][0]) / 2
            y_coord = (left_eye_coords[0][1] + right_eye_coords[0][1]) / 2

            # Map the predicted coordinates to a label
            predicted_label = map_coordinates_to_label(x_coord, y_coord)

            results.append({
                'predicted_coordinates': {'x': x_coord, 'y': y_coord},
                'predicted_label': predicted_label
            })

        cap.release()
        os.remove(video_path)  # Clean up the temporary video file

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    app.run(port=3000, debug=True)
