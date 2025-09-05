from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time

app = Flask(__name__)

# Load the trained model
try:
    model_dict = pickle.load(open('rf_model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract features from hand landmarks
def extract_enhanced_features(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    z_vals = [lm.z for lm in hand_landmarks.landmark]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)
    
    
    norm_coords = []
    for lm in hand_landmarks.landmark:
        norm_x = (lm.x - min_x) / (max_x - min_x + 1e-6)
        norm_y = (lm.y - min_y) / (max_y - min_y + 1e-6)
        norm_z = (lm.z - min_z) / (max_z - min_z + 1e-6)
        norm_coords.append([norm_x, norm_y, norm_z])
    
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    fingertips = [4, 8, 12, 16, 20]  
    distances = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        dist = np.linalg.norm(tip_coords - wrist)
        distances.append(dist)
    
    
    angles = []
    finger_joints = [
        [1, 2, 3], [2, 3, 4],  
        [5, 6, 7], [6, 7, 8],  
        [9, 10, 11], [10, 11, 12],  
        [13, 14, 15], [14, 15, 16],  
        [17, 18, 19], [18, 19, 20]   
    ]
    
    for joint in finger_joints:
        v1 = np.array([
            hand_landmarks.landmark[joint[0]].x - hand_landmarks.landmark[joint[1]].x,
            hand_landmarks.landmark[joint[0]].y - hand_landmarks.landmark[joint[1]].y,
            hand_landmarks.landmark[joint[0]].z - hand_landmarks.landmark[joint[1]].z
        ])
        v2 = np.array([
            hand_landmarks.landmark[joint[2]].x - hand_landmarks.landmark[joint[1]].x,
            hand_landmarks.landmark[joint[2]].y - hand_landmarks.landmark[joint[1]].y,
            hand_landmarks.landmark[joint[2]].z - hand_landmarks.landmark[joint[1]].z
        ])
        
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        else:
            angles.append(0)
    
    
    palm_center = np.mean([
        [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z],  # wrist
        [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z],  # index MCP
        [hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z]  # pinky MCP
    ], axis=0)
    
    relative_positions = []
    for tip in fingertips:
        tip_coords = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
        rel_pos = tip_coords - palm_center
        relative_positions.extend(rel_pos)
    
    
    hand_width = max_x - min_x
    hand_height = max_y - min_y
    hand_area = hand_width * hand_height
    
    
    features = []
    
    
    for coord in norm_coords:
        features.extend(coord)
    
    
    features.extend(distances)
    features.extend(angles)
    features.extend(relative_positions)
    
    
    features.append(hand_area)
    
    return features

def generate_frames():
    
    camera = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY, None]:
        try:
            if backend is None:
                camera = cv2.VideoCapture(0)
            else:
                camera = cv2.VideoCapture(0, backend)
                
            if camera.isOpened():
                print(f"Camera opened successfully with backend: {backend}")
                break
        except Exception as e:
            print(f"Failed to open camera with backend {backend}: {e}")
            if camera:
                camera.release()
                camera = None
    
    if camera is None or not camera.isOpened():
        
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera not available!", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)  
    
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    
    last_prediction = None
    prediction_confidence = 0
    prediction_time = time.time()
    prediction_cooldown = 0.5 
    
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame, 1)
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        results = hands.process(frame_rgb)
        
        
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            
            current_time = time.time()
            if current_time - prediction_time >= prediction_cooldown:
                try:
                    
                    features = extract_enhanced_features(results.multi_hand_landmarks[0])
                    
                    if model is not None:
                       
                        num_features = model.n_features_in_
                        if len(features) > num_features:
                            features = features[:num_features]
                        elif len(features) < num_features:
                            features = np.pad(features, (0, num_features - len(features)), 'constant')
                        
                        
                        prediction = model.predict([np.asarray(features)])[0]
                        
                        
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba([np.asarray(features)])[0]
                            confidence = np.max(proba) * 100
                            
                            
                            last_prediction = prediction
                            prediction_confidence = confidence
                            prediction_time = current_time
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        
        if last_prediction:
            
            text = f"Prediction: {last_prediction}"
            if prediction_confidence > 0:
                text += f" ({prediction_confidence:.1f}%)"
                
            
            cv2.rectangle(frame, (10, frame.shape[0] - 60), (350, frame.shape[0] - 10), (0, 0, 0), -1)
            
            
            cv2.putText(
                frame, text, (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting ASL Sign Language Recognition Server...")
    print(f"Model loaded: {model is not None}")
    
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except OSError:
        print("Port 5000 is in use, trying port 5001...")
        app.run(debug=True, host='0.0.0.0', port=5001)