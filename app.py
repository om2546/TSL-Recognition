import cv2
import threading
import time
import numpy as np
import mediapipe as mp
from keras.models import load_model
import math
from scipy.spatial import distance
import tensorflow as tf
import os
from flask import Flask, Response, render_template, jsonify
import base64

# Suppress TensorFlow and Mediapipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Load model
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
# model = load_model(ABSOLUTE_PATH + 'gru_2layer_seed0.keras')
model = load_model(ABSOLUTE_PATH + 'lstm_2layer_seed0.keras')

# Global variables
START = time.time()
pred_result = '[no action]'
prob = "0.00"
frame_queue = []
running = False
webcam_thread = None
predict_thread = None

# Words list for prediction
words_list = ['Hello',
              'Thank you',
              'Sorry',
              'Yes',
              'No',
              'No problem',
              'Fun',
              'You',
              'I',
              'Sports',
              'Tuesday',
              'Fine',
              'Sick',
              'Study',
              'Sign language',
              'What',
              'Like',
              'How much',
              'Football',
              'Fruits',
              'Watermelon',
              'Eat',
              'Hungry',
              'Sick/Cold',
              'Noodles',
              'Tomorrow',
              'Headache',
              'Age',
              '30',
              'Name',
              '[no action]']

def extract_landmarks(results,
                      face_indices=[[0, 2], [2, 7], [0, 9], [0, 5], [5, 8], [0, 10]],
                      right_body_indices=[[11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [0, 15]],
                      left_body_indices=[[12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [0, 16]]):
    # Function to extract and concatenate all landmarks into a single vector
    
    landmarks = []
    
    # Body Pose Landmarks
    if results.pose_landmarks:
        
        # Center point
        center = [(results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2.0,
                  (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2.0]
        normdist = distance.euclidean(center,
                                      [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
        
        # 18 angles + 18 distances for body & face
        for indices_list in [face_indices, right_body_indices, left_body_indices]:
            for pair in indices_list:
                landmarks.extend([
                                math.atan2(results.pose_landmarks.landmark[pair[1]].y - results.pose_landmarks.landmark[pair[0]].y,
                                           results.pose_landmarks.landmark[pair[1]].x - results.pose_landmarks.landmark[pair[0]].x),
                                
                                distance.euclidean([results.pose_landmarks.landmark[pair[0]].x, results.pose_landmarks.landmark[pair[0]].y],
                                                   [results.pose_landmarks.landmark[pair[1]].x, results.pose_landmarks.landmark[pair[1]].y])/normdist
                                ])
        
        # 2 angles from center
        landmarks.extend([
                        math.atan2(center[1] - results.pose_landmarks.landmark[11].y,
                                   center[0] - results.pose_landmarks.landmark[11].x),
                        
                        math.atan2(center[1] - results.pose_landmarks.landmark[12].y,
                                   center[0] - results.pose_landmarks.landmark[12].x)
                        ])
        
        # Left Hand Landmarks
        if results.left_hand_landmarks:
            for finger in range(5):
                landmarks.extend([
                                math.atan2(results.left_hand_landmarks.landmark[1+finger*4].y - results.left_hand_landmarks.landmark[0].y,
                                           results.left_hand_landmarks.landmark[1+finger*4].x - results.left_hand_landmarks.landmark[0].x),
                                
                                distance.euclidean([results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y],
                                                   [results.left_hand_landmarks.landmark[1+finger*4].x, results.left_hand_landmarks.landmark[1+finger*4].y])/normdist])
                
                for joint in [2,3,4]:
                    landmarks.extend([
                                     math.atan2(results.left_hand_landmarks.landmark[joint+finger*4].y - results.left_hand_landmarks.landmark[joint+finger*4-1].y,
                                                results.left_hand_landmarks.landmark[joint+finger*4].x - results.left_hand_landmarks.landmark[joint+finger*4-1].x),
                                    
                                     distance.euclidean([results.left_hand_landmarks.landmark[joint+finger*4-1].x, results.left_hand_landmarks.landmark[joint+finger*4-1].y],
                                                        [results.left_hand_landmarks.landmark[joint+finger*4].x, results.left_hand_landmarks.landmark[joint+finger*4].y])/normdist])
                
            for fingertips in range(1,5):
                landmarks.extend([
                                math.atan2(results.left_hand_landmarks.landmark[fingertips*4].y - results.left_hand_landmarks.landmark[(fingertips+1)*4].y,
                                           results.left_hand_landmarks.landmark[fingertips*4].x - results.left_hand_landmarks.landmark[(fingertips+1)*4].x),
                                
                                distance.euclidean([results.left_hand_landmarks.landmark[(fingertips+1)*4].x, results.left_hand_landmarks.landmark[(fingertips+1)*4].y],
                                                   [results.left_hand_landmarks.landmark[fingertips*4].x, results.left_hand_landmarks.landmark[fingertips*4].y])/normdist])
        else:  
            landmarks.extend([0] * 48)  # Placeholder for 24 hand landmarks * 2 attributes            
        
        # Right Hand Landmarks
        if results.right_hand_landmarks:
            for finger in range(5):
                landmarks.extend([
                                math.atan2(results.right_hand_landmarks.landmark[1+finger*4].y - results.right_hand_landmarks.landmark[0].y,
                                           results.right_hand_landmarks.landmark[1+finger*4].x - results.right_hand_landmarks.landmark[0].x),
                                
                                distance.euclidean([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y],
                                                   [results.right_hand_landmarks.landmark[1+finger*4].x, results.right_hand_landmarks.landmark[1+finger*4].y])/normdist])
                
                for joint in [2,3,4]:
                    landmarks.extend([
                                     math.atan2(results.right_hand_landmarks.landmark[joint+finger*4].y - results.right_hand_landmarks.landmark[joint+finger*4-1].y,
                                                results.right_hand_landmarks.landmark[joint+finger*4].x - results.right_hand_landmarks.landmark[joint+finger*4-1].x),
                                    
                                     distance.euclidean([results.right_hand_landmarks.landmark[joint+finger*4-1].x, results.right_hand_landmarks.landmark[joint+finger*4-1].y],
                                                        [results.right_hand_landmarks.landmark[joint+finger*4].x, results.right_hand_landmarks.landmark[joint+finger*4].y])/normdist])
            for fingertips in range(1,5):
                landmarks.extend([
                                math.atan2(results.right_hand_landmarks.landmark[fingertips*4].y - results.right_hand_landmarks.landmark[(fingertips+1)*4].y,
                                           results.right_hand_landmarks.landmark[fingertips*4].x - results.right_hand_landmarks.landmark[(fingertips+1)*4].x),
                                
                                distance.euclidean([results.right_hand_landmarks.landmark[(fingertips+1)*4].x, results.right_hand_landmarks.landmark[(fingertips+1)*4].y],
                                                   [results.right_hand_landmarks.landmark[fingertips*4].x, results.right_hand_landmarks.landmark[fingertips*4].y])/normdist])
        else:  
            landmarks.extend([0] * 48)  # Placeholder for 24 hand landmarks * 2 attributes   
    else:
        landmarks.extend([0] * 134)  # Placeholder for all 134 attributes

    return np.array(landmarks)

def model_predict(value):
    global prob
    pred = model.predict(value, verbose=0)
    max_prob = np.max(pred)
    index_max = np.argmax(pred)
    
    prob = f"{max_prob:.2f}"

    return words_list[index_max]

def interpolate_data(data, frame=75, sec=3):
    # Extract time stamp data
    time_stamp = data[:, -1]
    first_time = time_stamp[0]
    time_stamp = time_stamp - first_time
    
    # Extract data without time stamp
    data = data[:, :-1]
    
    # Create new time stamp data
    new_time_stamp = np.linspace(0, sec, frame)
    
    # Create new data
    new_data = np.zeros((frame, data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:, i] = np.interp(new_time_stamp, time_stamp, data[:, i])
        
    return new_data

def get_last_n_sec_data(data, sec=3):
    last_timestamp = data[-1, -1]  # Get the timestamp of the last row
    for i in range(len(data) - 1, -1, -1):  # Iterate from the last row backward
        if last_timestamp - data[i, -1] >= sec:  # Check if the time difference is 3 seconds
            return data[i:, :]  # Return rows from this index to the last row

    return data  # If no row meets the condition, return all data

# Function to generate video frames
def generate_frames():
    global frame_queue, pred_result, prob, running
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
        return
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            start_time = time.time()
            
            # Process image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Extract landmarks
            landmarks = extract_landmarks(results)
            landmarks_time = time.time()
            landmarks = np.append(landmarks, landmarks_time)
            frame_queue.append(landmarks)
            
            end_time = time.time()

            # Display data
            image = cv2.flip(image, 1)
            
            # Add text overlay
            cv2.putText(image, f"FPS: {1 / (end_time - start_time):.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Prediction: {pred_result}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Probability: {prob}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to jpeg for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def predict_loop():
    global running, pred_result, frame_queue, prob
    
    while running:
        if len(frame_queue) >= 1 and (frame_queue[-1][-1] - frame_queue[0][-1]) >= 3:
            input_data = np.array(frame_queue)
            
            # Get Last 3 seconds data
            input_data = get_last_n_sec_data(input_data)
            test_data = np.empty(shape=(1, 75, 134))
            test_data[0] = interpolate_data(input_data)

            # Cut 10 frames from the beginning if frame queue is more than 100
            if len(frame_queue) > 100:
                frame_queue = frame_queue[10:]
                            
            pred_result = model_predict(test_data)
        time.sleep(0.01)  # Small delay to prevent CPU overuse

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['GET'])
def start_webcam():
    global running, webcam_thread, predict_thread
    
    if not running:
        running = True
        frame_queue.clear()
        predict_thread = threading.Thread(target=predict_loop)
        predict_thread.daemon = True
        predict_thread.start()
        
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already running"})

@app.route('/stop_webcam', methods=['GET'])
def stop_webcam():
    global running
    
    running = False
    return jsonify({"status": "stopped"})

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global pred_result, prob
    
    return jsonify({
        "prediction": pred_result,
        "probability": prob
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True)