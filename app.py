from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from src.Geofencing import check_geofence
from src.video_analytics import VideoAnalytics
import sys
import base64
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.gesture_recognition import GestureRecognition
from src.Gesture import Gesture
from src.Speech_analyse import analyze_speech
import numpy as np
import cv2

app = Flask(__name__)

ALERT_RECIPIENT_PHONE = "+916307257097"

if not os.path.exists('temp'):
    os.makedirs('temp') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/geofencing', methods=['GET', 'POST'])
def geofencing():
    if request.method == 'POST':
        try:
            latitude = float(request.form.get('latitude'))
            longitude = float(request.form.get('longitude'))
            result = check_geofence(latitude, longitude)
            return render_template('geofencing.html', result=result)
        except ValueError as e:
            return render_template('geofencing.html', error=f"Invalid input: {e}")
    return render_template('geofencing.html')

@app.route('/speech_analysis', methods=['GET', 'POST'])
def speech_analysis():
    results = None
    if request.method == 'POST':
        try:
            duration = int(request.form.get('duration', 5))
            results = analyze_speech(duration)
            print("Results:", results)  
        except Exception as e:
            print("Error:", e)  
            return str(e)
    return render_template('speech_analysis.html', results=results)

@app.route('/gesture_recognition_page')
def gesture_recognition_page():
    return render_template('gesture_recognition.html')

@app.route('/recognize_gesture', methods=['POST'])
def recognize_gesture():
    
    recipient_phone = "+916307257097"
    cap = cv2.VideoCapture(0)
    gesture_recognition = GestureRecognition(recipient_phone)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gesture_recognition.recognize_gesture(frame)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    gesture_name = gesture_recognition.recognize_gesture(frame)

    return jsonify({'gesture': gesture_name})

@app.route('/video_analytics', methods=['GET', 'POST'])
def video_analytics():
    log_entries = [] 
    if request.method == 'POST' and 'video_file' in request.files:
        video_file = request.files['video_file']
        video_path = os.path.join('temp', video_file.filename)
        
        try:
            video_file.save(video_path)
            log_entries.append(f"Video file saved to {video_path}")
            
            video_analytics = VideoAnalytics(video_path)
            video_analytics.process_video()
            
            result = "Video processed successfully!"
            log_entries.append(result)
        except Exception as e:
            error_message = f"Error: {e}"
            log_entries.append(error_message)
            result = error_message
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
                log_entries.append(f"Video file {video_path} deleted.")
        
        return render_template('video_analytics.html', result=result, log_entries=log_entries)
    return render_template('video_analytics.html', log_entries=log_entries)

@app.route('/threat_level_detection', methods=['GET'])
def threat_level_detection():
    return render_template('threat_level_detection.html')

@app.route('/stream_feed')
def stream_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        gesturerecognition = Gesture()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gesturerecognition.recognizegesture(frame)

            cv2.imshow('Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)