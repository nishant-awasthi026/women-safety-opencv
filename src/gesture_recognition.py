import cv2
import mediapipe as mp
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from alert_system import AlertSystem

class GestureRecognition:
    def __init__(self, recipient_phone):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils
        self.smoothing_buffer = []

        self.alert_system = AlertSystem(recipient_phone)

    def recognize_gesture(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        gesture_name = None
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            gesture_detected = self.detect_gestures(results.pose_landmarks)
            if gesture_detected:
                self.update_smoothing_buffer(gesture_detected)
                if self.is_gesture_stable():
                    gesture_name = gesture_detected
                    self.trigger_alert(gesture_name)

        if gesture_name:
            self.display_gesture_name(frame, gesture_name)

        return gesture_name

    def update_smoothing_buffer(self, gesture_name):
        if len(self.smoothing_buffer) >= 5:
            self.smoothing_buffer.pop(0)
        self.smoothing_buffer.append(gesture_name)

    def is_gesture_stable(self):
        return len(self.smoothing_buffer) > 0 and all(g == self.smoothing_buffer[0] for g in self.smoothing_buffer)

    def detect_gestures(self, landmarks):
        if self.detect_sos_signal(landmarks):
            return "SOS"
        elif self.detect_help_signal(landmarks):
            return "Help"
        elif self.detect_call_for_help(landmarks):
            return "Call for Help"
        else:
            return None

    def detect_sos_signal(self, landmarks):
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_hand_up = left_wrist.y < left_shoulder.y
        right_hand_up = right_wrist.y < right_shoulder.y
        left_hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)

        return left_hand_up and right_hand_up and left_hand_open and right_hand_open

    def detect_help_signal(self, landmarks):
        left_hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        return left_hand_open and right_hand_open

    def detect_call_for_help(self, landmarks):
        right_hand_on_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x < landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x and \
                            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y
        return right_hand_on_ear

    def detect_stop_signal(self, landmarks):
        left_hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        return left_hand_open

    def is_hand_open(self, landmarks, wrist_landmark):
        wrist_y = landmarks.landmark[wrist_landmark].y
        shoulder_y = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER if wrist_landmark == self.mp_pose.PoseLandmark.LEFT_WRIST else self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        return wrist_y < shoulder_y

    def display_gesture_name(self, frame, gesture_name):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def trigger_alert(self, gesture_name):
        # Trigger the alert system based on detected gesture
        if gesture_name in ["SOS", "Help", "Call for Help", "Danger"]:
            self.alert_system.send_alert(gesture_name)

if __name__ == "__main__":
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

if __name__ == "__main__":
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
