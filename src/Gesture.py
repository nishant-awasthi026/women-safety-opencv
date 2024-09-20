import cv2
import mediapipe as mp
import numpy as np

class Gesture:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils
        self.smoothing_buffer = []

    def recognizegesture(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        gesture_name = None
        threat_level = "None"
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            gesture_detected = self.detect_gestures(results.pose_landmarks)
            if gesture_detected:
                self.update_smoothing_buffer(gesture_detected)
                if self.is_gesture_stable():
                    gesture_name = gesture_detected
                    threat_level = self.get_threat_level(gesture_name)

        if gesture_name:
            self.display_gesture_info(frame, gesture_name, threat_level)

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
    
    def detect_wave_signal(self, landmarks):
        # Assume the hand is waving if the wrist is moving up and down
        wrist_y = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        shoulder_y = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        return wrist_y < shoulder_y

    def detect_thumbs_up_signal(self, landmarks):
        thumb_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_PINKY_TIP]

        thumb_up = thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y
        all_fingers_closed = index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y

        return thumb_up and all_fingers_closed

    def detect_peace_sign_signal(self, landmarks):
        index_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_MIDDLE_FINGER_TIP]
        thumb_tip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_THUMB_TIP]
        hand_open = self.is_hand_open(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)

        return index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and hand_open

    def is_hand_open(self, landmarks, wrist_landmark):
        wrist_y = landmarks.landmark[wrist_landmark].y
        shoulder_y = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER if wrist_landmark == self.mp_pose.PoseLandmark.LEFT_WRIST else self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        return wrist_y < shoulder_y

    def get_threat_level(self, gesture_name):
        # Simple threat level assignment
        threat_levels = {
            "SOS": "High",
            "Help": "Medium",
            "Call for Help": "High",
            "Wave": "Low",
            "Thumbs Up": "Low",
            "Peace Sign": "Low",
        }
        return threat_levels.get(gesture_name, "None")

    def display_gesture_info(self, frame, gesture_name, threat_level):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Threat Level: {threat_level}", (10, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

if __name__ == "__main__":
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

if __name__ == "__main__":
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
