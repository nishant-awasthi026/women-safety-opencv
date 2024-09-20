import numpy as np

class AnomalyDetection:
    def detect_anomaly(self, data):
        # Simple threshold-based anomaly detection
        threshold = 0.5
        if np.mean(data) > threshold:
            print("Anomaly detected!")
            return True
        return False

def detect_anomalies(data):
    detector = AnomalyDetection()
    return detector.detect_anomaly(data)