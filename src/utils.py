import cv2

def save_frame(frame, output_path):
    cv2.imwrite(output_path, frame)
