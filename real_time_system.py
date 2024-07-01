import cv2
import numpy as np
import pickle
from object_detection import ProductDetector, draw_detections
from weight_prediction import WeightPredictor
from fusion_model import FusionModel
from config import PRODUCT_DATABASE, CAMERA_1_ID, CAMERA_2_ID
import time
import threading
import queue
import sqlite3
from flask import Flask, render_template, jsonify
from utils import logger, resize_frame, apply_clahe

app = Flask(__name__)

class CameraThread(threading.Thread):
    def __init__(self, camera_id, frame_queue):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                frame = resize_frame(frame, 50)  # Resize for performance
                frame = apply_clahe(frame)  # Enhance contrast
                self.frame_queue.put((self.camera_id, frame))
            else:
                logger.error(f"Failed to read frame from camera {self.camera_id}")
        cap.release()

    def stop(self):
        self.stopped = True

def load_camera_calibration():
    try:
        with open('camera_calibration.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading camera calibration: {e}")
        return None

def undistort_frame(frame, camera_matrix, dist_coeffs):
    try:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]
    except Exception as e:
        logger.error(f"Error undistorting frame: {e}")
        return frame

def save_to_database(timestamp, detections, weight):
    try:
        conn = sqlite3.connect('product_detection.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                     (timestamp TEXT, product_id INTEGER, product_name TEXT, confidence REAL, weight REAL)''')
        for det in detections:
            c.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?)",
                      (timestamp, det['product_id'], det['product_name'], det['confidence'], weight))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving to database: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    conn = sqlite3.connect('product_detection.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10")
    data = c.fetchall()
    conn.close()
    return jsonify(data)

def main():
    calibration_data = load_camera_calibration()
    product_detector = ProductDetector()
    weight_predictor = WeightPredictor()
    fusion_model = FusionModel(PRODUCT_DATABASE)

    frame_queue = queue.Queue(maxsize=10)
    camera_threads = [
        CameraThread(CAMERA_1_ID, frame_queue),
        CameraThread(CAMERA_2_ID, frame_queue)
    ]

    for thread in camera_threads:
        thread.start()

    try:
        while True:
            frames = {}
            for _ in range(2):  # Get frames from both cameras
                camera_id, frame = frame_queue.get()
                frames[camera_id] = frame

            all_detections = []
            for camera_id, frame in frames.items():
                if calibration_data:
                    frame = undistort_frame(frame, calibration_data[f'camera{camera_id+1}']['mtx'],
                                            calibration_data[f'camera{camera_id+1}']['dist'])
                detections = product_detector.detect_products(frame)
                all_detections.extend(detections)
                frames[camera_id] = draw_detections(frame, detections)

            weight_prediction = weight_predictor.predict()
            fused_results = fusion_model.fuse_detections(all_detections, weight_prediction)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            save_to_database(timestamp, fused_results, weight_prediction)

            for camera_id, frame in frames.items():
                cv2.imshow(f'Camera {camera_id}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        for thread in camera_threads:
            thread.stop()
        for thread in camera_threads:
            thread.join()
        cv2.destroyAllWindows()
        weight_predictor.close()

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()
    main()