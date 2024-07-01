import cv2
import torch
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD

class ProductDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def detect_products(self, frame):
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            if confidence > CONFIDENCE_THRESHOLD:
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': int(class_id),
                    'class_name': self.model.names[int(class_id)]
                })
        return detections

def draw_detections(frame, detections):
    for det in detections:
        bbox = det['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    detector = ProductDetector()
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        detections1 = detector.detect_products(frame1)
        detections2 = detector.detect_products(frame2)

        frame1 = draw_detections(frame1, detections1)
        frame2 = draw_detections(frame2, detections2)

        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()