import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger('product_detection_system')
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler('product_detection_system.log', maxBytes=10000000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def resize_frame(frame, scale_percent):
    try:
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Error resizing frame: {e}")
        return frame

def apply_clahe(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.error(f"Error applying CLAHE: {e}")
        return frame

def draw_text(img, text, position, font_scale=0.7, color=(255,255,255), thickness=2):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    except Exception as e:
        logger.error(f"Error drawing text: {e}")

def calculate_iou(box1, box2):
    try:
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    except Exception as e:
        logger.error(f"Error calculating IoU: {e}")
        return 0