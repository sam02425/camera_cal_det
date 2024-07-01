# Camera calibration settings
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.025  # meters
CALIBRATION_FRAMES = 20

# YOLOv8 settings
YOLO_MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5

# Product database
PRODUCT_DATABASE = {
    0: {'name': 'Product A', 'weight': 100},
    1: {'name': 'Product B', 'weight': 200},
    2: {'name': 'Product C', 'weight': 150},
    # Add more products as needed
}

# Camera settings
CAMERA_1_ID = 0
CAMERA_2_ID = 1

# Weight sensor settings
WEIGHT_SENSOR_PORT = '/dev/ttyUSB0'  # Adjust as needed
WEIGHT_SENSOR_BAUDRATE = 9600