# Dual Camera Product Detection System

This project implements a real-time product detection system using two cameras, YOLOv8 for object detection, and a weight sensing system. The system performs camera calibration, product detection, and fusion of visual and weight data for accurate product identification.

## Hardware Requirements

- 2x USB cameras (or Raspberry Pi cameras)
- Computer with CUDA-capable GPU (for YOLOv8)
- HX711 load cell amplifier and compatible load cell(s)

## Software Requirements

- Python 3.8+
- CUDA and cuDNN (for GPU acceleration)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/dual_camera_product_detection.git
   cd dual_camera_product_detection
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 weights file and place it in the project directory.

## Usage

1. Calibrate the cameras:
   ```
   python camera_calibration.py
   ```
   Follow the on-screen instructions to capture calibration images.

2. Train or fine-tune the YOLOv8 model on your product dataset.

3. Update the `config.py` file with your specific settings.

4. Run the real-time detection system:
   ```
   python real_time_system.py
   ```

## File Descriptions

- `camera_calibration.py`: Script for calibrating the dual camera setup
- `object_detection.py`: Implements YOLOv8 for product detection
- `weight_prediction.py`: Contains the weight prediction model
- `fusion_model.py`: Combines results from object detection and weight prediction
- `real_time_system.py`: Runs the real-time product detection system
- `utils.py`: Utility functions for various operations
- `config.py`: Configuration settings for the system

## Camera Setup

1. Position the two cameras to cover the area where products will be placed.
2. Ensure the cameras have a clear, unobstructed view of the product area.
3. Adjust lighting to minimize glare and shadows.

## Troubleshooting

- If calibration fails, ensure the chessboard pattern is fully visible in both camera views.
- For YOLOv8 issues, check CUDA and cuDNN installations.
- Verify all connections if weight readings are inconsistent.

For further assistance, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Requirements

- Ensure proper lighting conditions for accurate product detection
- Calibrate the weight sensor regularly for accurate weight predictions
- Use a stable mounting solution for cameras to minimize vibrations
- For production use, implement proper error handling, logging, and data persistence
- Consider adding a local display for real-time visualization without requiring a separate monitor

## Connection Instructions

1. Connect the two USB cameras to the computer's USB ports.
2. Connect the HX711 weight sensor to the specified serial port (default: /dev/ttyUSB0).
3. Ensure all connections are secure and the weight sensor is properly configured.
4. If using a different port for the weight sensor, update the WEIGHT_SENSOR_PORT in config.py.

## Calibration Tips

- Use a high-quality chessboard pattern for camera calibration
- Ensure the chessboard is fully visible in both camera views during calibration
- Perform camera calibration in the same lighting conditions as the intended use
- Calibrate the weight sensor using known weights that cover the expected range of product weights

## Maintenance

- Regularly check and clean camera lenses for optimal performance
- Periodically recalibrate both cameras and the weight sensor
- Update the YOLOv8 model and product database as new products are added to the system
- Monitor system logs and performance metrics to identify and address any issues promptly