import cv2
import numpy as np
import glob
import pickle
from config import CHESSBOARD_SIZE, SQUARE_SIZE, CALIBRATION_FRAMES
import os

def capture_calibration_frames(camera_id):
    cap = cv2.VideoCapture(camera_id)
    frames = []
    while len(frames) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret:
                frames.append(frame)
                cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
            cv2.imshow(f'Camera {camera_id}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frames

def calibrate_camera(frames):
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE

    objpoints = []
    imgpoints = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def main():
    print("Calibrating Camera 1...")
    frames1 = capture_calibration_frames(0)
    mtx1, dist1 = calibrate_camera(frames1)

    print("Calibrating Camera 2...")
    frames2 = capture_calibration_frames(1)
    mtx2, dist2 = calibrate_camera(frames2)

    calibration_data = {
        'camera1': {'mtx': mtx1, 'dist': dist1},
        'camera2': {'mtx': mtx2, 'dist': dist2}
    }

    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

    print("Calibration complete. Data saved to camera_calibration.pkl")

if __name__ == "__main__":
    main()