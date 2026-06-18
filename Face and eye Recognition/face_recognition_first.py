import cv2 as cv
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Corrected URL and File Path targeting Google's official server
MODEL_PATH = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    print("Downloading official MediaPipe Face Detection model file...")
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

def run_face_detection_logic():
    """
    Handles the webcam processing loop using the modern MediaPipe Tasks API.
    """
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file missing. Please download it or check your internet connection.")

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found or already in use by another application.")

    print("\nInitializing modern MediaPipe Face Detector. Starting video stream...")

    # Configure MediaPipe Tasks for sequential video processing
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    # Open the modern detector instance
    with vision.FaceDetector.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error: Unable to capture video frame.")
                break

            # MediaPipe Tasks requires RGB color space
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Video mode requires a unique timestamp for every frame in milliseconds
            timestamp_ms = int(time.time() * 1000)
            
            # Run object tracking inference
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            # Process the modern results structural objects
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    
                    # Modern MediaPipe tasks give us RAW PIXEL values directly!
                    x_pixel = bbox.origin_x
                    y_pixel = bbox.origin_y
                    w_pixel = bbox.width
                    h_pixel = bbox.height
                    
                    center_x = x_pixel + (w_pixel // 2)
                    center_y = y_pixel + (h_pixel // 2)

                    # Draw green bounding box around face
                    cv.rectangle(frame, (x_pixel, y_pixel), (x_pixel + w_pixel, y_pixel + h_pixel), (0, 255, 0), 2)
                    # Draw blue circle at center point
                    cv.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            cv.imshow("MediaPipe Tasks Face Detection - Press ENTER to close", frame)

            key = cv.waitKey(1) & 0xFF
            if key == 13: # Enter key
                print("Video session closed manually.")
                break

    cap.release()
    cv.destroyAllWindows()


def main_menu():
    """
    Command-line interface to coordinate project initialization.
    """
    while True:
        print("\n" + "="*30)
        print("   BIOMETRIC PROJECT - MENU   ")
        print("="*30)
        print("[1] : Run Face Detection (MediaPipe Tasks API)")
        print("[2] : Exit Program")
        
        user_choice = input("Your choice: ").strip()

        if user_choice == "1":
            try:
                run_face_detection_logic()
            except RuntimeError as hardware_error:
                print(f"\n[HARDWARE ERROR]: {hardware_error}")
            except Exception as unexpected_error:
                print(f"\n[UNKNOWN ERROR]: {unexpected_error}")
        
        elif user_choice == "2":
            print("Exiting application safely. Goodbye!")
            break
        else:
            print("Invalid response. Please select option 1 or 2.")

if __name__ == "__main__":
    main_menu()
