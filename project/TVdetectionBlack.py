import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from threading import Condition
from ultralytics import YOLO
import socket

# Constants
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480

def load_model():
    # Load the YOLO model
    model = YOLO("weights/yolov8n.pt", "v8")
    # Load class names from coco.txt file
    with open("utils/coco.txt", "r") as file:
        class_list = file.read().splitlines()
    # Find the index of the class "tv" in the class list
    tv_index = class_list.index("tv")

    return tv_index, model

def preprocess_frame(frame):
    # Resizing the frame to lower resolution for processing
    processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    # Adjusting the contrast of the frame
    contrast_value = 50
    processed_frame = cv2.addWeighted(processed_frame, 1 + (contrast_value / 100), np.zeros(processed_frame.shape, processed_frame.dtype), 0, 0)
    return processed_frame

def resize_frame(frame, width, height):

    return cv2.resize(frame, (width, height))

def detect_tv_and_mask(frame, tv_index, model, conf_threshold):
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Predict on image
    detect_params = model.predict(source=[rgb_frame], conf=conf_threshold, save=False)
    # Convert tensor array to numpy
    detect_params_np = detect_params[0].numpy()

    masked_frame = np.zeros_like(frame)  # Initialize masked frame

    if len(detect_params_np) != 0:
        for box in detect_params_np.boxes:
            cls_id = box.cls[0]
            if int(cls_id) == tv_index:  # Check if class ID matches "tv"
                bb = box.xyxy[0]

                # Create a mask for the TV region
                mask = np.zeros_like(frame)
                cv2.rectangle(mask, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 255, 255), -1)
                # Apply the mask to the frame
                masked_frame = cv2.bitwise_and(frame, mask)

                # Draw bounding box
                cv2.rectangle(masked_frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(masked_frame, 'TV', (int(bb[0]), int(bb[1]) - 10), font, 0.9, (0, 255, 0), 2)

    return masked_frame

def zoom_on_tv(frame, bb):
    # Extract coordinates of TV bounding box
    x1, y1, x2, y2 = map(int, bb)

    # Calculate the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate the size of the zoomed-in region
    zoom_width = (x2 - x1) * 2
    zoom_height = (y2 - y1) * 2

    # Calculate the coordinates of the zoomed-in region
    zoom_x1 = max(0, center_x - zoom_width // 2)
    zoom_y1 = max(0, center_y - zoom_height // 2)
    zoom_x2 = min(frame.shape[1], zoom_x1 + zoom_width)
    zoom_y2 = min(frame.shape[0], zoom_y1 + zoom_height)

    # Crop the frame to the zoomed-in region
    zoomed_frame = frame[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

    # Resize the cropped frame to the original size
    zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]))

    return zoomed_frame


if __name__ == "__main__":
    tv_index, yolo_model = load_model()
    picam2 = Picamera2()

    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)

    picam2.start()  # Start camera capture

    try:
        while True:
            # Capture a frame
            frame = picam2.capture_image(name="main")

            # Convert the frame to numpy array
            frame_array = np.array(frame)

            # Preprocess the frame
            processed_frame = preprocess_frame(frame_array)

            # Detect TV and mask the frame
            masked_frame = detect_tv_and_mask(processed_frame, tv_index, yolo_model, 0.4)

            # Resize the frame
            resized_frame = resize_frame(masked_frame, 1920, 1080)

            # Display the frame
            cv2.imshow('Frame', resized_frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        # Stop camera capture
        picam2.stop()

        # Release resources
        cv2.destroyAllWindows()