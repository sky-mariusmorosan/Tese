import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Constants
PROCESSING_WIDTH = 1080
PROCESSING_HEIGHT = 720

def load_model():
    # Load the YOLO model
    model = YOLO("weights/yolov8n.pt", "v8")
    # Load class names from coco.txt file
    with open("utils/coco.txt", "r") as file:
        class_list = file.read().splitlines()
    # Find the index of the class "tv" in the class list
    tv_index = class_list.index("tv")
    return tv_index, model

def preprocess_frame(frame, contrast_value):
    # Resizing the frame to lower resolution for processing
    processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    # Adjusting the contrast of the frame
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

    if len(detect_params_np) != 0:
        for box in detect_params_np.boxes:
            cls_id = box.cls[0]
            if int(cls_id) == tv_index:  # Check if class ID matches "tv"
                bb = box.xyxy[0]
                # Draw bounding box
                cv2.rectangle(rgb_frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)
                # Display class name and confidence
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(rgb_frame, 'TV', (int(bb[0]), int(bb[1]) - 10), font, 0.9, (0, 255, 0), 2)

    return rgb_frame

if __name__ == "__main__":
    tv_index, yolo_model = load_model()
    picam2 = Picamera2()
    camera_started = False

    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)

    # Trackbar callback function (does nothing)
    def trackbar_callback(value):
        pass

    # Create a window
    cv2.namedWindow('TV', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TV', 1080, 720)

    # Create trackbars
    cv2.createTrackbar('Contrast', 'TV', 50, 100, trackbar_callback)
    cv2.createTrackbar('Threshold', 'TV', 25, 100, trackbar_callback)

    # Define button parameters
    button_text = 'Start'
    button_state = False
    button_color = (0, 0, 255)  # Red

    while True:
        # Display button
        button_img = np.zeros((50, 100, 3), np.uint8)
        cv2.rectangle(button_img, (0, 0), (100, 50), button_color, -1)
        cv2.putText(button_img, button_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Button', button_img)

        # Check for button click
        if cv2.waitKey(1) & 0xFF == ord('b'):
            button_state = not button_state
            if button_state:
                button_text = 'Stop'
                button_color = (0, 255, 0)  # Green
                # Start the camera
                picam2.start()
                camera_started = True
            else:
                button_text = 'Start'
                button_color = (255, 0, 0)  # Red
                # Stop the camera
                picam2.stop()
                camera_started = False

        # Capture a frame if the camera is started
        if camera_started:
            frame = picam2.capture_image(name="main")
            frame_array = np.array(frame)

            # Preprocess the frame
            processed_frame = preprocess_frame(frame_array, cv2.getTrackbarPos('Contrast', 'TV'))

            # Detect TV and mask the frame
            masked_frame = detect_tv_and_mask(processed_frame, tv_index, yolo_model, cv2.getTrackbarPos('Threshold', 'TV') / 100)

            # Resize the frame
            resized_frame = resize_frame(masked_frame, 1920, 1080)

            # Display the frame
            cv2.imshow('TV', resized_frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    picam2.stop()
    cv2.destroyAllWindows()
