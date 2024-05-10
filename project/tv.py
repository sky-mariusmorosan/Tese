from flask import Flask, render_template, Response
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Constants
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 420

app = Flask(__name__)
camera = None

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
    return processed_frame

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

def generate_frames():
    global camera
    tv_index, yolo_model = load_model()
    picam2 = Picamera2()

    while True:
        if camera is None:
            camera = picam2.start()
        frame = picam2.capture_image(name="main")
        frame_array = np.array(frame)

        # Preprocess the frame
        processed_frame = preprocess_frame(frame_array)

        # Detect TV and mask the frame
        masked_frame = detect_tv_and_mask(processed_frame, tv_index, yolo_model, conf_threshold=0.5)

        # Resize the frame
        resized_frame = cv2.resize(masked_frame, (2028, 1520))

        # Encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode('.jpg', resized_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
