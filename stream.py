from flask import Flask, Response, render_template
import cv2
import os
import datetime
from threading import Thread
from queue import Queue
import numpy as np
import time

app = Flask(__name__)
fps = 0

rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'

def create_directory():
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(today_date, exist_ok=True)
    return today_date
        
def load_face_detection_model():
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def detect_faces(net, frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    face_locations = []
    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            extra_size = 50
            x -= extra_size
            y -= extra_size
            x1 += extra_size
            y1 += extra_size
            face_locations.append((x, y, x1 - x, y1 - y))
    
    return face_locations

def process_frame(net, frame_queue, output_queue):
    global fps
    last_save_time = datetime.datetime.now()
    prev_frame_time = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Start time for FPS calculation
        start_time = time.time()

        face_locations = detect_faces(net, frame)

        today_directory = create_directory()
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

        for idx, (x, y, w, h) in enumerate(face_locations):
            top, right, bottom, left = y, x + w, y + h, x
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            face_image = frame[max(0, top):bottom, max(0, left):right]
            filename = f"{current_time}_{idx}.jpg"
            filepath = os.path.join(today_directory, filename)
            if (datetime.datetime.now() - last_save_time).total_seconds() >= 60:
                cv2.imwrite(filepath, face_image)
                last_save_time = datetime.datetime.now()

        # End time for FPS calculation
        end_time = time.time()

        # Calculate and update FPS
        elapsed_time = end_time - prev_frame_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        prev_frame_time = end_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        output_queue.put(frame)
        frame_queue.task_done()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    video_capture = cv2.VideoCapture(rtsp_url)
    if not video_capture.isOpened():
        raise RuntimeError(f"Error opening video source {rtsp_url}")

    net = load_face_detection_model()

    frame_queue = Queue(maxsize=5)
    output_queue = Queue(maxsize=5)

    thread = Thread(target=process_frame, args=(net, frame_queue, output_queue))
    thread.start()

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to read frame. Reconnecting to the RTSP stream...")
            video_capture.release()
            video_capture = cv2.VideoCapture(rtsp_url)
            continue

        if not frame_queue.full():
            frame_queue.put(frame)
        
        if not output_queue.empty():
            processed_frame = output_queue.get()
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    frame_queue.put(None)
    thread.join()
    video_capture.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
