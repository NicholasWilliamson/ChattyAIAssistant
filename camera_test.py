#!/usr/bin/env python3
from flask import Flask, Response
import cv2
from picamera2 import Picamera2
import time

app = Flask(__name__)
picam2 = None

def init_camera():
    global picam2
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)}
        ))
        picam2.start()
        time.sleep(2)
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Camera init failed: {e}")
        return False

@app.route('/')
def index():
    return '<html><body><h1>Camera Test</h1><img src="/video_feed" width="640" height="480"></body></html>'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = picam2.capture_array()
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Frame error: {e}")
            time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if init_camera():
        app.run(host='0.0.0.0', port=5001, debug=False)  # debug=False important!
    else:
        print("Failed to initialize camera")