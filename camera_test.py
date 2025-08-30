#!/usr/bin/env python3
"""
Camera test script to verify Picamera2 is working
"""

import cv2
from picamera2 import Picamera2
import time

print("Testing Picamera2...")

try:
    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}
    ))
    picam2.start()
    
    print("Camera started successfully!")
    print("Capturing test frame...")
    
    time.sleep(2)
    
    # Capture a frame
    frame = picam2.capture_array()
    print(f"Frame captured: {frame.shape}")
    
    # Convert and save test image
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("/tmp/camera_test.jpg", frame)
    print("Test image saved to /tmp/camera_test.jpg")
    
    picam2.stop()
    picam2.close()
    print("Camera test completed successfully!")
    
except Exception as e:
    print(f"Camera test failed: {e}")
    import traceback
    traceback.print_exc()