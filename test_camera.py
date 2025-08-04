#!/usr/bin/env python3
"""
test_camera.py - Simple camera test for Chatty AI
This script tests basic camera functionality
"""

import sys
import time
import cv2
import numpy as np
from datetime import datetime

def test_camera():
    """Test camera functionality"""
    print("🎥 Testing Raspberry Pi Camera...")
    print("=" * 50)
    
    try:
        # Try to import Picamera2
        print("📦 Importing Picamera2...")
        from picamera2 import Picamera2
        print("✅ Picamera2 imported successfully")
        
        # Create camera instance
        print("🔧 Creating camera instance...")
        picam2 = Picamera2()
        print("✅ Camera instance created")
        
        # Configure camera
        print("⚙️ Configuring camera...")
        config = picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)}
        )
        picam2.configure(config)
        print("✅ Camera configured")
        
        # Start camera
        print("🚀 Starting camera...")
        picam2.start()
        print("✅ Camera started")
        
        # Wait for camera to warm up
        print("⏱️ Waiting for camera to warm up...")
        time.sleep(3)
        
        # Test capture
        print("📸 Testing capture...")
        for i in range(5):
            try:
                frame = picam2.capture_array()
                print(f"✅ Capture {i+1}: Shape={frame.shape}, dtype={frame.dtype}")
                
                # Convert and save test image
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:  # RGBA
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:  # RGB
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    # Add timestamp overlay
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame_bgr, f"Test {i+1} - {timestamp}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save test image
                    filename = f"camera_test_{i+1}.jpg"
                    cv2.imwrite(filename, frame_bgr)
                    print(f"💾 Saved test image: {filename}")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Capture {i+1} failed: {e}")
        
        # Stop camera
        print("🛑 Stopping camera...")
        picam2.stop()
        print("✅ Camera stopped successfully")
        
        print("\n🎉 Camera test completed successfully!")
        print("✅ Your camera is working properly")
        print("📁 Check the saved test images in the current directory")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Picamera2: {e}")
        print("💡 Make sure you have picamera2 installed: pip install picamera2")
        return False
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\n👁️ Testing OpenCV...")
    print("=" * 50)
    
    try:
        print(f"📦 OpenCV version: {cv2.__version__}")
        
        # Test creating an image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "OpenCV Test", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Test encoding
        ret, buffer = cv2.imencode('.jpg', test_img)
        if ret:
            print("✅ OpenCV image encoding test passed")
            cv2.imwrite("opencv_test.jpg", test_img)
            print("💾 Saved OpenCV test image: opencv_test.jpg")
        else:
            print("❌ OpenCV image encoding failed")
            return False
        
        print("✅ OpenCV is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_audio_permissions():
    """Test audio permissions"""
    print("\n🔊 Testing audio setup...")
    print("=" * 50)
    
    try:
        import sounddevice as sd
        print("✅ sounddevice imported successfully")
        
        # Query audio devices
        devices = sd.query_devices()
        print("🎤 Available audio devices:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Chatty AI Component Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    if not test_opencv():
        all_passed = False
    
    if not test_camera():
        all_passed = False
    
    if not test_audio_permissions():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your system should work with the web interface")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please fix the issues above before running the web interface")
    
    print("=" * 50)