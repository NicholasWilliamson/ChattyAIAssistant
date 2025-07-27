#!/usr/bin/env python3
"""
Simple test to debug OpenCV display issues
"""

import cv2
from picamera2 import Picamera2
import numpy as np
import time

def test_opencv_display():
    print("Testing OpenCV display functionality...")
    
    # Test 1: Simple image display
    print("Test 1: Creating and displaying a simple image...")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[100:380, 100:540] = [0, 255, 0]  # Green rectangle
    cv2.putText(test_img, "OpenCV Test Image", (150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    try:
        cv2.imshow('Test Image', test_img)
        print("✅ Image window created successfully")
        key = cv2.waitKey(2000)  # Wait 2 seconds
        cv2.destroyAllWindows()
        print("✅ Image window closed successfully")
    except Exception as e:
        print(f"❌ Image display failed: {e}")
        return False
    
    # Test 2: Live camera feed
    print("\nTest 2: Live camera feed...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print("✅ Camera started successfully")
        
        print("Displaying live feed for 5 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add frame counter
            frame_count += 1
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, "Live Camera Feed", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # This is the critical test - can we display the camera frame?
            cv2.imshow('Live Camera Test', frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"✅ Live camera display successful - {frame_count} frames")
        return True
        
    except Exception as e:
        print(f"❌ Live camera display failed: {e}")
        return False

def test_window_creation():
    print("\nTest 3: Testing window creation and destruction...")
    
    try:
        # Create multiple windows to test
        for i in range(3):
            img = np.zeros((200, 300, 3), dtype=np.uint8)
            img[:] = [i*80, 100, 255-i*80]  # Different colors
            cv2.putText(img, f"Window {i+1}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            window_name = f'Test Window {i+1}'
            cv2.imshow(window_name, img)
            cv2.waitKey(500)  # Show each for 0.5 seconds
        
        cv2.waitKey(1000)  # Show all for 1 second
        cv2.destroyAllWindows()
        print("✅ Multiple window test successful")
        return True
        
    except Exception as e:
        print(f"❌ Window creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 OpenCV Display Diagnostic Test")
    print("=" * 40)
    
    success = True
    success &= test_opencv_display()
    success &= test_window_creation()
    
    if success:
        print("\n✅ All display tests passed!")
        print("The display system should work in your detection application.")
    else:
        print("\n❌ Some display tests failed.")
        print("Check your X11 setup or run directly on the Pi.")
    
    print("\nIf tests pass but main app doesn't show window:")
    print("1. Check for threading issues")
    print("2. Verify cv2.waitKey(1) is being called")
    print("3. Ensure no exception is breaking the display loop")