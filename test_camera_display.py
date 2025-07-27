#!/usr/bin/env python3
"""
Simple camera display test to diagnose display issues
"""

import cv2
from picamera2 import Picamera2
import time
import os

def test_camera_display():
    print("🔧 Testing Camera Display...")
    print("=" * 40)
    
    # Check if we're running in a display environment
    if 'DISPLAY' not in os.environ:
        print("❌ No DISPLAY environment variable found")
        print("Are you running this via SSH? Try:")
        print("ssh -X username@raspberry_pi_ip")
        return False
    else:
        print(f"✅ DISPLAY environment: {os.environ['DISPLAY']}")
    
    # Test OpenCV display capability
    try:
        test_img = cv2.imread('/usr/share/pixmaps/debian-logo.png')
        if test_img is None:
            # Create a simple test image
            import numpy as np
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[25:75, 25:75] = [0, 255, 0]  # Green square
        
        cv2.imshow('Test Window', test_img)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()
        print("✅ OpenCV display test passed")
    except Exception as e:
        print(f"❌ OpenCV display test failed: {e}")
        return False
    
    # Test camera initialization
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print("✅ Camera initialization successful")
        
        # Capture and display a few frames
        print("📹 Testing live camera display (5 seconds)...")
        start_time = time.time()
        
        while time.time() - start_time < 5:
            frame = picam2.capture_array()
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(frame_bgr, f"Camera Test - {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            try:
                cv2.imshow('Camera Test', frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"❌ Display error: {e}")
                print("The camera is working but display has issues")
                break
        
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Camera display test completed")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def show_troubleshooting_tips():
    print("\n🛠️  TROUBLESHOOTING TIPS:")
    print("=" * 40)
    print("If the display doesn't work:")
    print("1. Make sure you're running on the Pi directly (not SSH)")
    print("2. If using SSH, enable X11 forwarding: ssh -X pi@your_pi_ip")
    print("3. Check if you have a desktop environment installed")
    print("4. Try: export DISPLAY=:0.0")
    print("5. Install X11 if missing: sudo apt install xorg")
    print("\nThe detection system works even without display!")
    print("Photos are saved automatically when people are detected.")

if __name__ == "__main__":
    success = test_camera_display()
    if not success:
        show_troubleshooting_tips()