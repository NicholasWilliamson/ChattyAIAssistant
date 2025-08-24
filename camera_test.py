#!/usr/bin/env python3
"""
Test LibCamera integration with OpenCV using the cam utility approach
"""

import cv2
import subprocess
import os
import sys

def check_camera_with_cam():
    """Use cam utility to check camera availability"""
    print("=== Testing with cam utility ===")
    
    try:
        # List cameras
        result = subprocess.run(['cam', '--list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ cam --list output:")
            print(result.stdout)
            return True
        else:
            print("❌ cam --list failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ cam utility error: {e}")
        return False

def test_libcamerify():
    """Test using libcamerify wrapper"""
    print("\n=== Testing libcamerify wrapper ===")
    
    try:
        # Create a simple test script that uses cv2.VideoCapture
        test_script = '''
import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"SUCCESS: Frame shape {frame.shape}")
        cv2.imwrite("libcamerify_test.jpg", frame)
        sys.exit(0)
    else:
        print("FAILED: No frames")
        sys.exit(1)
else:
    print("FAILED: Cannot open camera")
    sys.exit(1)
cap.release()
'''
        
        # Write test script
        with open('libcamerify_test.py', 'w') as f:
            f.write(test_script)
        
        # Run with libcamerify
        result = subprocess.run(['libcamerify', 'python3', 'libcamerify_test.py'], 
                              capture_output=True, text=True, timeout=15)
        
        print(f"libcamerify output: {result.stdout}")
        if result.stderr:
            print(f"libcamerify errors: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ libcamerify approach works!")
            return True
        else:
            print("❌ libcamerify approach failed")
            return False
            
    except Exception as e:
        print(f"❌ libcamerify test error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('libcamerify_test.py'):
            os.remove('libcamerify_test.py')

def test_media_device_approach():
    """Test using media device information"""
    print("\n=== Testing media device approach ===")
    
    try:
        # Check media devices
        result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
        print("Media devices:")
        print(result.stdout)
        
        # Look for rp1-cfe devices and test them specifically
        lines = result.stdout.split('\n')
        rp1_devices = []
        
        capture_next = False
        for line in lines:
            if 'rp1-cfe' in line:
                capture_next = True
                continue
            if capture_next and line.strip().startswith('/dev/video'):
                device = line.strip()
                rp1_devices.append(device)
            elif line.strip() == '':
                capture_next = False
        
        print(f"Found rp1-cfe devices: {rp1_devices}")
        
        # Test these devices with specific libcamera settings
        for device in rp1_devices[:2]:  # Test first 2 devices
            device_num = device.replace('/dev/video', '')
            print(f"\nTesting {device} (index {device_num})...")
            
            # Try with specific media device configuration
            pipeline = f"v4l2src device={device} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
            
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ SUCCESS with {device}! Frame: {frame.shape}")
                    cv2.imwrite(f'media_test_{device_num}.jpg', frame)
                    cap.release()
                    return device_num, pipeline
                else:
                    print(f"❌ {device} opens but no frames")
            else:
                print(f"❌ Cannot open {device}")
            cap.release()
        
        return None, None
        
    except Exception as e:
        print(f"❌ Media device test error: {e}")
        return None, None

def main():
    print("🔍 Advanced LibCamera Test for Pi Camera Module 2")
    print("=" * 60)
    
    # Test 1: Check camera with cam utility
    cam_works = check_camera_with_cam()
    
    # Test 2: Try libcamerify wrapper
    libcamerify_works = test_libcamerify()
    
    # Test 3: Try media device approach
    device_num, pipeline = test_media_device_approach()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY:")
    
    if cam_works:
        print("✅ Camera hardware detected by libcamera")
    else:
        print("❌ Camera hardware not detected by libcamera")
    
    if libcamerify_works:
        print("✅ libcamerify wrapper approach works!")
        print("\n🎯 SOLUTION FOR CHATTY AI:")
        print("Run your chatty_ai.py with: libcamerify python3 chatty_ai.py")
        return True
    
    if device_num and pipeline:
        print(f"✅ Direct GStreamer approach works with /dev/video{device_num}")
        print(f"\n🎯 SOLUTION FOR CHATTY AI:")
        print(f"Use this pipeline in your code:")
        print(f'cap = cv2.VideoCapture("{pipeline}", cv2.CAP_GSTREAMER)')
        return True
    
    print("\n❌ No working solution found yet.")
    print("Next steps:")
    print("1. Check physical camera connection")
    print("2. Verify camera in raspi-config")
    print("3. Check boot configuration")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready to fix your Chatty AI!")