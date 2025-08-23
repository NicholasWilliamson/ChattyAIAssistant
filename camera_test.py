#!/usr/bin/env python3
"""
Camera diagnostic script for Chatty AI
"""

import cv2
import os
import sys
import subprocess

print("=== Camera Diagnostic Test ===")

# Check video devices
print("\n1. Checking video devices:")
video_devices = []
for i in range(10):
    device_path = f"/dev/video{i}"
    if os.path.exists(device_path):
        video_devices.append(device_path)
        print(f"   Found: {device_path}")

if not video_devices:
    print("   No video devices found!")
else:
    print(f"   Total video devices: {len(video_devices)}")

# Check user permissions
print("\n2. Checking user permissions:")
try:
    result = subprocess.run(['groups'], capture_output=True, text=True)
    groups = result.stdout.strip()
    print(f"   Current user groups: {groups}")
    
    if 'video' in groups:
        print("   ✅ User is in video group")
    else:
        print("   ❌ User is NOT in video group - this may be the issue!")
        print("   Run: sudo usermod -a -G video $USER")
        print("   Then logout and login again")
except Exception as e:
    print(f"   Error checking groups: {e}")

# Test camera access with different backends
print("\n3. Testing camera access:")

backends = [
    (cv2.CAP_ANY, "CAP_ANY (auto)"),
    (cv2.CAP_V4L2, "CAP_V4L2 (Video4Linux)"),
    (cv2.CAP_GSTREAMER, "CAP_GSTREAMER"),
]

for backend_id, backend_name in backends:
    print(f"\n   Testing with {backend_name}:")
    
    for camera_index in range(4):  # Test indices 0-3
        try:
            print(f"     Camera index {camera_index}: ", end="")
            cap = cv2.VideoCapture(camera_index, backend_id)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"✅ SUCCESS - Resolution: {width}x{height}")
                    
                    # Get camera properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"                      FPS: {fps}")
                    
                    cap.release()
                    
                    # Save this working configuration
                    print(f"\n   🎉 WORKING CAMERA FOUND!")
                    print(f"   Backend: {backend_name}")
                    print(f"   Index: {camera_index}")
                    print(f"   Resolution: {width}x{height}")
                    print(f"   FPS: {fps}")
                    
                    # Test if we can capture multiple frames
                    print("\n   Testing continuous capture...")
                    cap = cv2.VideoCapture(camera_index, backend_id)
                    success_count = 0
                    for i in range(5):
                        ret, frame = cap.read()
                        if ret:
                            success_count += 1
                    cap.release()
                    
                    print(f"   Captured {success_count}/5 frames successfully")
                    
                    if success_count >= 4:
                        print("   ✅ Camera is working reliably!")
                        print(f"\n   UPDATE YOUR CHATTY_AI.PY:")
                        print(f"   Change camera initialization to:")
                        print(f"   self.cap = cv2.VideoCapture({camera_index}, {backend_id})")
                        sys.exit(0)
                    else:
                        print("   ⚠️ Camera is unreliable")
                        
                else:
                    print("❌ Can't read frames")
                    cap.release()
            else:
                print("❌ Can't open")
                cap.release()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                cap.release()
            except:
                pass

print("\n❌ No working camera found!")
print("\nTroubleshooting steps:")
print("1. Make sure camera is connected")
print("2. Check if user is in video group: sudo usermod -a -G video $USER")
print("3. For Pi camera: sudo raspi-config -> Interface Options -> Camera -> Enable")
print("4. Try different USB ports")
print("5. Check camera permissions: ls -la /dev/video*")