#!/usr/bin/env python3
"""
Comprehensive Raspberry Pi Camera Module test
Tries multiple approaches to access Pi Camera Module 2
"""

import cv2
import subprocess
import sys
import os

def test_command_available(cmd):
    """Test if a command is available"""
    try:
        result = subprocess.run(['which', cmd], capture_output=True)
        return result.returncode == 0
    except:
        return False

def test_direct_v4l2_devices():
    """Test direct access to V4L2 devices with specific formats"""
    print("=== Testing Direct V4L2 with Pi Camera Formats ===")
    
    working_configs = []
    
    # Common Pi Camera formats and resolutions
    formats = [
        {'width': 640, 'height': 480, 'fourcc': 'MJPG'},
        {'width': 640, 'height': 480, 'fourcc': 'YUYV'},
        {'width': 1920, 'height': 1080, 'fourcc': 'MJPG'},
        {'width': 320, 'height': 240, 'fourcc': 'YUYV'},
    ]
    
    for device_idx in range(8):  # Test /dev/video0 to /dev/video7
        device_path = f"/dev/video{device_idx}"
        if not os.path.exists(device_path):
            continue
            
        print(f"\nTesting {device_path}:")
        
        for fmt in formats:
            try:
                cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    continue
                
                # Set format
                if fmt['fourcc'] == 'MJPG':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                elif fmt['fourcc'] == 'YUYV':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, fmt['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fmt['height'])
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"  ✅ SUCCESS: {device_path} with {fmt['fourcc']} {actual_width}x{actual_height}")
                    print(f"     Frame shape: {frame.shape}")
                    
                    working_configs.append({
                        'device': device_idx,
                        'format': fmt['fourcc'],
                        'width': actual_width,
                        'height': actual_height,
                        'backend': cv2.CAP_V4L2
                    })
                    
                    # Save test image
                    cv2.imwrite(f'test_camera_{device_idx}_{fmt["fourcc"]}.jpg', frame)
                    
                else:
                    print(f"  ❌ {device_path} {fmt['fourcc']} {fmt['width']}x{fmt['height']}: No frames")
                
                cap.release()
                
            except Exception as e:
                print(f"  ❌ {device_path} {fmt['fourcc']}: {e}")
    
    return working_configs

def test_gstreamer_alternatives():
    """Test alternative GStreamer approaches"""
    print("\n=== Testing Alternative GStreamer Pipelines ===")
    
    # More GStreamer pipeline variations
    pipelines = [
        # Try with v4l2src instead of libcamerasrc
        "v4l2src device=/dev/video0 ! video/x-raw,format=BGR,width=640,height=480 ! appsink",
        "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=BGR ! appsink",
        
        # Try different video devices
        "v4l2src device=/dev/video2 ! video/x-raw,format=BGR,width=640,height=480 ! appsink",
        "v4l2src device=/dev/video2 ! videoconvert ! video/x-raw,format=BGR ! appsink",
        
        # Try with MJPEG
        "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink",
        
        # Legacy approach
        "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink",
    ]
    
    for i, pipeline in enumerate(pipelines):
        print(f"\nTesting pipeline {i+1}: {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✅ SUCCESS! Frame: {frame.shape}")
                    cv2.imwrite(f'gst_test_{i}.jpg', frame)
                    cap.release()
                    return pipeline
                else:
                    print(f"  ❌ Opens but no frames")
            else:
                print(f"  ❌ Cannot open")
            
            cap.release()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return None

def main():
    print("🔍 Comprehensive Pi Camera Module Test")
    print("=" * 60)
    
    # Check available commands
    print("\n=== Command Availability ===")
    commands = ['libcamera-hello', 'raspistill', 'cam', 'v4l2-ctl']
    for cmd in commands:
        available = test_command_available(cmd)
        print(f"{cmd}: {'✅ Available' if available else '❌ Not found'}")
    
    # Test direct V4L2 access
    working_v4l2 = test_direct_v4l2_devices()
    
    # Test GStreamer alternatives
    working_gst = test_gstreamer_alternatives()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    
    if working_v4l2:
        print(f"\n✅ WORKING V4L2 CONFIGURATIONS:")
        for config in working_v4l2:
            print(f"  Device: /dev/video{config['device']}")
            print(f"  Format: {config['format']}")
            print(f"  Size: {config['width']}x{config['height']}")
            print(f"  Backend: V4L2")
            print()
        
        print("🎯 RECOMMENDED SOLUTION FOR CHATTY AI:")
        best_config = working_v4l2[0]  # Use first working config
        print(f"cap = cv2.VideoCapture({best_config['device']}, cv2.CAP_V4L2)")
        if best_config['format'] == 'MJPG':
            print(f"cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))")
        print(f"cap.set(cv2.CAP_PROP_FRAME_WIDTH, {best_config['width']})")
        print(f"cap.set(cv2.CAP_PROP_FRAME_HEIGHT, {best_config['height']})")
    
    if working_gst:
        print(f"\n✅ WORKING GSTREAMER PIPELINE:")
        print(f"cap = cv2.VideoCapture('{working_gst}', cv2.CAP_GSTREAMER)")
    
    if not working_v4l2 and not working_gst:
        print("\n❌ NO WORKING CONFIGURATION FOUND")
        print("Possible issues:")
        print("1. Camera ribbon cable not properly connected")
        print("2. Camera not enabled in boot config")
        print("3. Wrong camera module or hardware issue")

if __name__ == "__main__":
    main()