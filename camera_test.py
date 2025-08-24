#!/usr/bin/env python3
"""
Raspberry Pi 5 Camera Test with multiple approaches
"""

import cv2
import subprocess
import sys
import os

def test_libcamera_direct():
    """Test if libcamera can detect cameras"""
    print("=== Testing libcamera directly ===")
    try:
        result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ libcamera detected cameras:")
            print(result.stdout)
            return True
        else:
            print("❌ libcamera failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ libcamera not available: {e}")
        return False

def test_opencv_with_gstreamer():
    """Test OpenCV with GStreamer pipeline for libcamera"""
    print("\n=== Testing OpenCV with GStreamer libcamera ===")
    
    # GStreamer pipeline for libcamera
    pipeline = "libcamerasrc ! video/x-raw,width=640,height=480,format=BGR ! appsink"
    
    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ GStreamer libcamera works! Frame shape: {frame.shape}")
                cap.release()
                return True, pipeline
            else:
                print("⚠️ GStreamer opens but no frames")
        else:
            print("❌ Could not open GStreamer libcamera pipeline")
        cap.release()
    except Exception as e:
        print(f"❌ GStreamer test failed: {e}")
    
    return False, None

def test_opencv_v4l2_extended():
    """Extended V4L2 test with all video devices"""
    print("\n=== Extended V4L2 Test ===")
    
    working_cameras = []
    
    # Test all video devices 0-7 (you have these available)
    for i in range(8):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                # Try to set a common format
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ /dev/video{i} works! Frame shape: {frame.shape}")
                    working_cameras.append(i)
                else:
                    print(f"⚠️ /dev/video{i} opens but no frames")
            cap.release()
        except Exception as e:
            print(f"❌ /dev/video{i} failed: {e}")
    
    return working_cameras

def main():
    print("🔍 Raspberry Pi 5 Camera Diagnostic")
    print("=" * 50)
    
    # Test 1: libcamera directly
    libcamera_works = test_libcamera_direct()
    
    # Test 2: GStreamer with libcamera
    gstreamer_works, gst_pipeline = test_opencv_with_gstreamer()
    
    # Test 3: Extended V4L2 test
    working_v4l2 = test_opencv_v4l2_extended()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"libcamera command: {'✅ Working' if libcamera_works else '❌ Not working'}")
    print(f"GStreamer pipeline: {'✅ Working' if gstreamer_works else '❌ Not working'}")
    print(f"V4L2 cameras: {working_v4l2 if working_v4l2 else '❌ None working'}")
    
    if gstreamer_works:
        print(f"\n🎯 RECOMMENDED SOLUTION:")
        print(f"Use GStreamer pipeline: {gst_pipeline}")
        print(f"This should work with your Raspberry Pi 5 camera!")
    elif working_v4l2:
        print(f"\n🎯 ALTERNATIVE SOLUTION:")
        print(f"Use V4L2 with camera index: {working_v4l2[0]}")
    else:
        print(f"\n❌ No working camera found. Check hardware connections.")

if __name__ == "__main__":
    main()