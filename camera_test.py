#!/usr/bin/env python3
"""
Test Pi Camera Module 2 with OpenCV and GStreamer
"""

import cv2
import numpy as np

def test_pi_camera():
    """Test Pi Camera Module with different GStreamer pipelines"""
    
    # Different GStreamer pipelines to try for Pi Camera Module 2
    pipelines = [
        # Modern libcamera pipeline
        "libcamerasrc ! video/x-raw,width=640,height=480,format=BGR ! appsink drop=1",
        
        # Alternative with framerate
        "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1,format=BGR ! appsink drop=1",
        
        # With specific camera (if multiple cameras)
        "libcamerasrc camera-name=0 ! video/x-raw,width=640,height=480,format=BGR ! appsink drop=1",
        
        # Legacy pipeline (might work)
        "libcamerasrc ! videoconvert ! video/x-raw,format=BGR ! appsink",
    ]
    
    for i, pipeline in enumerate(pipelines):
        print(f"\nTesting pipeline {i+1}: {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                print("✅ Pipeline opened successfully")
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    print(f"✅ SUCCESS! Frame captured: {frame.shape}")
                    print(f"✅ This pipeline works: {pipeline}")
                    
                    # Save test image
                    cv2.imwrite('pi_camera_test.jpg', frame)
                    print("✅ Test image saved as 'pi_camera_test.jpg'")
                    
                    cap.release()
                    return pipeline
                else:
                    print("❌ Pipeline opened but no frames received")
            else:
                print("❌ Could not open pipeline")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
    
    print("\n❌ No working pipeline found")
    return None

def test_working_pipeline(pipeline):
    """Test the working pipeline more thoroughly"""
    print(f"\n🎥 Testing working pipeline: {pipeline}")
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ Could not reopen pipeline")
        return False
    
    frame_count = 0
    max_frames = 10
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"✅ Frame {frame_count}: {frame.shape}")
        else:
            print(f"❌ Failed to read frame {frame_count + 1}")
            break
    
    cap.release()
    return frame_count > 0

if __name__ == "__main__":
    print("🔍 Testing Pi Camera Module 2 with OpenCV")
    print("=" * 50)
    
    working_pipeline = test_pi_camera()
    
    if working_pipeline:
        print(f"\n🎯 WORKING SOLUTION FOUND!")
        print(f"Pipeline: {working_pipeline}")
        
        # Test it more thoroughly
        if test_working_pipeline(working_pipeline):
            print(f"\n✅ Ready to update chatty_ai.py with this pipeline!")
        else:
            print(f"\n⚠️ Pipeline works but may be unstable")
    else:
        print(f"\n❌ No working camera configuration found")
        print("Check that:")
        print("1. Camera ribbon cable is properly connected")
        print("2. Camera is enabled in raspi-config")
        print("3. libcamera-tools is installed")