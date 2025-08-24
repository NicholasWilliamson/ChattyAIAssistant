#!/usr/bin/env python3
"""
Direct OpenCV solution using GStreamer with libcamerasrc
Now that we know the camera works with cam utility
"""

import cv2
import subprocess

def test_libcamera_gstreamer_pipelines():
    """Test GStreamer pipelines with libcamerasrc"""
    
    print("🔍 Testing libcamerasrc GStreamer pipelines")
    print("We know camera 'imx219' exists, so these should work...")
    
    # Since cam utility works with camera=1, try libcamerasrc pipelines
    pipelines = [
        # Basic pipeline
        "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
        
        # With specific camera selection
        "libcamerasrc camera-name=imx219 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1",
        
        # More explicit format conversion
        "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! queue ! appsink drop=1",
        
        # Try different format first
        "libcamerasrc ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink",
        
        # Minimal pipeline
        "libcamerasrc ! videoconvert ! appsink"
    ]
    
    for i, pipeline in enumerate(pipelines):
        print(f"\n--- Testing Pipeline {i+1} ---")
        print(f"Pipeline: {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                print("✅ Pipeline opened!")
                
                # Try to read multiple frames to ensure it's stable
                success_count = 0
                for attempt in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        success_count += 1
                        if attempt == 0:  # Save first successful frame
                            cv2.imwrite(f'gstreamer_test_{i}.jpg', frame)
                            print(f"✅ Frame {attempt+1}: {frame.shape}")
                    else:
                        print(f"❌ Frame {attempt+1}: Failed")
                
                if success_count >= 3:
                    print(f"🎉 WORKING SOLUTION FOUND!")
                    print(f"Success rate: {success_count}/5 frames")
                    cap.release()
                    return pipeline
                else:
                    print(f"⚠️ Unstable: {success_count}/5 frames")
            else:
                print("❌ Cannot open pipeline")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    return None

def test_subprocess_cam_approach():
    """Test using subprocess to call cam utility and process the output"""
    print("\n🔍 Testing subprocess cam approach")
    
    try:
        # Use cam to capture a frame as JPEG
        result = subprocess.run([
            'cam', '--camera=1', '--capture=1', 
            '--file=opencv_test.jpg',
            '--stream', 'role=still,width=640,height=480'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Try to load the image with OpenCV
            import os
            if os.path.exists('opencv_test.jpg'):
                frame = cv2.imread('opencv_test.jpg')
                if frame is not None:
                    print(f"✅ Subprocess approach works! Frame: {frame.shape}")
                    return True
                else:
                    print("❌ Could not load captured image")
            else:
                print("❌ Image file not created")
        else:
            print(f"❌ cam command failed: {result.stderr}")
    
    except Exception as e:
        print(f"❌ Subprocess approach failed: {e}")
    
    return False

def main():
    print("🎯 Final Camera Solution Test")
    print("We know your imx219 camera works with cam utility!")
    print("=" * 60)
    
    # Test GStreamer approach
    working_pipeline = test_libcamera_gstreamer_pipelines()
    
    # Test subprocess approach as backup
    subprocess_works = test_subprocess_cam_approach()
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    
    if working_pipeline:
        print("🎉 SUCCESS! GStreamer pipeline works!")
        print(f"Working pipeline: {working_pipeline}")
        print("\n🔧 TO FIX YOUR CHATTY AI:")
        print("Replace the cv2.VideoCapture line in chatty_ai.py with:")
        print(f'self.cap = cv2.VideoCapture("{working_pipeline}", cv2.CAP_GSTREAMER)')
        return True
    
    elif subprocess_works:
        print("🎉 SUCCESS! Subprocess approach works!")
        print("\n🔧 TO FIX YOUR CHATTY AI:")
        print("We'll need to modify your video feed function to use subprocess calls")
        return True
    
    else:
        print("❌ Still no OpenCV solution found")
        print("But your camera definitely works - we may need a different approach")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to update your Chatty AI code!")