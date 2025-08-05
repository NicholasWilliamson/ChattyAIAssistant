#!/usr/bin/env python3
"""
fix_camera.py - Fix camera resource conflicts
This script helps resolve camera "device or resource busy" errors
"""

import subprocess
import time
import os
import signal
import psutil

def kill_camera_processes():
    """Kill any processes using the camera"""
    print("🔍 Searching for camera processes...")
    
    killed_processes = []
    
    # Look for processes using camera-related keywords
    camera_keywords = ['libcamera', 'picamera2', 'python3.*camera', 'gstreamer']
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            name = proc.info['name'] or ''
            
            # Check if process is camera-related
            is_camera_process = False
            for keyword in camera_keywords:
                if keyword in name.lower() or keyword in cmdline.lower():
                    is_camera_process = True
                    break
            
            # Also check for specific camera device usage
            if '/dev/video' in cmdline or '/dev/media' in cmdline:
                is_camera_process = True
            
            if is_camera_process:
                print(f"📹 Found camera process: PID {proc.info['pid']} - {name}")
                print(f"   Command: {cmdline[:100]}...")
                
                try:
                    # Try graceful termination first
                    proc.terminate()
                    proc.wait(timeout=3)
                    killed_processes.append(f"PID {proc.info['pid']} ({name})")
                    print(f"✅ Gracefully terminated PID {proc.info['pid']}")
                except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                    try:
                        # Force kill if graceful doesn't work
                        proc.kill()
                        killed_processes.append(f"PID {proc.info['pid']} ({name}) - FORCED")
                        print(f"💀 Force killed PID {proc.info['pid']}")
                    except psutil.NoSuchProcess:
                        print(f"⚠️  Process PID {proc.info['pid']} already terminated")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return killed_processes

def check_camera_devices():
    """Check camera device status"""
    print("\n🔍 Checking camera devices...")
    
    # Check for video devices
    video_devices = []
    for i in range(10):  # Check /dev/video0 to /dev/video9
        device = f"/dev/video{i}"
        if os.path.exists(device):
            video_devices.append(device)
    
    if video_devices:
        print(f"📹 Found video devices: {', '.join(video_devices)}")
    else:
        print("❌ No video devices found")
    
    # Check for media devices (libcamera)
    media_devices = []
    for i in range(10):  # Check /dev/media0 to /dev/media9
        device = f"/dev/media{i}"
        if os.path.exists(device):
            media_devices.append(device)
    
    if media_devices:
        print(f"📹 Found media devices: {', '.join(media_devices)}")
    else:
        print("❌ No media devices found")
    
    return video_devices, media_devices

def reset_camera_modules():
    """Reset camera kernel modules"""
    print("\n🔄 Resetting camera modules...")
    
    try:
        # List of camera-related modules to reset
        modules = ['bcm2835_v4l2', 'bcm2835_mmal_vchiq', 'v4l2_common']
        
        for module in modules:
            try:
                # Check if module is loaded
                result = subprocess.run(['lsmod'], capture_output=True, text=True)
                if module in result.stdout:
                    print(f"🔧 Removing module: {module}")
                    subprocess.run(['sudo', 'rmmod', module], check=True, capture_output=True)
                    time.sleep(1)
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not remove module {module} (may not be loaded)")
        
        # Wait a moment
        time.sleep(2)
        
        # Reload modules
        for module in reversed(modules):
            try:
                print(f"🔧 Loading module: {module}")
                subprocess.run(['sudo', 'modprobe', module], check=True, capture_output=True)
                time.sleep(1)
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not load module {module}")
        
        print("✅ Camera modules reset complete")
        
    except Exception as e:
        print(f"❌ Error resetting modules: {e}")

def test_camera_access():
    """Test if camera is accessible"""
    print("\n🧪 Testing camera access...")
    
    try:
        # Try to import and initialize picamera2
        from picamera2 import Picamera2
        print("✅ Picamera2 import successful")
        
        # Try to create camera instance
        picam2 = Picamera2()
        print("✅ Camera instance created")
        
        # Try to configure
        config = picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)}
        )
        picam2.configure(config)
        print("✅ Camera configured")
        
        # Try to start
        picam2.start()
        print("✅ Camera started")
        
        # Wait a moment
        time.sleep(2)
        
        # Try to capture
        frame = picam2.capture_array()
        print(f"✅ Test capture successful - Frame shape: {frame.shape}")
        
        # Stop camera
        picam2.stop()
        picam2.close()
        print("✅ Camera stopped and closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def fix_permissions():
    """Fix camera device permissions"""
    print("\n🔧 Fixing camera permissions...")
    
    try:
        # Get current user
        user = os.getenv('USER')
        if not user:
            print("❌ Could not determine current user")
            return False
        
        # Check if user is in video group
        result = subprocess.run(['groups', user], capture_output=True, text=True)
        if 'video' not in result.stdout:
            print(f"➕ Adding {user} to video group...")
            subprocess.run(['sudo', 'usermod', '-a', '-G', 'video', user], check=True)
            print("✅ User added to video group")
            print("⚠️  You may need to log out and back in for group changes to take effect")
        else:
            print("✅ User already in video group")
        
        # Set permissions on device files
        devices_to_fix = ['/dev/video*', '/dev/media*', '/dev/vchiq']
        
        for device_pattern in devices_to_fix:
            try:
                # Use shell expansion to handle wildcards
                subprocess.run(f'sudo chmod 666 {device_pattern}', 
                             shell=True, check=True, capture_output=True)
                print(f"✅ Fixed permissions for {device_pattern}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not fix permissions for {device_pattern}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing permissions: {e}")
        return False

def main():
    """Main camera fix routine"""
    print("🔧 Chatty AI Camera Fix Utility")
    print("=" * 50)
    
    # Step 1: Kill camera processes
    killed = kill_camera_processes()
    
    if killed:
        print(f"\n💀 Killed {len(killed)} camera processes:")
        for proc in killed:
            print(f"   - {proc}")
        print("\n⏱️  Waiting 3 seconds for cleanup...")
        time.sleep(3)
    else:
        print("\n✅ No camera processes found to kill")
    
    # Step 2: Check devices
    video_devs, media_devs = check_camera_devices()
    
    # Step 3: Fix permissions
    fix_permissions()
    
    # Step 4: Reset modules (optional, commented out as it may require reboot)
    # reset_camera_modules()
    
    # Step 5: Test camera
    print("\n" + "=" * 50)
    if test_camera_access():
        print("\n🎉 SUCCESS! Camera is now accessible")
        print("✅ You can now run your Chatty AI application")
    else:
        print("\n❌ Camera still not accessible")
        print("🔧 Additional troubleshooting may be needed:")
        print("   1. Try rebooting the system")
        print("   2. Check camera cable connections")
        print("   3. Verify camera is enabled in raspi-config")
        print("   4. Check if camera works with: libcamera-hello")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()