import subprocess
import time

def fix_camera_state():
    """Force reset camera state"""
    try:
        # Kill any existing camera processes
        subprocess.run(['sudo', 'pkill', '-f', 'python.*chatty'], capture_output=True)
        time.sleep(2)
        
        # Reset camera hardware
        subprocess.run(['sudo', 'modprobe', '-r', 'bcm2835_v4l2'], capture_output=True)
        time.sleep(1)
        subprocess.run(['sudo', 'modprobe', 'bcm2835_v4l2'], capture_output=True)
        time.sleep(2)
        
        print("Camera state reset complete")
    except Exception as e:
        print(f"Camera reset failed: {e}")

if __name__ == "__main__":
    fix_camera_state()