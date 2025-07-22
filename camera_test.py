python3 -c "
from picamera2 import Picamera2
import time

print('Initializing camera...')
picam2 = Picamera2()
print('Camera hardware info:')
print(f'Camera controls: {picam2.camera_controls}')
picam2.start()
time.sleep(2)
print('Camera started successfully!')
picam2.stop()
print('Test completed - camera is working!')
"