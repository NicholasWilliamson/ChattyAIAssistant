from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time

# Load model
model = YOLO("best.pt")

# Set up camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Run YOLO
    results = model(frame)

    # Annotate frame
    annotated = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 + PiCamera2", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.close()