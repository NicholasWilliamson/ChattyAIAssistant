1. Update and Install Dependencies

First, update your package list and install necessary dependencies.

sudo apt update
sudo apt upgrade -y
sudo apt-get -y install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev

2. Set Up Python Environment

Create a virtual environment and activate it.

python3 -m venv yolov11
source yolov11/bin/activate

3. Install Required Python Packages

Install PyTorch, Ultralytics, OpenCV, and NumPy.

pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy --upgrade

4. Test PyTorch Installation

Verify that PyTorch is installed correctly.

python3 -c "import torch; print(torch.__version__)"

Running YOLO 11

Example Script

Create a script named yolov11-test.py to run YOLO 11.

import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
ret, frame = cap.read()
if not ret:
break

# Run YOLO inference on the frame
results = model(frame)

# Visualize the results on the frame
annotated_frame = results[0].plot()

# Display the resulting frame
cv2.imshow("Camera", annotated_frame)

# Break the loop if 'q' is pressed
if cv2.waitKey(1) == ord("q"):
break

cap.release()
cv2.destroyAllWindows()

Run the script:

python3 yolov11-test.py