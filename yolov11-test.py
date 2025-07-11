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