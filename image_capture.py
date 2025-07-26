import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time
import face_recognition

# Change this to the name of the person you're photographing
PERSON_NAME = "Spiderman"  

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos_with_validation(name):
    folder = create_folder(name)
    
    # Initialize the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    # Allow camera to warm up
    time.sleep(2)

    photo_count = 0
    valid_photos = 0
    
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")
    print("For masks: Take photos from multiple angles, distances, and lighting conditions")
    print("Recommended: 30-50 photos for good mask recognition")
    
    while True:
        # Capture frame from Pi Camera
        frame = picam2.capture_array()
        
        # Convert to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in real-time
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # Draw rectangles around detected faces
        display_frame = frame.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Detected", (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show face count
        cv2.putText(display_frame, f"Faces: {len(face_locations)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Photos: {photo_count} (Valid: {valid_photos})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Capture - Face Detection Active', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            if len(face_locations) > 0:
                photo_count += 1
                valid_photos += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, frame)
                print(f"✅ Valid photo {photo_count} saved: {filepath} ({len(face_locations)} face(s) detected)")
            else:
                photo_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}_noface.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, frame)
                print(f"⚠️  Photo {photo_count} saved but NO FACE detected: {filepath}")
        
        elif key == ord('q'):  # Q key
            break
    
    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Photo capture completed. {photo_count} total photos saved for {name}.")
    print(f"Valid photos with faces detected: {valid_photos}")
    if valid_photos < 20:
        print("⚠️  Warning: Consider taking more photos for better recognition accuracy")

if __name__ == "__main__":
    capture_photos_with_validation(PERSON_NAME)