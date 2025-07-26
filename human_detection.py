#!/usr/bin/env python3
"""
security_detection.py
Security system with person detection that works regardless of face coverings
Combines facial recognition with general person detection for comprehensive security
"""

import cv2
import face_recognition
import pickle
import os
import subprocess
import time
import numpy as np
from picamera2 import Picamera2
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
ENCODINGS_FILE = "encodings.pickle"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
RESPONSE_AUDIO = "security_response.wav"
ALERTS_FOLDER = "security_alerts"

# Security responses
SECURITY_RESPONSES = {
    "Nick": [
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?"
    ],
    "Known_Person": [
        "Hello! Welcome back. Security system recognizes you.",
        "Hi there! Access granted. Have a good day."
    ],
    "Unknown_Person_Day": [
        "Hello! I see someone is here. Please identify yourself.",
        "Hello stranger! Unknown person detected. May I help you?",
        "Hello stranger! I don't recognize you. Are you expected by the home owner?"
    ],
    "Unknown_Person_Night": [
        "Security Alert! Unknown person detected during night hours!",
        "Warning! Unauthorized person detected. Authorities may be contacted.",
        "Alert! Unknown individual detected outside day light hours!"
    ],
    "Masked_Person": [
        "Alert! Person detected with face covering. Please identify yourself immediately!",
        "Security Warning! Individual with concealed face detected!",
        "Attention! Person with mask detected. State your identity and purpose immediately!"
    ]
}

class SecurityPersonDetector:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        self.last_alert_time = {}
        self.alert_cooldown = 10  # seconds between alerts for same type
        self.setup_person_detection()
        self.create_alerts_folder()
        
    def create_alerts_folder(self):
        """Create folder for storing security alert images"""
        if not os.path.exists(ALERTS_FOLDER):
            os.makedirs(ALERTS_FOLDER)
    
    def setup_person_detection(self):
        """Initialize YOLO person detection"""
        try:
            # Load YOLO
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.classes = []
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            print("[INFO] YOLO person detection initialized")
            return True
        except Exception as e:
            print(f"[WARNING] YOLO not available ({e}). Using backup person detection.")
            self.use_yolo = False
            return False
    
    def load_encodings(self):
        """Load facial recognition encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_encodings)} face encodings")
            return True
        except FileNotFoundError:
            print("[WARNING] No face encodings found. Running in person detection only mode.")
            return False
    
    def detect_persons_yolo(self, frame):
        """Detect persons using YOLO"""
        height, width, channels = frame.shape
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Information to show on screen
        class_ids = []
        confidences = []
        boxes = []
        
        # For each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only detect persons (class_id = 0 in COCO dataset)
                if confidence > 0.3 and class_id == 0:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        detected_persons = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                detected_persons.append(boxes[i])
        
        return detected_persons
    
    def detect_persons_backup(self, frame):
        """Backup person detection using Haar cascades"""
        # This is a simpler fallback method
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load cascade classifier for full body detection
        try:
            body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
            return [[x, y, w, h] for (x, y, w, h) in bodies]
        except:
            return []
    
    def detect_faces_and_analyze(self, frame):
        """Detect and analyze faces for recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if len(face_locations) == 0:
            return [], "no_face"
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        recognized_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
            
            recognized_names.append(name)
        
        return face_locations, recognized_names
    
    def is_night_time(self):
        """Check if it's night time (6 PM to 6 AM)"""
        current_hour = datetime.now().hour
        return current_hour >= 18 or current_hour <= 6
    
    def should_alert(self, alert_type):
        """Check if enough time has passed since last alert of this type"""
        current_time = time.time()
        if alert_type not in self.last_alert_time:
            self.last_alert_time[alert_type] = current_time
            return True
        
        time_since_last = current_time - self.last_alert_time[alert_type]
        if time_since_last >= self.alert_cooldown:
            self.last_alert_time[alert_type] = current_time
            return True
        
        return False
    
    def speak_text(self, text):
        """Convert text to speech using Piper"""
        print(f"🔊 Security Alert: {text}")
        try:
            command = [
                PIPER_EXECUTABLE,
                "--model", VOICE_PATH,
                "--config", CONFIG_PATH,
                "--output_file", RESPONSE_AUDIO
            ]
            subprocess.run(command, input=text.encode("utf-8"), check=True)
            subprocess.run(["aplay", RESPONSE_AUDIO], check=True)
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    def save_security_alert(self, frame, alert_type, person_count):
        """Save frame when security alert is triggered"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ALERT_{alert_type}_{person_count}persons_{timestamp}.jpg"
        filepath = os.path.join(ALERTS_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Security alert saved: {filepath}")
        return filepath
    
    def get_security_response(self, face_names, person_count, has_faces):
        """Determine appropriate security response"""
        import random
        
        # Known person recognized
        if "Nick" in face_names:
            return random.choice(SECURITY_RESPONSES["Nick"]), "known_user"
        elif any(name != "Unknown" for name in face_names):
            return random.choice(SECURITY_RESPONSES["Known_Person"]), "known_person"
        
        # Person detected but no face or unknown face
        if person_count > 0:
            if not has_faces:
                # Person detected but no face visible (likely masked)
                return random.choice(SECURITY_RESPONSES["Masked_Person"]), "masked_person"
            elif "Unknown" in face_names:
                # Unknown person with visible face
                if self.is_night_time():
                    return random.choice(SECURITY_RESPONSES["Unknown_Person_Night"]), "unknown_night"
                else:
                    return random.choice(SECURITY_RESPONSES["Unknown_Person_Day"]), "unknown_day"
        
        return None, None
    
    def process_security_frame(self, frame):
        """Process frame for comprehensive security analysis"""
        # Detect persons (works with or without face coverings)
        if hasattr(self, 'net'):
            person_boxes = self.detect_persons_yolo(frame)
        else:
            person_boxes = self.detect_persons_backup(frame)
        
        # Detect and recognize faces
        face_locations, face_names = self.detect_faces_and_analyze(frame)
        
        person_count = len(person_boxes)
        has_faces = len(face_locations) > 0
        
        # Draw person detection boxes (red for persons)
        for (x, y, w, h) in person_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "PERSON DETECTED", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw face detection boxes (green for recognized, yellow for unknown)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 255, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Security status overlay
        status_text = f"Persons: {person_count} | Faces: {len(face_locations)}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Determine security response
        response_text, alert_type = self.get_security_response(face_names, person_count, has_faces)
        
        if response_text and alert_type and self.should_alert(alert_type):
            print(f"[SECURITY ALERT] {alert_type}: {response_text}")
            self.save_security_alert(frame, alert_type, person_count)
            self.speak_text(response_text)
        
        return frame
    
    def run_security_monitoring(self):
        """Run the security monitoring system"""
        print("[INFO] Starting Security Person Detection System...")
        print("This system detects people regardless of face coverings!")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Initialize camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
        
        time.sleep(2)
        print("🎥 Security camera active! Monitoring for persons...")
        
        frame_count = 0
        
        try:
            while True:
                frame = picam2.capture_array()
                frame_count += 1
                
                # Process every 2nd frame for better performance
                if frame_count % 2 == 0:
                    processed_frame = self.process_security_frame(frame.copy())
                    cv2.imshow('Security Person Detection', processed_frame)
                else:
                    cv2.imshow('Security Person Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"security_snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Snapshot saved as {filename}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Security monitoring stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            picam2.stop()
            print("[INFO] Security monitoring completed")

def download_yolo_files():
    """Download YOLO files if not present"""
    files_needed = [
        ("yolov3.weights", "https://pjreddie.com/media/files/yolov3.weights"),
        ("yolov3.cfg", "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"),
        ("coco.names", "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
    ]
    
    missing_files = []
    for filename, url in files_needed:
        if not os.path.exists(filename):
            missing_files.append((filename, url))
    
    if missing_files:
        print("[INFO] YOLO files missing. Please download them manually:")
        for filename, url in missing_files:
            print(f"  wget {url}")
        print("\nOr the system will use backup person detection.")
        return False
    return True

def main():
    print("🛡️ Security Human Detection Application")
    print("=" * 50)
    print("This system detects people even when wearing masks!")
    
    # Check for YOLO files
    download_yolo_files()
    
    # Initialize and run
    detector = SecurityPersonDetector()
    detector.run_security_monitoring()

if __name__ == "__main__":
    main()