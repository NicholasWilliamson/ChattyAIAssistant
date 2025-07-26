#!/usr/bin/env python3
"""
human_detection.py
Fixed security system with proper person detection and Telegram alerts
"""

import cv2
import face_recognition
import pickle
import os
import subprocess
import time
import numpy as np
import json
import requests
from picamera2 import Picamera2
from datetime import datetime
import logging

# -------------------------------
# Configuration
# -------------------------------
ENCODINGS_FILE = "encodings.pickle"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
RESPONSE_AUDIO = "security_response.wav"
ALERTS_FOLDER = "security_alerts"
CONFIG_FILE = "/home/nickspi5/Chatty_AI/server/config.json"
LOG_FILE = "security_detection.log"

# Security responses
SECURITY_RESPONSES = {
    "Nick": [
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
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
        self.alert_cooldown = 15  # seconds between alerts for same type
        self.setup_person_detection()
        self.create_alerts_folder()
        self.setup_logging()
        self.load_config()
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_count = 0
        self.motion_threshold = 1000  # Minimum contour area for motion detection
        
    def setup_logging(self):
        """Setup logging for security events"""
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """Load Telegram configuration"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            self.telegram_bot_token = config.get('telegram_bot_token')
            self.telegram_chat_id = config.get('telegram_chat_id')
            print("[INFO] Telegram configuration loaded")
        except Exception as e:
            print(f"[WARNING] Could not load Telegram config: {e}")
            self.telegram_bot_token = None
            self.telegram_chat_id = None
            
    def create_alerts_folder(self):
        """Create folder for storing security alert images"""
        if not os.path.exists(ALERTS_FOLDER):
            os.makedirs(ALERTS_FOLDER)
    
    def setup_person_detection(self):
        """Initialize YOLO person detection"""
        try:
            # Check if weights file is actually valid (should be large)
            if os.path.exists("yolov3.weights"):
                file_size = os.path.getsize("yolov3.weights")
                if file_size < 1000000:  # Less than 1MB means it's probably HTML
                    print(f"[WARNING] YOLO weights file too small ({file_size} bytes). Probably HTML page.")
                    raise Exception("Invalid weights file")
                    
            # Load YOLO
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.classes = []
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            self.use_yolo = True
            print("[INFO] YOLO person detection initialized successfully")
            return True
        except Exception as e:
            print(f"[WARNING] YOLO not available ({e}). Using improved backup detection.")
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
    
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        fg_mask = self.background_subtractor.apply(frame)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant motion
        significant_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                significant_contours.append(contour)
        
        return significant_contours, fg_mask
    
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
                if confidence > 0.5 and class_id == 0:  # Increased confidence threshold
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detected_persons = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                detected_persons.append(boxes[i])
        
        return detected_persons
    
    def detect_persons_backup(self, frame):
        """Improved backup person detection using motion + face detection"""
        detected_persons = []
        
        # Only detect if there's significant motion
        motion_contours, fg_mask = self.detect_motion(frame)
        
        if len(motion_contours) == 0:
            return []  # No motion, no person
        
        # If there's motion, check for faces or use body detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # First try face detection (most reliable)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Convert face detections to person boxes (expand the area)
            for (x, y, w, h) in faces:
                # Expand face area to approximate person area
                person_x = max(0, x - w//2)
                person_y = max(0, y - h//4)
                person_w = min(frame.shape[1] - person_x, w * 2)
                person_h = min(frame.shape[0] - person_y, h * 3)
                detected_persons.append([person_x, person_y, person_w, person_h])
        else:
            # If no faces but significant motion, try body detection
            try:
                body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
                bodies = body_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
                detected_persons = [[x, y, w, h] for (x, y, w, h) in bodies if w*h > 2500]  # Filter small detections
            except:
                pass
        
        return detected_persons
    
    def detect_faces_and_analyze(self, frame):
        """Detect and analyze faces for recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if len(face_locations) == 0:
            return [], []
        
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
    
    def send_telegram_alert(self, message, photo_path=None):
        """Send alert to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            print("[WARNING] Telegram not configured")
            return
        
        try:
            # Send text message
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            
            # Send photo if provided
            if photo_path and os.path.exists(photo_path):
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
                with open(photo_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {
                        'chat_id': self.telegram_chat_id,
                        'caption': f"Security Alert Photo - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                    requests.post(url, data=data, files=files)
            
            print("📱 Telegram alert sent successfully")
        except Exception as e:
            print(f"❌ Telegram alert failed: {e}")
    
    def save_security_alert(self, frame, alert_type, person_count):
        """Save frame when security alert is triggered"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ALERT_{alert_type}_{person_count}persons_{timestamp}.jpg"
        filepath = os.path.join(ALERTS_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Security alert saved: {filepath}")
        
        # Log the event
        self.logger.info(f"Security Alert: {alert_type} - {person_count} persons detected - Photo: {filename}")
        
        return filepath
    
    def get_security_response(self, face_names, person_count, has_faces):
        """Determine appropriate security response"""
        import random
        
        # No persons detected
        if person_count == 0:
            return None, None
        
        # Known person recognized
        if "Nick" in face_names:
            return random.choice(SECURITY_RESPONSES["Nick"]), "known_user"
        elif any(name != "Unknown" for name in face_names):
            return random.choice(SECURITY_RESPONSES["Known_Person"]), "known_person"
        
        # Person detected but no face or unknown face
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
        self.frame_count += 1
        
        # Detect persons (works with or without face coverings)
        if self.use_yolo:
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
        status_text = f"Persons: {person_count} | Faces: {len(face_locations)} | Frame: {self.frame_count}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Determine security response
        response_text, alert_type = self.get_security_response(face_names, person_count, has_faces)
        
        if response_text and alert_type and self.should_alert(alert_type):
            print(f"[SECURITY ALERT] {alert_type}: {response_text}")
            photo_path = self.save_security_alert(frame, alert_type, person_count)
            
            # Send Telegram alert for unknown/masked persons
            if alert_type in ["unknown_day", "unknown_night", "masked_person"]:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                telegram_message = f"🚨 <b>SECURITY ALERT</b> 🚨\n\n"
                telegram_message += f"<b>Type:</b> {alert_type.replace('_', ' ').title()}\n"
                telegram_message += f"<b>Time:</b> {timestamp}\n"
                telegram_message += f"<b>Persons Detected:</b> {person_count}\n"
                telegram_message += f"<b>Faces Visible:</b> {len(face_locations)}\n"
                telegram_message += f"<b>Message:</b> {response_text}"
                
                self.send_telegram_alert(telegram_message, photo_path)
            
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
        
        time.sleep(3)  # Allow more time for background subtractor to initialize
        print("🎥 Security camera active! Monitoring for Humans ...")
        
        try:
            while True:
                frame = picam2.capture_array()
                
                # Process every 3rd frame for better performance
                if self.frame_count % 3 == 0:
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

def main():
    print("🛡️ Security Human Detection Application")
    print("=" * 50)
    print("This system detects people even when wearing masks!")
    
    # Initialize and run
    detector = SecurityPersonDetector()
    detector.run_security_monitoring()

if __name__ == "__main__":
    main()