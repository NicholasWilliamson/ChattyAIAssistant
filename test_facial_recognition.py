
#!/usr/bin/env python3
"""
Enhanced Facial Recognition Test with Piper TTS, Telegram Alerts & Security Logging
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import subprocess
import pygame
import json
import requests
from datetime import datetime
import time
import logging

class FacialRecognitionSystem:
    def __init__(self):
        # Define directories
        self.base_dir = "/home/nickspi5/Chatty_AI"
        self.security_photos_dir = os.path.join(self.base_dir, "security_photos")
        self.security_logs_dir = os.path.join(self.base_dir, "security_logs")
        
        # Create directories if they don't exist
        os.makedirs(self.security_photos_dir, exist_ok=True)
        os.makedirs(self.security_logs_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize facial recognition
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        # Telegram configuration
        self.telegram_config = None
        
        # Initialize pygame for audio
        pygame.mixer.init()
        
        # Load known faces and telegram config
        self.load_known_faces()
        self.load_telegram_config()
    
    def setup_logging(self):
        """Setup logging for human detections"""
        log_file = os.path.join(self.security_logs_dir, "human_detections.log")
        
        # Create logger
        self.security_logger = logging.getLogger('security_logger')
        self.security_logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.security_logger.handlers:
            self.security_logger.addHandler(file_handler)
    
    def load_known_faces(self):
        """Load known face encodings from pickle file"""
        encodings_file = "known_faces.pkl"
        
        if os.path.exists(encodings_file):
            print("[INFO] Loading facial encodings...")
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_face_encodings)} face encodings")
        else:
            print("[ERROR] No known faces file found! Please run face enrollment first.")
            exit(1)
    
    def load_telegram_config(self):
        """Load Telegram configuration"""
        config_file = "telegram_config.json"
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.telegram_config = json.load(f)
                
                # Test connection
                if self.test_telegram_connection():
                    print("[INFO] Telegram configuration loaded successfully")
                else:
                    print("[WARNING] Telegram configuration loaded but connection failed")
                    self.telegram_config = None
            except Exception as e:
                print(f"[ERROR] Failed to load Telegram config: {e}")
                self.telegram_config = None
        else:
            print("[WARNING] Telegram config file not found. Creating template...")
            self.create_telegram_config_template()
    
    def create_telegram_config_template(self):
        """Create a template telegram config file"""
        template = {
            "bot_token": "YOUR_BOT_TOKEN_HERE",
            "chat_id": "YOUR_CHAT_ID_HERE",
            "bot_name": "Your Bot Name"
        }
        
        with open("telegram_config.json", 'w') as f:
            json.dump(template, f, indent=4)
        
        print("[INFO] Created telegram_config.json template")
        print("[INFO] Please edit this file with your Telegram bot credentials")
    
    def test_telegram_connection(self):
        """Test Telegram bot connection"""
        if not self.telegram_config:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    bot_name = bot_info['result'].get('first_name', 'Unknown')
                    print(f"[INFO] Telegram bot '{bot_name}' connected successfully")
                    return True
            
            return False
        except Exception as e:
            print(f"[ERROR] Telegram connection test failed: {e}")
            return False
    
    def send_telegram_alert(self, message, photo_path=None):
        """Send alert to Telegram"""
        if not self.telegram_config:
            print("[WARNING] Telegram not configured - skipping alert")
            return False
        
        try:
            if photo_path and os.path.exists(photo_path):
                # Send photo with caption
                url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendPhoto"
                
                with open(photo_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {
                        'chat_id': self.telegram_config['chat_id'],
                        'caption': message
                    }
                    
                    response = requests.post(url, files=files, data=data, timeout=10)
            else:
                # Send text message only
                url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
                data = {
                    'chat_id': self.telegram_config['chat_id'],
                    'text': message
                }
                
                response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                print(f"[TELEGRAM] ✅ Alert sent: {message}")
                return True
            else:
                print(f"[TELEGRAM] ❌ Failed to send alert: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[TELEGRAM] ❌ Error sending alert: {e}")
            return False
    
    def generate_tts_response(self, text, person_name="Unknown"):
        """Generate TTS response using Piper"""
        try:
            print(f"🔊 Speaking: {text}")
            
            # Generate TTS
            piper_command = [
                "piper",
                "--model", "en_US-lessac-medium.onnx",
                "--output_file", "recognition_response.wav"
            ]
            
            process = subprocess.Popen(
                piper_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0:
                # Play the generated audio
                pygame.mixer.music.load("recognition_response.wav")
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                return True
            else:
                print(f"[ERROR] Piper TTS failed: {stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            return False
    
    def save_detection_photo(self, frame, person_name, confidence):
        """Save detection photo to security_photos directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name.lower()}_{timestamp}.jpg"
        photo_path = os.path.join(self.security_photos_dir, filename)
        
        # Save the photo
        cv2.imwrite(photo_path, frame)
        print(f"📸 Detection photo saved: {photo_path}")
        
        return photo_path
    
    def log_detection(self, person_name, confidence, photo_path):
        """Log detection to security log file"""
        log_message = f"DETECTION | Person: {person_name} | Confidence: {confidence:.2f} | Photo: {os.path.basename(photo_path)}"
        self.security_logger.info(log_message)
    
    def process_frame(self, frame):
        """Process a single frame for face recognition"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        
        self.face_names = []
        confidences = []
        
        for face_encoding in self.face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > 0.4:  # Confidence threshold
                    name = self.known_face_names[best_match_index]
                    print(f"[DEBUG] Recognized {name} with confidence: {confidence:.2f}")
                else:
                    print(f"[DEBUG] Unknown person detected with confidence: {confidence:.2f}")
            
            self.face_names.append(name)
            confidences.append(confidence)
        
        return confidences
    
    def draw_face_boxes(self, frame, confidences):
        """Draw face detection boxes on frame"""
        # Scale back up face locations
        for (top, right, bottom, left), name, confidence in zip(self.face_locations, self.face_names, confidences):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else f"Unknown ({confidence:.2f})"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def handle_recognition(self, frame, person_name, confidence):
        """Handle a successful recognition"""
        # Save detection photo
        photo_path = self.save_detection_photo(frame, person_name, confidence)
        
        # Log the detection
        self.log_detection(person_name, confidence, photo_path)
        
        # Generate personalized response
        if person_name == "Nick":
            response_text = "Hello Nick, my master. It is wonderful to see you again. Thank you so much for creating me. How can I assist you my friend?"
        else:
            response_text = f"Hello! I d not recognize you. It is nice to meet you. How can I assist you?"
        
        print(f"[RECOGNIZED] {person_name} - {response_text}")
        
        # Generate TTS response
        self.generate_tts_response(response_text, person_name)
        
        # Send Telegram alert
        if person_name != "Unknown":
            alert_message = f"✅ KNOWN - {person_name}"
        else:
            alert_message = f"⚠️ UNKNOWN PERSON DETECTED"
        
        self.send_telegram_alert(alert_message, photo_path)
    
    def run_recognition_test(self):
        """Run the facial recognition test"""
        print("🤖 Facial Recognition Test with Piper TTS & Telegram Alerts")
        print("=" * 60)
        print("[INFO] Starting facial recognition test with Telegram alerts...")
        print("Press 'q' to quit, 's' to save current frame, 't' to test Telegram")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
        
        print("🎥 Camera ready! Look at the camera for recognition.")
        
        last_recognition_time = 0
        recognition_cooldown = 10  # seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to capture frame")
                    break
                
                # Process frame
                confidences = self.process_frame(frame)
                
                # Draw face boxes
                frame = self.draw_face_boxes(frame, confidences)
                
                # Check for recognition
                current_time = time.time()
                if (current_time - last_recognition_time) > recognition_cooldown:
                    for name, confidence in zip(self.face_names, confidences):
                        if name != "Unknown" and confidence > 0.4:
                            self.handle_recognition(frame, name, confidence)
                            last_recognition_time = current_time
                            break
                        elif name == "Unknown" and confidence > 0.3:  # Lower threshold for unknown
                            self.handle_recognition(frame, name, confidence)
                            last_recognition_time = current_time
                            break
                
                # Display frame
                cv2.imshow('Facial Recognition Security System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_save_{timestamp}.jpg"
                    save_path = os.path.join(self.security_photos_dir, filename)
                    cv2.imwrite(save_path, frame)
                    print(f"📸 Frame saved manually: {save_path}")
                elif key == ord('t'):
                    # Test Telegram
                    test_message = "🧪 Test message from Facial Recognition System"
                    self.send_telegram_alert(test_message)
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print("[INFO] Facial recognition test completed")

def main():
    """Main function"""
    try:
        # Create and run facial recognition system
        recognition_system = FacialRecognitionSystem()
        recognition_system.run_recognition_test()
        
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())