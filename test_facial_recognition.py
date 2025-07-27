#!/usr/bin/env python3
"""
test_facial_recognition.py with Telegram Integration
Test facial recognition with personalized TTS responses using Piper and Telegram alerts
"""

import cv2
import face_recognition
import pickle
import os
import subprocess
import time
import json
import requests
import logging
from picamera2 import Picamera2
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
ENCODINGS_FILE = "encodings.pickle"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
RESPONSE_AUDIO = "recognition_response.wav"
TELEGRAM_CONFIG_FILE = "telegram_config.json"

# Security directories
SECURITY_PHOTOS_DIR = "/home/nickspi5/Chatty_AI/security_photos"
SECURITY_LOGS_DIR = "/home/nickspi5/Chatty_AI/security_logs"

# Personalized responses for different people
PERSON_RESPONSES = {
    "Nick": [
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?"
    ],
    "Unknown": [
        "Hello there! I don't recognize you yet.",
        "Hi! You're new to me. Nice to meet you!",
        "Hello stranger! Would you like to be registered?",
        "Hi there! I haven't seen you before."
    ]
}

# Telegram messages for different people
TELEGRAM_MESSAGES = {
    "known": {
        "Nick": "🏠 **AUTHORIZED ACCESS** 🏠\n\n👤 **Person:** Nick (Master)\n⏰ **Time:** {timestamp}\n✅ **Status:** Authorized User\n🎯 **Confidence:** {confidence:.1%}\n\n💬 **Response:** Greeted with personalized welcome message",
        "Spiderman": "🕷️ **SUPER HERO DETECTED** 🕷️\n\n👤 **Person:** Spider-Man\n⏰ **Time:** {timestamp}\n✅ **Status:** Favorite Super Hero!\n🎯 **Confidence:** {confidence:.1%}\n\n💬 **Response:** Special hero greeting delivered"
    },
    "unknown": "🚨 **UNKNOWN PERSON DETECTED** 🚨\n\n👤 **Person:** Unknown Individual\n⏰ **Time:** {timestamp}\n⚠️ **Status:** Unregistered Person\n🔍 **Action:** Photo captured for review\n\n💡 **Note:** Consider registering this person if they should have access"
}

class FacialRecognitionTester:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        self.last_recognition_time = {}
        self.recognition_cooldown = 5  # seconds between recognitions for same person
        
        # Telegram configuration
        self.telegram_token = None
        self.telegram_chat_id = None
        self.load_telegram_config()
        
        # Setup security directories and logging
        self.setup_security_directories()
        self.setup_security_logging()
        
    def setup_security_directories(self):
        """Create security directories if they don't exist"""
        os.makedirs(SECURITY_PHOTOS_DIR, exist_ok=True)
        os.makedirs(SECURITY_LOGS_DIR, exist_ok=True)
    
    def setup_security_logging(self):
        """Setup logging for human detections"""
        log_file = os.path.join(SECURITY_LOGS_DIR, "human_detections.log")
        
        # Create logger
        self.security_logger = logging.getLogger('security_logger')
        self.security_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.security_logger.handlers[:]:
            self.security_logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.security_logger.addHandler(file_handler)
    
    def log_detection(self, person_name, confidence, photo_filename):
        """Log detection to security log file"""
        log_message = f"DETECTION | Person: {person_name} | Confidence: {confidence:.2f} | Photo: {photo_filename}"
        self.security_logger.info(log_message)
        
    def load_encodings(self):
        """Load the facial recognition encodings"""
        try:
            print("[INFO] Loading facial encodings...")
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_encodings)} face encodings")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Encodings file '{ENCODINGS_FILE}' not found!")
            print("Please run model_training.py first to create the encodings.")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load encodings: {e}")
            return False
    
    def load_telegram_config(self):
        """Load Telegram configuration"""
        try:
            with open(TELEGRAM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.telegram_token = config.get('bot_token')
                self.telegram_chat_id = config.get('chat_id')
            
            if self.telegram_token and self.telegram_chat_id:
                print("[INFO] Telegram configuration loaded successfully")
                self.test_telegram_connection()
            else:
                print("[WARNING] Incomplete Telegram configuration")
        except FileNotFoundError:
            print("[WARNING] Telegram config file not found. Creating template...")
            self.create_telegram_config_template()
        except Exception as e:
            print(f"[ERROR] Failed to load Telegram config: {e}")
    
    def create_telegram_config_template(self):
        """Create a template telegram config file"""
        template_config = {
            "bot_token": "YOUR_BOT_TOKEN_HERE",
            "chat_id": "YOUR_CHAT_ID_HERE",
            "instructions": {
                "step1": "Create a bot by messaging @BotFather on Telegram",
                "step2": "Get your bot token from BotFather",
                "step3": "Start a chat with your bot and send any message",
                "step4": "Visit https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates to get your chat_id",
                "step5": "Replace YOUR_BOT_TOKEN_HERE and YOUR_CHAT_ID_HERE with actual values"
            }
        }
        
        try:
            with open(TELEGRAM_CONFIG_FILE, 'w') as f:
                json.dump(template_config, f, indent=4)
            print(f"[INFO] Created {TELEGRAM_CONFIG_FILE} template")
            print("[INFO] Please edit this file with your Telegram bot credentials")
        except Exception as e:
            print(f"[ERROR] Failed to create config template: {e}")
    
    def test_telegram_connection(self):
        """Test the Telegram bot connection"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info['result']['first_name']
                print(f"[INFO] Telegram bot '{bot_name}' connected successfully")
                return True
            else:
                print(f"[ERROR] Telegram bot connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Telegram connection test failed: {e}")
            return False
    
    def send_telegram_alert(self, person_name, confidence, photo_path):
        """Send Telegram alert with photo and message"""
        if not self.telegram_token or not self.telegram_chat_id:
            print("[WARNING] Telegram not configured - skipping alert")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Choose appropriate message
            if person_name == "Unknown":
                message = TELEGRAM_MESSAGES["unknown"].format(timestamp=timestamp)
                emoji_status = "🚨 UNKNOWN"
            else:
                if person_name in TELEGRAM_MESSAGES["known"]:
                    message = TELEGRAM_MESSAGES["known"][person_name].format(
                        timestamp=timestamp, 
                        confidence=confidence
                    )
                else:
                    # Generic known person message
                    message = f"✅ **KNOWN PERSON DETECTED** ✅\n\n👤 **Person:** {person_name}\n⏰ **Time:** {timestamp}\n✅ **Status:** Registered User\n🎯 **Confidence:** {confidence:.1%}"
                emoji_status = "✅ KNOWN"
            
            # Send photo with caption
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.telegram_chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, data=data, files=files, timeout=30)
            
            if response.status_code == 200:
                print(f"[TELEGRAM] ✅ Alert sent: {emoji_status} - {person_name}")
                return True
            else:
                print(f"[TELEGRAM] ❌ Failed to send alert: {response.status_code}")
                print(f"[TELEGRAM] Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Telegram alert failed: {e}")
            return False
    
    def speak_text(self, text):
        """Convert text to speech using Piper"""
        print(f"🔊 Speaking: {text}")
        try:
            # Generate audio with Piper
            command = [
                PIPER_EXECUTABLE,
                "--model", VOICE_PATH,
                "--config", CONFIG_PATH,
                "--output_file", RESPONSE_AUDIO
            ]
            subprocess.run(command, input=text.encode("utf-8"), check=True)
            
            # Play the audio
            subprocess.run(["aplay", RESPONSE_AUDIO], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ TTS failed: {e}")
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    def get_personalized_response(self, name):
        """Get a personalized response for the recognized person"""
        import random
        
        if name in PERSON_RESPONSES:
            responses = PERSON_RESPONSES[name]
        else:
            # Use generic responses for unknown people
            responses = PERSON_RESPONSES["Unknown"]
        
        return random.choice(responses)
    
    def should_recognize_person(self, name):
        """Check if enough time has passed since last recognition of this person"""
        current_time = time.time()
        if name not in self.last_recognition_time:
            self.last_recognition_time[name] = current_time
            return True
        
        time_since_last = current_time - self.last_recognition_time[name]
        if time_since_last >= self.recognition_cooldown:
            self.last_recognition_time[name] = current_time
            return True
        
        return False
    
    def save_detection_photo(self, frame, person_name, confidence):
        """Save a photo of the detected person to security_photos directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if person_name == "Unknown":
            filename = f"unknown_person_{timestamp}.jpg"
        else:
            filename = f"{person_name.lower()}_{timestamp}.jpg"
        
        # Save to security_photos directory
        filepath = os.path.join(SECURITY_PHOTOS_DIR, filename)
        
        # Add timestamp and info overlay to the image
        overlay_frame = frame.copy()
        
        # Add background rectangle for text
        cv2.rectangle(overlay_frame, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (10, 10), (500, 100), (255, 255, 255), 2)
        
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_frame, f"Person: {person_name}", (20, 35), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (20, 60), font, 0.6, (255, 255, 255), 2)
        if person_name != "Unknown":
            cv2.putText(overlay_frame, f"Confidence: {confidence:.1%}", (20, 85), font, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(filepath, overlay_frame)
        print(f"📸 Detection photo saved: {filepath}")
        
        # Log the detection
        self.log_detection(person_name, confidence, filename)
        
        return filepath
    
    def process_frame(self, frame):
        """Process a frame for facial recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Try both detection methods for better mask recognition
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        if len(face_locations) == 0:
            # Try CNN model if HOG fails (slower but sometimes better for masks)
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_names = []
        
        # Process each face found
        for face_encoding in face_encodings:
            # Compare with known faces - increased tolerance for masks
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                confidence = 1.0 - face_distances[best_match_index]
                
                # Lower confidence threshold for masks
                if matches[best_match_index] and confidence > 0.4:
                    name = self.known_names[best_match_index]
                    print(f"[DEBUG] Recognized {name} with confidence: {confidence:.2f}")
            
            recognized_names.append((name, confidence))
        
        # Draw rectangles and labels on faces
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, recognized_names):
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with confidence
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.1%})"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Handle recognition events
            if self.should_recognize_person(name):
                # Speak greeting
                response = self.get_personalized_response(name)
                print(f"[RECOGNIZED] {name} - {response}")
                self.speak_text(response)
                
                # Save photo and send Telegram alert
                photo_path = self.save_detection_photo(frame, name, confidence)
                self.send_telegram_alert(name, confidence, photo_path)
        
        return frame, [name for name, _ in recognized_names]
    
    def run_test(self):
        """Run the facial recognition test"""
        if not self.known_encodings:
            print("❌ No encodings loaded. Cannot run test.")
            return
        
        print("[INFO] Starting facial recognition test with Telegram alerts...")
        print("Press 'q' to quit, 's' to save current frame, 't' to test Telegram")
        
        # Initialize camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
        
        # Allow camera to warm up
        time.sleep(2)
        print("🎥 Camera ready! Look at the camera for recognition.")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                frame = picam2.capture_array()
                frame_count += 1
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0:
                    processed_frame, names = self.process_frame(frame.copy())
                    cv2.imshow('Facial Recognition with Telegram Alerts', processed_frame)
                else:
                    cv2.imshow('Facial Recognition with Telegram Alerts', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame to security_photos directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_save_{timestamp}.jpg"
                    filepath = os.path.join(SECURITY_PHOTOS_DIR, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"📸 Frame saved as {filepath}")
                elif key == ord('t'):
                    # Test Telegram connection
                    print("[INFO] Testing Telegram connection...")
                    if self.test_telegram_connection():
                        # Send test message
                        test_photo = self.save_detection_photo(frame, "Test", 1.0)
                        self.send_telegram_alert("Test", 1.0, test_photo)
        
        except KeyboardInterrupt:
            print("\n[INFO] Test interrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            picam2.stop()
            print("[INFO] Facial recognition test completed")

def main():
    """Main function"""
    print("🤖 Facial Recognition Test with Piper TTS & Telegram Alerts")
    print("=" * 60)
    
    # Check if required files exist
    if not os.path.exists(ENCODINGS_FILE):
        print(f"❌ Error: {ENCODINGS_FILE} not found!")
        print("Please run model_training.py first to create the face encodings.")
        return
    
    if not os.path.exists(VOICE_PATH):
        print(f"❌ Error: Voice file not found at {VOICE_PATH}")
        return
    
    if not os.path.exists(PIPER_EXECUTABLE):
        print(f"❌ Error: Piper executable not found at {PIPER_EXECUTABLE}")
        return
    
    # Initialize and run the test
    tester = FacialRecognitionTester()
    tester.run_test()

if __name__ == "__main__":
    main()