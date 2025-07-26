#!/usr/bin/env python3
"""
test_facial_recognition.py
Test facial recognition with personalized TTS responses using Piper
"""

import cv2
import face_recognition
import pickle
import os
import subprocess
import time
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

# Personalized responses for different people
PERSON_RESPONSES = {
    "Nick": [
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?",
        "Hello Nick my master. It is wonderful to see you again. Thank you so much for creating me. How can I help you my friend?"
    ],
    "Spiderman": [
        "Hello Spider Man. O M G. You are my favourite super hero. It will be my honor to help you!",
        "Hello Spider Man. O M G. You are my favourite super hero. It will be my honor to help you!",
        "Hello Spider Man. O M G. You are my favourite super hero. It will be my honor to help you!",
        "Hello Spider Man. O M G. You are my favourite super hero. It will be my honor to help you!"
    ],
    "Unknown": [
        "Hello there! I don't recognize you yet.",
        "Hi! You're new to me. Nice to meet you!",
        "Hello stranger! Would you like to be registered?",
        "Hi there! I haven't seen you before."
    ]
}

class FacialRecognitionTester:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_encodings()
        self.last_recognition_time = {}
        self.recognition_cooldown = 5  # seconds between recognitions for same person
        
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
            
            recognized_names.append(name)
        
        # Draw rectangles and labels on faces
        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Speak greeting if enough time has passed
            if self.should_recognize_person(name):
                response = self.get_personalized_response(name)
                print(f"[RECOGNIZED] {name} - {response}")
                self.speak_text(response)
        
        return frame, recognized_names
    
    def run_test(self):
        """Run the facial recognition test"""
        if not self.known_encodings:
            print("❌ No encodings loaded. Cannot run test.")
            return
        
        print("[INFO] Starting facial recognition test...")
        print("Press 'q' to quit, 's' to save current frame")
        
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
                    cv2.imshow('Facial Recognition Test', processed_frame)
                else:
                    cv2.imshow('Facial Recognition Test', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recognition_test_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved as {filename}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Test interrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            picam2.stop()
            print("[INFO] Facial recognition test completed")

def main():
    """Main function"""
    print("🤖 Facial Recognition Test with Piper TTS")
    print("=" * 50)
    
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