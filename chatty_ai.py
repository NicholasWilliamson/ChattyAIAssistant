#!/usr/bin/env python3
"""
chatty_ai.py - Complete AI Assistant with Facial Recognition and Wake Word Detection
Combines facial recognition, wake word detection, AI assistant, STT and TTS capabilities.
Updated with camera window display and proper ESC key handling.
"""

import os
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import random
import re
import cv2
import face_recognition
import pickle
import json
import requests
import logging
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from llama_cpp import Llama
from picamera2 import Picamera2

# -------------------------------
# Configuration
# -------------------------------
WHISPER_MODEL_SIZE = "base"
LLAMA_MODEL_PATH = "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
BEEP_SOUND = "/home/nickspi5/Chatty_AI/audio_files/beep.wav"
LAUGHING_SOUND = "/home/nickspi5/Chatty_AI/audio_files/laughing.wav"
ENCODINGS_FILE = "encodings.pickle"
TELEGRAM_CONFIG_FILE = "telegram_config.json"

# Audio files
WAV_FILENAME = "user_input.wav"
RESPONSE_AUDIO = "output.wav"
WAKE_WORD_AUDIO = "wake_word_check.wav"

# Security directories
SECURITY_PHOTOS_DIR = "/home/nickspi5/Chatty_AI/security_photos"
SECURITY_LOGS_DIR = "/home/nickspi5/Chatty_AI/security_logs"

# Response files
JOKES_FILE = "jokes.txt"
LISTENING_RESPONSES_FILE = "listening_responses.txt"
WAITING_RESPONSES_FILE = "waiting_responses.txt"
WARNING_RESPONSES_FILE = "warning_responses.txt"

# Wake word phrases
WAKE_WORDS = [
    "are you awake",
    "are you alive",
    "hey chatty",
    "hello chatty",
    "sup chatty",
    "sub-chatty",
    "how's it chatty",
    "howzit chatty",
    "hi chatty",
    "yo chatty",
    "hey chuddy",
    "hello chuddy",
    "sup chuddy",
    "sub-chuddy",
    "how's it chuddy",
    "howzit chuddy",
    "hi chuddy",
    "yo chuddy",
    "hey cheddy",
    "hello cheddy",
    "sup cheddy",
    "sub-cheddy",
    "how's it cheddy",
    "howzit cheddy",
    "hi cheddy",
    "yo cheddy",
    "hey chetty",
    "hello chetty",
    "sup chetty",
    "sub-chetty",
    "how's it chetty",
    "howzit chetty",
    "hi chetty",
    "yo chetty",
    "hey cherry",
    "hello cherry",
    "sup cherry",
    "sub-cherry",
    "how's it cherry",
    "howzit cherry",
    "hi cherry",
    "yo cherry"
]

# Command keywords
COMMANDS = {
    "flush the toilet": "toilet_flush",
    "turn on the lights": "lights_on",
    "turn off the lights": "lights_off",
    "play music": "play_music",
    "stop music": "stop_music",
    "what time is it": "get_time",
    "shutdown system": "shutdown_system",
    "who is sponsoring this video": "who_is_sponsoring_this_video",
    "how is the weather today": "how_is_the_weather_today",
    "reboot system": "reboot_system"
}

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.035
MIN_SILENCE_DURATION = 1.5
MAX_RECORDING_DURATION = 30

# Timing parameters
GREETING_COOLDOWN = 300  # 5 minutes in seconds
WAITING_INTERVAL = 30    # 30 seconds before offering help
PERSON_DETECTION_INTERVAL = 0.5  # Check for people every 0.5 seconds

def send_telegram_alert(message, photo_path=None):
    """Send alert to Telegram"""
    try:
        bot_token = telegram_config.get('bot_token')
        chat_ids = telegram_config.get('chat_ids', [])
        
        if not bot_token or not chat_ids:
            return False
            
        for chat_id in chat_ids:
            if photo_path and os.path.exists(photo_path):
                # Send photo with caption
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(photo_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {'chat_id': chat_id, 'caption': message}
                    response = requests.post(url, files=files, data=data, timeout=10)
            else:
                # Send text message
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {'chat_id': chat_id, 'text': message}
                response = requests.post(url, json=data, timeout=10)
                
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def speak_text(text):
    """Convert text to speech using Piper TTS"""
    try:
        # Use Piper TTS
        cmd = [
            'echo', text, '|', 
            './piper/piper', 
            '--model', './piper/en_US-lessac-medium.onnx', 
            '--output_file', 'recognition_response.wav'
        ]
        subprocess.run(' '.join(cmd), shell=True, check=True)
        
        # Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load('recognition_response.wav')
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"TTS Error: {e}")

def save_detection_photo(name, frame):
    """Save detection photo with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"security_photos/{name.lower()}_{timestamp}.jpg"
        os.makedirs("security_photos", exist_ok=True)
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        print(f"Error saving photo: {e}")
        return None

def process_face_recognition(frame, responses):
    """Process facial recognition on frame"""
    global recognition_active
    
    if not recognition_active:
        return frame
        
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for (face_encoding, face_location) in zip(face_encodings_frame, face_locations):
        # Check against known faces
        matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(face_encodings, face_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                print(f"[DEBUG] Recognized {name} with confidence: {confidence:.2f}")
                
                # Get response for this person
                greeting = responses.get('greetings', {}).get(name.lower(), 
                    f"Hello {name}! How can I help you today?")
                
                print(f"[RECOGNIZED] {name} - {greeting}")
                print(f"🔊 Speaking: {greeting}")
                
                # Speak greeting in separate thread
                threading.Thread(target=speak_text, args=(greeting,), daemon=True).start()
                
                # Save photo
                photo_path = save_detection_photo(name, frame)
                
                # Send Telegram alert
                telegram_msg = f"✅ KNOWN - {name}"
                threading.Thread(target=send_telegram_alert, 
                               args=(telegram_msg, photo_path), daemon=True).start()
                
                # Pause recognition for 30 seconds
                recognition_active = False
                threading.Timer(30.0, lambda: setattr(sys.modules[__name__], 'recognition_active', True)).start()
                
                # Draw rectangle and name
                top, right, bottom, left = face_location
                top *= 4; right *= 4; bottom *= 4; left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Unknown face
                print("[UNKNOWN] Unrecognized person detected")
                
                # Save photo
                photo_path = save_detection_photo("unknown", frame)
                
                # Send Telegram alert
                telegram_msg = "🚨 UNKNOWN PERSON DETECTED"
                threading.Thread(target=send_telegram_alert, 
                               args=(telegram_msg, photo_path), daemon=True).start()
                
                # Draw rectangle
                top, right, bottom, left = face_location
                top *= 4; right *= 4; bottom *= 4; left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def initialize_camera():
    """Initialize the camera"""
    global picam2
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Let camera warm up
        print("Camera initialized")
        return True
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return False

def load_ai_models():
    """Load AI models (placeholder)"""
    print("Loading AI models...")
    print("Whisper model loaded")
    print("LLaMA model loaded")

def main():
    """Main function"""
    print("Chatty AI - Your smart AI Assistant System")
    print("=" * 60)
    
    chatty = ChattyAI()
    chatty.run()

if __name__ == "__main__":
    main()