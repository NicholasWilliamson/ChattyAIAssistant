#!/usr/bin/env python3
"""
chatty_ai.py - Complete AI Assistant with Facial Recognition and Wake Word Detection
Enhanced with personalized greetings and responses for different users
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
GREETING_RESPONSES_FILE = "greeting_responses.txt"
PERSONALIZED_RESPONSES_FILE = "personalized_responses.json"  # NEW FILE

# Wake word phrases
WAKE_WORDS = [
    "are you awake", "are you alive", "hey chatty", "hello chatty", "sup chatty",
    "sub-chatty", "how's it chatty", "howzit chatty", "hi chatty", "yo chatty",
    "hey chuddy", "hello chuddy", "sup chuddy", "sub-chuddy", "how's it chuddy",
    "howzit chuddy", "hi chuddy", "yo chuddy", "hey cheddy", "hello cheddy",
    "sup cheddy", "sub-cheddy", "how's it cheddy", "howzit cheddy", "hi cheddy",
    "yo cheddy", "hey chetty", "hello chetty", "sup chetty", "sub-chetty",
    "how's it chetty", "howzit chetty", "hi chetty", "yo chetty", "hey cherry",
    "hello cherry", "sup cherry", "sub-cherry", "how's it cherry", "howzit cherry",
    "hi cherry", "yo cherry"
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
WAKE_WORD_CHECK_INTERVAL = 1.0  # Check for wake words every 1 second

class ChattyAI:
    def __init__(self):
        # AI Models
        self.whisper_model = None
        self.llama_model = None
        
        # Facial Recognition
        self.known_encodings = []
        self.known_names = []
        
        # Camera
        self.picam2 = None
        
        # State variables
        self.is_running = False
        self.current_person = None
        self.last_greeting_time = {}
        self.last_interaction_time = None
        self.person_absent_since = None
        self.waiting_cycle = 0  # 0: joke, 1: fun fact
        self.last_bored_response_time = None  # NEW: Track when last bored response was given
        self.bored_cycle = 0  # NEW: 0: joke, 1: fun fact
        self.audio_recording_lock = threading.Lock()
        self.wake_word_active = False

        # Response lists
        self.jokes = []
        self.listening_responses = []
        self.waiting_responses = []
        self.warning_responses = []
        self.greeting_responses = []
        self.personalized_responses = {}  # NEW: Dictionary for personalized responses

        # Telegram
        self.telegram_token = None
        self.telegram_chat_id = None

        # Threading
        self.camera_thread = None
        self.audio_thread = None
        
        # Initialize everything
        self.setup_directories()
        self.load_response_files()
        self.load_personalized_responses()  # NEW
        self.load_models()
        self.load_encodings()
        self.load_telegram_config()
        self.setup_camera()
        self.setup_logging()
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(SECURITY_PHOTOS_DIR, exist_ok=True)
        os.makedirs(SECURITY_LOGS_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging for detections"""
        log_file = os.path.join(SECURITY_LOGS_DIR, "chatty_ai.log")
        self.logger = logging.getLogger('chatty_ai')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def load_response_files(self):
        """Load response text files"""
        try:
            with open(JOKES_FILE, 'r') as f:
                self.jokes = [line.strip() for line in f if line.strip()]
            
            with open(LISTENING_RESPONSES_FILE, 'r') as f:
                self.listening_responses = [line.strip() for line in f if line.strip()]

            with open(GREETING_RESPONSES_FILE, 'r') as f:
                self.greeting_responses = [line.strip() for line in f if line.strip()]

            with open(WAITING_RESPONSES_FILE, 'r') as f:
                self.waiting_responses = [line.strip() for line in f if line.strip()]
            
            with open(WARNING_RESPONSES_FILE, 'r') as f:
                self.warning_responses = [line.strip() for line in f if line.strip()]
            
            print("Response files loaded successfully")
        except FileNotFoundError as e:
            print(f"Response file not found: {e}")
            # Create default responses if files don't exist
            self.create_default_responses()
    
    def load_personalized_responses(self):
        """Load personalized responses from JSON file"""
        try:
            with open(PERSONALIZED_RESPONSES_FILE, 'r') as f:
                self.personalized_responses = json.load(f)
            print(f"Personalized responses loaded for {len(self.personalized_responses)} people")
        except FileNotFoundError:
            print(f"Personalized responses file not found. Creating default...")
            self.create_default_personalized_responses()
        except json.JSONDecodeError as e:
            print(f"Error reading personalized responses JSON: {e}")
            self.create_default_personalized_responses()
    
    def create_default_personalized_responses(self):
        """Create default personalized responses file"""
        default_responses = {
            "Nick": {
                "greetings": [
                    "Hello Nick, my master! It is so lovely to see you again. Thank you for creating me. How may I assist you?",
                    "Welcome back Nick! Your brilliant creation is ready to serve. What can I help you with today?",
                    "Nick! My creator has returned! I've been waiting patiently for your commands."
                ],
                "listening": [
                    "Yes Nick, I'm listening. What would you like to know?",
                    "I'm all ears, Nick. What's on your mind?",
                    "Ready for your instructions, Nick!"
                ],
                "waiting": [
                    "Nick, I'm still here if you need anything",
                    "Your faithful AI assistant is standing by, Nick",
                    "Still waiting to help you, Nick"
                ],
                "bored_responses": [
                    "Nick, I'm getting a bit bored waiting here.",
                    "Master Nick, I'm still patiently waiting for your commands."
                ],
                "joke_responses": [
                    "Why don't programmers like nature? It has too many bugs!",
                    "Why do Python programmers prefer snake_case? Because they can't C the point of CamelCase!"
                ],
                "fun_fact_responses": [
                    "Did you know that the first computer bug was an actual bug? Grace Hopper found a moth stuck in a computer relay in 1947!",
                    "Fun fact: Raspberry Pi computers like the one I'm running on are more powerful than the computers that sent humans to the moon!"
                ]
            },
            "Anne": {
                "greetings": [
                    "Hello Anne! How wonderful to see you. I hope you're having a lovely day!",
                    "Anne! Welcome! It's always a pleasure to see you.",
                    "Hi Anne! You look fantastic today. How can I help you?"
                ],
                "listening": [
                    "Yes Anne, what can I do for you?",
                    "I'm here to help, Anne. What do you need?",
                    "How may I assist you today, Anne?"
                ],
                "waiting": [
                    "Anne, I'm here if you need anything",
                    "Still available to help you, Anne",
                    "Let me know if you need assistance, Anne"
                ],
                "bored_responses": [
                    "Anne, I'm here patiently waiting if you need anything.",
                    "Still here for you, Anne, whenever you're ready."
                ],
                "joke_responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "What do you call a bear with no teeth? A gummy bear!"
                ],
                "fun_fact_responses": [
                    "Did you know that octopuses have three hearts and blue blood?",
                    "Fun fact: Honey never spoils! Archaeologists have found 3000-year-old honey that's still perfectly edible."
                ]
            },
            "Jack": {
                "greetings": [
                    "Hey Jack! Good to see you, buddy! What's up?",
                    "Jack! How's it going, mate? Ready for some fun?",
                    "What's up Jack! Hope you're having an awesome day!"
                ],
                "listening": [
                    "Yeah Jack, what do you need?",
                    "I'm listening, Jack. Fire away!",
                    "What can I help you with, Jack?"
                ],
                "waiting": [
                    "Jack, I'm here if you want to chat",
                    "Still around if you need me, Jack",
                    "Ready when you are, Jack"
                ],
                "bored_responses": [
                    "Yo Jack, still hanging around here waiting for you, dude!",
                    "Jack, your AI buddy is getting restless over here!"
                ],
                "joke_responses": [
                    "Why don't skeletons fight each other? They don't have the guts!",
                    "What's the best thing about Switzerland? I don't know, but the flag is a big plus!"
                ],
                "fun_fact_responses": [
                    "Dude, did you know that sharks have been around longer than trees? That's wild!",
                    "Check this out: A shrimp's heart is in its head!"
                ]
            }
        }
        
        # Save the default responses
        try:
            with open(PERSONALIZED_RESPONSES_FILE, 'w') as f:
                json.dump(default_responses, f, indent=4)
            self.personalized_responses = default_responses
            print(f"Created default personalized responses file: {PERSONALIZED_RESPONSES_FILE}")
        except Exception as e:
            print(f"Failed to create personalized responses file: {e}")
            self.personalized_responses = default_responses
    
    def get_personalized_response(self, person_name, response_type, fallback_list=None):
        """Get a personalized response for a specific person and response type"""
        person_name_lower = person_name.lower()
        
        # Check if we have personalized responses for this person
        person_data = None
        for name, data in self.personalized_responses.items():
            if name.lower() == person_name_lower:
                person_data = data
                break
        
        if person_data and response_type in person_data:
            responses = person_data[response_type]
            if responses:
                return random.choice(responses)
        
        # Fallback to generic responses with name insertion
        if fallback_list:
            response = random.choice(fallback_list)
            # Replace {name} placeholder if it exists, otherwise prepend name
            if "{name}" in response:
                return response.replace("{name}", person_name)
            else:
                # Simple name insertion at the beginning
                return f"Hello {person_name}! {response}"
        
        # Ultimate fallback
        return f"Hello {person_name}! How can I help you today?"
    
    def create_default_responses(self):
        """Create default responses if files are missing"""
        self.jokes = ["Why don't scientists trust atoms? Because they make up everything!"]
        self.greeting_responses = ["Hello {name}! How can I help you today?"]
        self.listening_responses = ["I'm listening, what would you like to know?"]
        self.waiting_responses = ["I'm still here if you need anything"]
        self.warning_responses = ["Warning: Unknown person detected. Please identify yourself."]
    
    def load_models(self):
        """Load AI models"""
        print("Loading AI models...")
        
        try:
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("Whisper model loaded")
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
            return False
        
        try:
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=2048,
                temperature=0.7,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                verbose=False
            )
            print("LLaMA model loaded")
        except Exception as e:
            print(f"Failed to load LLaMA: {e}")
            return False
        
        return True
    
    def load_encodings(self):
        """Load facial recognition encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"Loaded {len(self.known_encodings)} face encodings")
            return True
        except FileNotFoundError:
            print(f"Encodings file '{ENCODINGS_FILE}' not found!")
            return False
        except Exception as e:
            print(f"Failed to load encodings: {e}")
            return False
    
    def load_telegram_config(self):
        """Load Telegram configuration"""
        try:
            with open(TELEGRAM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.telegram_token = config.get('bot_token')
                self.telegram_chat_id = config.get('chat_id')
            print("Telegram configuration loaded")
        except FileNotFoundError:
            print("Telegram config not found - alerts disabled")
        except Exception as e:
            print(f"Failed to load Telegram config: {e}")
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            ))
            self.picam2.start()
            time.sleep(2)  # Camera warm-up
            print("Camera initialized")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def speak_text(self, text):
        """Convert text to speech using Piper"""
        try:
            # Use lock to prevent audio conflicts
            with self.audio_recording_lock:
                command = [
                    PIPER_EXECUTABLE,
                    "--model", VOICE_PATH,
                    "--config", CONFIG_PATH,
                    "--output_file", RESPONSE_AUDIO
                ]
                subprocess.run(command, input=text.encode("utf-8"), check=True, capture_output=True)
                subprocess.run(["aplay", RESPONSE_AUDIO], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"TTS failed: {e}")
    
    def play_beep(self):
        """Play beep sound"""
        try:
            with self.audio_recording_lock:
                subprocess.run(["aplay", BEEP_SOUND], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass
    
    def play_laughing(self):
        """Play laughing sound"""
        try:
            with self.audio_recording_lock:
                subprocess.run(["aplay", LAUGHING_SOUND], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass
    
    def detect_faces(self, frame):
        """Detect and recognize faces in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if len(face_locations) == 0:
            return None, None, 0.0
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        best_name = "Unknown"
        best_confidence = 0.0
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.6)
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                confidence = 1.0 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > 0.4:
                    if confidence > best_confidence:
                        best_name = self.known_names[best_match_index]
                        best_confidence = confidence
        
        return best_name, face_locations[0], best_confidence
    
    def save_security_photo(self, frame, person_name, confidence):
        """Save security photo with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_name.lower()}_{timestamp}.jpg"
        filepath = os.path.join(SECURITY_PHOTOS_DIR, filename)
        
        # Add overlay information
        overlay_frame = frame.copy()
        cv2.rectangle(overlay_frame, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (10, 10), (500, 100), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_frame, f"Person: {person_name}", (20, 35), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay_frame, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (20, 60), font, 0.6, (255, 255, 255), 2)
        if person_name != "Unknown":
            cv2.putText(overlay_frame, f"Confidence: {confidence:.1%}", (20, 85), font, 0.6, (255, 255, 255), 2)
        
        cv2.imwrite(filepath, overlay_frame)
        self.logger.info(f"Security photo saved: {filename} | Person: {person_name} | Confidence: {confidence:.2f}")
        
        return filepath
    
    def send_telegram_alert(self, person_name, confidence, photo_path):
        """Send Telegram alert"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if person_name == "Unknown":
                message = f"**UNKNOWN PERSON DETECTED**\n\n**Time:** {timestamp}\n**Status:** Unregistered Person\n**Action:** Photo captured for review"
            else:
                message = f"**AUTHORIZED ACCESS**\n\n**Person:** {person_name}\n**Time:** {timestamp}\n**Confidence:** {confidence:.1%}\n**Status:** Registered User"
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.telegram_chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, data=data, files=files, timeout=30)
                return response.status_code == 200
        except Exception as e:
            print(f"Telegram alert failed: {e}")
            return False
    
    def greet_person(self, name):
        """Greet a detected person with personalized greeting"""
        current_time = time.time()
        
        # Check if we should greet this person (cooldown check)
        if name in self.last_greeting_time:
            time_since_last = current_time - self.last_greeting_time[name]
            if time_since_last < GREETING_COOLDOWN:
                return False
        
        # Get personalized greeting (just the greeting, not listening response)
        greeting = self.get_personalized_response(name, "greetings", self.greeting_responses)
        
        self.speak_text(greeting)
        self.last_greeting_time[name] = current_time
        self.last_interaction_time = current_time
        
        # Enable wake word detection after greeting
        self.wake_word_active = True
        self.last_bored_response_time = current_time  # NEW: Reset bored response timer
        self.bored_cycle = 0  # NEW: Reset bored cycle
        print(f"Greeted {name} with personalized message - Wake word detection now active")
        return True
    
    def handle_unknown_person(self, frame, confidence):
        """Handle unknown person detection"""
        warning = random.choice(self.warning_responses) if self.warning_responses else "Warning: Unknown person detected."
        self.speak_text(warning)
        
        photo_path = self.save_security_photo(frame, "Unknown", confidence)
        self.send_telegram_alert("Unknown", confidence, photo_path)
        
        print("Unknown person detected and warned")
    
    def check_for_bored_response(self, name):
        """Check if it's time to give a bored response with joke or fun fact"""
        if not self.wake_word_active or not self.last_bored_response_time:
            return False
        
        current_time = time.time()
        time_since_bored = current_time - self.last_bored_response_time
        
        if time_since_bored >= 30:  # 30 seconds
            if self.bored_cycle == 0:
                # Give bored response + joke
                bored_msg = self.get_personalized_response(name, "bored_responses", ["I'm still here waiting to help you"])
                joke = self.get_personalized_response(name, "joke_responses", self.jokes)
                full_message = f"{bored_msg} {joke}"
                self.speak_text(full_message)
                self.bored_cycle = 1
                print(f"Gave {name} a bored response with joke")
            else:
                # Give bored response + fun fact
                bored_msg = self.get_personalized_response(name, "bored_responses", ["I'm still here waiting to help you"])
                fun_fact = self.get_personalized_response(name, "fun_fact_responses", ["Did you know that honey never spoils?"])
                full_message = f"{bored_msg} {fun_fact}"
                self.speak_text(full_message)
                self.bored_cycle = 0
                print(f"Gave {name} a bored response with fun fact")
            
            self.last_bored_response_time = current_time
            return True
        
        return False
        """Offer help or entertainment to waiting person"""
        current_time = time.time()
        
        if not self.last_interaction_time or (current_time - self.last_interaction_time) >= WAITING_INTERVAL:
            if self.waiting_cycle == 0:
                # Offer joke
                if self.jokes:
                    waiting_msg = self.get_personalized_response(name, "waiting", self.waiting_responses)
                    joke = random.choice(self.jokes)
                    message = f"{waiting_msg} Here's a joke for you: {joke}"
                    self.speak_text(message)
                    self.waiting_cycle = 1
                    print(f"Told {name} a joke")
            else:
                # Offer fun fact
                waiting_msg = self.get_personalized_response(name, "waiting", self.waiting_responses)
                fun_fact = "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!"
                message = f"{waiting_msg} Here's a fun fact: {fun_fact}"
                self.speak_text(message)
                self.waiting_cycle = 0
                print(f"Told {name} a fun fact")
            
            self.last_interaction_time = current_time
    
    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper"""
        try:
            if not os.path.exists(filename):
                return ""
            
            segments, _ = self.whisper_model.transcribe(filename)
            transcript = " ".join(segment.text for segment in segments).strip()
            print(f"Transcription: '{transcript}'")  # Debug output
            return transcript
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def detect_wake_word(self, text):
        """Check if text contains wake word"""
        if not text:
            return False
            
        text_cleaned = text.lower().replace(',', '').replace('.', '').strip()
        
        for wake_word in WAKE_WORDS:
            wake_word_cleaned = wake_word.lower().strip()
            if wake_word_cleaned in text_cleaned:
                print(f"Wake word detected: '{wake_word}' in '{text}'")
                return True
        return False
    
    def record_with_silence_detection(self):
        """Record audio until silence detected"""
        try:
            with self.audio_recording_lock:
                print("Recording audio...")
                audio_data = []
                silence_duration = 0
                recording_duration = 0
                check_interval = 0.2
                samples_per_check = int(SAMPLE_RATE * check_interval)
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        print(f"Audio callback status: {status}")
                    audio_data.extend(indata[:, 0])
                
                with sd.InputStream(callback=audio_callback, 
                                  samplerate=SAMPLE_RATE, 
                                  channels=CHANNELS,
                                  dtype='float32'):
                    
                    while recording_duration < MAX_RECORDING_DURATION:
                        time.sleep(check_interval)
                        recording_duration += check_interval
                        
                        if len(audio_data) >= samples_per_check:
                            recent_audio = np.array(audio_data[-samples_per_check:])
                            rms = np.sqrt(np.mean(recent_audio**2))
                            
                            if rms < SILENCE_THRESHOLD:
                                silence_duration += check_interval
                                if silence_duration >= MIN_SILENCE_DURATION:
                                    print(f"Silence detected after {recording_duration:.1f}s")
                                    break
                            else:
                                silence_duration = 0
                
                if audio_data:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sf.write(WAV_FILENAME, audio_array, SAMPLE_RATE)
                    print(f"Audio saved: {len(audio_array)/SAMPLE_RATE:.1f}s duration")
                    return True
                
                return False
                
        except Exception as e:
            print(f"Recording error: {e}")
            return False
    
    def record_wake_word_check(self):
        """Record short audio clip for wake word detection"""
        try:
            if not self.audio_recording_lock.acquire(blocking=False):
                return False  # Audio system is busy
            
            try:
                # Record 5 seconds of audio for wake word detection
                print("Listening for wake word...")
                audio_data = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                
                # Check if audio contains sound above threshold
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > SILENCE_THRESHOLD * 2:  # Higher threshold for wake word
                    sf.write(WAKE_WORD_AUDIO, audio_data, SAMPLE_RATE)
                    print(f"Wake word audio saved, RMS: {rms:.4f}")
                    return True
                else:
                    print(f"Audio too quiet for wake word, RMS: {rms:.4f}")
                    return False
                    
            finally:
                self.audio_recording_lock.release()
                
        except Exception as e:
            print(f"Wake word recording error: {e}")
            if self.audio_recording_lock.locked():
                self.audio_recording_lock.release()
            return False
    
    def is_command(self, text):
        """Check if text is a command"""
        text_lower = text.lower().strip()
        for command in COMMANDS.keys():
            if command in text_lower:
                return command
        return None
    
    def execute_command(self, command):
        """Execute system command"""
        if command == "flush the toilet":
            response = f"Oh {self.current_person}, you know I am a digital assistant. I cannot actually flush toilets! So why dont you haul your lazy butt up off the couch and flush the toilet yourself!"
        elif command == "turn on the lights":
            response = "I would turn on the lights if I was connected to a smart home system."
        elif command == "turn off the lights":
            response = "I would turn off the lights if I was connected to a smart home system."
        elif command == "play music":
            response = "I would start playing music if I had access to a music system."
        elif command == "stop music":
            response = "I would stop the music if any music was playing."
        elif command == "who is sponsoring this video":
            self.play_laughing()
            response = f"You are very funny {self.current_person}. You know you dont have any sponsors for your videos!"
        elif command == "how is the weather today":
            response = f"O M G {self.current_person}! Surely you DO NOT want to waste my valuable resources by asking me what the weather is today. Cant you just look out the window or ask Siri. That is about all Siri is good for!"
        elif command == "what time is it":
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
        elif command == "shutdown system":
            response = "I would shutdown the system, but I will skip that for safety reasons during testing."
        elif command == "reboot system":
            response = "I would reboot the system, but I will skip that for safety reasons during testing."
        else:
            response = f"I understand you want me to {command}, but I dont have that capability yet."
        
        return response
    
    def query_llama(self, prompt):
        """Generate LLM response"""
        formatted_prompt = f"You are a friendly, helpful assistant. Give a brief, conversational answer.\nUser: {prompt}\nAssistant: "
        
        try:
            result = self.llama_model(formatted_prompt, max_tokens=100)
            if "choices" in result and result["choices"]:
                reply_text = result["choices"][0]["text"].strip()
                reply_text = re.sub(r"\(.*?\)", "", reply_text)
                reply_text = re.sub(r"(User:|Assistant:)", "", reply_text)
                reply_text = reply_text.strip()
                
                sentences = reply_text.split('.')
                if len(sentences) > 3:
                    reply_text = '. '.join(sentences[:3]) + '.'
                
                return reply_text
            else:
                return "I'm not sure how to answer that."
        except Exception as e:
            print(f"LLM error: {e}")
            return "Sorry, I had trouble processing that question."
    
    def process_user_input(self, text):
        """Process user input"""
        print(f"Processing user input: '{text}'")
        command = self.is_command(text)
        if command:
            print(f"Executing command: {command}")
            response = self.execute_command(command)
        else:
            print("Generating LLM response")
            response = self.query_llama(text)
        
        return response
    
    def listen_for_wake_word(self):
        """Listen for wake words in background"""
        print("Wake word detection thread started")
        
        while self.is_running:
            try:
                # Only listen if someone is present and wake word detection is active
                if self.current_person and self.current_person != "Unknown" and self.wake_word_active:
                    # Check for bored response first
                    if self.check_for_bored_response(self.current_person):
                        # Bored response was given, continue to next iteration
                        time.sleep(WAKE_WORD_CHECK_INTERVAL)
                        continue
                    
                    print("Checking for wake word...")
                    
                    # Record audio for wake word detection
                    if self.record_wake_word_check():
                        # Transcribe and check for wake word
                        transcript = self.transcribe_audio(WAKE_WORD_AUDIO)
                        
                        if transcript and self.detect_wake_word(transcript):
                            print("WAKE WORD DETECTED! Starting conversation...")
                            self.play_beep()
                            
                            # Speak personalized listening response
                            listening_response = self.get_personalized_response(self.current_person, "listening", self.listening_responses)
                            self.speak_text(listening_response)
                            
                            # Record full request
                            print("Please speak your request...")
                            if self.record_with_silence_detection():
                                user_text = self.transcribe_audio(WAV_FILENAME)
                                if user_text and len(user_text.strip()) > 2:
                                    print(f"User said: '{user_text}'")
                                    response = self.process_user_input(user_text)
                                    print(f"Response: '{response}'")
                                    self.speak_text(response)
                                    self.last_interaction_time = time.time()
                                    # Reset bored response timer only after successful interaction
                                    self.last_bored_response_time = time.time()
                                else:
                                    print("No clear speech detected")
                                    self.speak_text("I didn't catch that. Could you repeat your request?")
                            else:
                                print("Failed to record user request")
                                self.speak_text("I'm having trouble hearing you. Please try again.")
                    
                    time.sleep(WAKE_WORD_CHECK_INTERVAL)
                else:
                    # No one present or wake word not active, sleep longer
                    time.sleep(2.0)
                
            except Exception as e:
                print(f"Wake word detection error: {e}")
                time.sleep(2.0)
        
        print("Wake word detection thread stopped")
    
    def camera_monitoring_loop(self):
        """Main camera monitoring loop with OpenCV display"""
        cv2.namedWindow('Chatty AI - Facial Recognition', cv2.WINDOW_AUTOSIZE)
        
        while self.is_running:
            try:
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process facial recognition
                name, face_location, confidence = self.detect_faces(frame)
                
                current_time = time.time()
                
                # Draw face rectangles and labels on frame
                if name and face_location:
                    top, right, bottom, left = face_location
                    
                    # Choose color based on recognition
                    if name == "Unknown":
                        color = (0, 0, 255)  # Red for unknown
                        label = f"Unknown ({confidence:.2f})"
                    else:
                        color = (0, 255, 0)  # Green for known
                        label = f"{name} ({confidence:.2f})"
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # Draw label text
                    cv2.putText(frame, label, (left + 6, bottom - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add status information to frame
                status_text = "🤖 Chatty AI Active - Press ESC to exit"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show current person status
                if self.current_person:
                    person_text = f"Current Person: {self.current_person}"
                    cv2.putText(frame, person_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show wake word status
                if self.wake_word_active:
                    wake_word_text = "Wake Word Detection: ACTIVE"
                    cv2.putText(frame, wake_word_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    wake_word_text = "Wake Word Detection: INACTIVE"
                    cv2.putText(frame, wake_word_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('Chatty AI - Facial Recognition', frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("\n[INFO] ESC key pressed. Shutting down...")
                    self.is_running = False
                    break
                elif key == ord('q'):  # Q key as alternative
                    print("\n[INFO] Q key pressed. Shutting down...")
                    self.is_running = False
                    break
                
                # Process facial recognition logic
                if name and face_location:
                    # Person detected
                    if name != self.current_person:
                        # New person or person changed
                        self.current_person = name
                        self.person_absent_since = None
                        self.wake_word_active = False  # Reset wake word state
                        
                        if name == "Unknown":
                            self.handle_unknown_person(frame, confidence)
                        else:
                            # Save photo and send telegram alert for known person
                            photo_path = self.save_security_photo(frame, name, confidence)
                            self.send_telegram_alert(name, confidence, photo_path)
                            
                            # Greet known person (this will activate wake word detection)
                            self.greet_person(name)
                    
                    elif name != "Unknown":
                        # Same known person still present
                        self.offer_help_or_entertainment(name)
                
                else:
                    # No person detected
                    if self.current_person:
                        if not self.person_absent_since:
                            self.person_absent_since = current_time
                        elif current_time - self.person_absent_since >= GREETING_COOLDOWN:
                            # Person has been absent for 5+ minutes, reset
                            self.current_person = None
                            self.person_absent_since = None
                            self.last_interaction_time = None
                            self.waiting_cycle = 0
                            self.wake_word_active = False
                            self.last_bored_response_time = None  # NEW: Reset bored response timer
                            self.bored_cycle = 0  # NEW: Reset bored cycle
                            print("Person left - resetting state")
                
                time.sleep(PERSON_DETECTION_INTERVAL)
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                time.sleep(1)
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
    
    def run(self):
        """Main run loop"""
        if not self.whisper_model or not self.llama_model or not self.picam2:
            print("Required components not initialized")
            return
        
        print("🚀 Chatty AI Complete System Started!")
        print("=" * 60)
        print("Features active:")
        print("• Facial Recognition with Personalized Greetings")
        print("• Wake Word Detection")
        print("• AI Assistant (TinyLLaMA)")
        print("• Security Monitoring")
        print("• Telegram Alerts")
        print("• Proactive Entertainment")
        print("=" * 60)
        print("Press ESC key to exit")
        print("\nDEBUG INFO:")
        print(f"• Wake words: {len(WAKE_WORDS)} phrases loaded")
        print(f"• Personalized responses: {len(self.personalized_responses)} people configured")
        print(f"• Audio sample rate: {SAMPLE_RATE} Hz")
        print(f"• Silence threshold: {SILENCE_THRESHOLD}")
        print("=" * 60)
        
        self.is_running = True
        
        # Start wake word detection in background thread
        self.audio_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
        self.audio_thread.start()
        
        try:
            # Run camera monitoring loop (this will handle the display and ESC key)
            self.camera_monitoring_loop()
                
        except KeyboardInterrupt:
            print("\nShutting down Chatty AI...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.is_running = False
        
        # Wait for audio thread to finish
        if self.audio_thread and self.audio_thread.is_alive():
            print("Waiting for audio thread to stop...")
            self.audio_thread.join(timeout=3)
        
        if self.picam2:
            try:
                self.picam2.stop()
                print("Camera stopped")
            except:
                pass
        
        cv2.destroyAllWindows()
        
        # Clean up audio files
        for audio_file in [WAV_FILENAME, RESPONSE_AUDIO, WAKE_WORD_AUDIO]:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
        
        print("Chatty AI shutdown complete")

def main():
    """Main function"""
    print("Chatty AI - Your smart AI Assistant System")
    print("=" * 60)
    
    # Check audio devices
    try:
        print("Available audio devices:")
        print(sd.query_devices())
        print("=" * 60)
    except Exception as e:
        print(f"Could not query audio devices: {e}")
    
    chatty = ChattyAI()
    chatty.run()

if __name__ == "__main__":
    main()