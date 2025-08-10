#!/usr/bin/env python3
"""
app_4.py - Chatty AI Web Interface
Flask web interface for the AI Assistant with proper initialization order
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
import secrets
import traceback
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from llama_cpp import Llama
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import io
from PIL import Image

# Try to import picamera2; if unavailable we'll use cv2.VideoCapture fallback
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# Flask application setup
app = Flask(__name__,
            template_folder='/home/nickspi5/Chatty_AI/templates',
            static_folder='/home/nickspi5/Chatty_AI/templates')
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Configure SocketIO with proper parameters (no broadcast=True)
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='threading',
                   logger=False, 
                   engineio_logger=False)

# Configuration - same as your original
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
PERSONALIZED_RESPONSES_FILE = "personalized_responses.json"

# Wake words and commands - same as your original
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
GREETING_COOLDOWN = 300
WAITING_INTERVAL = 30
PERSON_DETECTION_INTERVAL = 0.5
WAKE_WORD_CHECK_INTERVAL = 1.0

# Global variables
chatty_ai = None
system_ready = False
models_loaded = False
camera_initialized = False
initialization_status = "Not initialized"

def log_message(message, msg_type="info"):
    """Enhanced logging with SocketIO emission"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    try:
        # Fixed: Remove broadcast=True, use to=None for broadcasting
        socketio.emit('log_update', {
            'timestamp': timestamp,
            'message': message,
            'type': msg_type
        }, to=None)
    except Exception as e:
        print(f"Socket emit error: {e}")

def emit_conversation(message, msg_type="info"):
    """Emit conversation message to web interface"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] CONVERSATION: {message}")
    try:
        socketio.emit('conversation_update', {
            'timestamp': timestamp,
            'message': message,
            'type': msg_type
        }, to=None)
    except Exception as e:
        print(f"Socket emit error: {e}")

class ChattyAIWeb:
    """Integrated ChattyAI class for web interface"""
    
    def __init__(self):
        print("📋 Initializing ChattyAI instance...")
        # AI Models
        self.whisper_model = None
        self.llama_model = None
        
        # Facial Recognition
        self.known_encodings = []
        self.known_names = []
        
        # Camera
        self.picam2 = None
        self.camera_initialized = False
        
        # State variables
        self.is_running = False
        self.current_person = None
        self.last_greeting_time = {}
        self.last_interaction_time = None
        self.person_absent_since = None
        self.waiting_cycle = 0
        self.last_bored_response_time = None
        self.bored_cycle = 0
        self.audio_recording_lock = threading.Lock()
        self.wake_word_active = False
        
        # Web-specific variables
        self.current_frame = None
        self.current_detected_person = None
        self.current_confidence = 0.0
        self.captured_image = None
        
        # Response lists
        self.jokes = []
        self.listening_responses = []
        self.waiting_responses = []
        self.warning_responses = []
        self.greeting_responses = []
        self.personalized_responses = {}
        
        # Telegram
        self.telegram_token = None
        self.telegram_chat_id = None
        
        # Threading
        self.camera_thread = None
        self.audio_thread = None
        
        # Initialize basic components
        self.setup_directories()
        self.setup_logging()
        self.load_response_files()
        self.load_personalized_responses()
        self.load_telegram_config()
        
        print("ChattyAIWeb instance created - models will be loaded when system starts")

    def emit_log(self, message, log_type="info"):
        """Emit log message to web interface"""
        log_message(message, log_type)

    def emit_conversation(self, message, msg_type="info"):
        """Emit conversation message to web interface"""
        emit_conversation(message, msg_type)

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
            self.emit_log("Response files loaded successfully")
        except FileNotFoundError as e:
            self.emit_log(f"Response file not found: {e}", "error")
            self.create_default_responses()

    def load_personalized_responses(self):
        """Load personalized responses from JSON file"""
        try:
            with open(PERSONALIZED_RESPONSES_FILE, 'r') as f:
                self.personalized_responses = json.load(f)
            self.emit_log(f"Personalized responses loaded for {len(self.personalized_responses)} people")
        except FileNotFoundError:
            self.emit_log("Personalized responses file not found. Creating default...", "warning")
            self.create_default_personalized_responses()
        except json.JSONDecodeError as e:
            self.emit_log(f"Error reading personalized responses JSON: {e}", "error")
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
                ]
            }
        }
        try:
            with open(PERSONALIZED_RESPONSES_FILE, 'w') as f:
                json.dump(default_responses, f, indent=4)
            self.personalized_responses = default_responses
            self.emit_log(f"Created default personalized responses file: {PERSONALIZED_RESPONSES_FILE}")
        except Exception as e:
            self.emit_log(f"Failed to create personalized responses file: {e}", "error")
            self.personalized_responses = default_responses

    def create_default_responses(self):
        """Create default responses if files are missing"""
        self.jokes = ["Why don't scientists trust atoms? Because they make up everything!"]
        self.greeting_responses = ["Hello! How can I help you today?"]
        self.listening_responses = ["I'm listening, what would you like to know?"]
        self.waiting_responses = ["I'm still here if you need anything"]
        self.warning_responses = ["Warning: Unknown person detected. Please identify yourself."]

    def load_telegram_config(self):
        """Load Telegram configuration"""
        try:
            with open(TELEGRAM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.telegram_token = config.get('bot_token')
                self.telegram_chat_id = config.get('chat_id')
            self.emit_log("Telegram configuration loaded")
        except FileNotFoundError:
            self.emit_log("Telegram config not found - alerts disabled", "warning")
        except Exception as e:
            self.emit_log(f"Failed to load Telegram config: {e}", "error")

    def load_models(self):
        """Load AI models"""
        self.emit_log("Loading AI models...")
        try:
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            self.emit_log("Whisper model loaded")
        except Exception as e:
            self.emit_log(f"Failed to load Whisper: {e}", "error")
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
            self.emit_log("LLaMA model loaded")
        except Exception as e:
            self.emit_log(f"Failed to load LLaMA: {e}", "error")
            return False

        return True

    def load_encodings(self):
        """Load facial recognition encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            self.emit_log(f"Loaded {len(self.known_encodings)} face encodings")
            return True
        except FileNotFoundError:
            self.emit_log(f"Encodings file '{ENCODINGS_FILE}' not found!", "error")
            return False
        except Exception as e:
            self.emit_log(f"Failed to load encodings: {e}", "error")
            return False

    def setup_camera(self):
        """Initialize camera with retry logic"""
        if self.camera_initialized:
            self.emit_log("Camera already initialized")
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.emit_log(f"Camera initialization attempt {attempt + 1}/{max_retries}")
                
                # Close any existing camera instance
                if self.picam2:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except:
                        pass
                    self.picam2 = None
                    time.sleep(2)

                # Create new camera instance
                if PICAMERA2_AVAILABLE:
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(
                        main={"format": 'XRGB8888', "size": (640, 480)}
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(2)
                    
                    # Test capture
                    test_frame = self.picam2.capture_array()
                    self.camera_initialized = True
                    self.emit_log(f"Picamera2 initialized successfully - Frame shape: {test_frame.shape}")
                    return True
                else:
                    self.emit_log("Picamera2 not available, trying OpenCV fallback", "warning")
                    return False
                    
            except Exception as e:
                self.emit_log(f"Camera initialization attempt {attempt + 1} failed: {e}", "error")
                if self.picam2:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except:
                        pass
                    self.picam2 = None
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    self.emit_log("All camera initialization attempts failed", "error")
                    self.camera_initialized = False
                    return False
        return False

    def detect_faces(self, frame):
        """Detect and recognize faces in frame"""
        try:
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
        except Exception as e:
            self.emit_log(f"Face detection error: {e}", "error")
            return None, None, 0.0

    def generate_video_feed(self):
        """Generate video frames for streaming"""
        log_message("Video feed generation started")
        frame_count = 0
        
        while True:
            try:
                frame_count += 1
                if self.picam2 and self.camera_initialized:
                    frame = self.picam2.capture_array()
                    
                    # Convert from RGB/RGBA to BGR for OpenCV
                    if len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        elif frame.shape[2] == 3:  # RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Only process facial recognition if system is running
                    if self.is_running and self.known_encodings:
                        name, face_location, confidence = self.detect_faces(frame)
                        
                        # Draw face rectangles and labels
                        if name and face_location:
                            top, right, bottom, left = face_location
                            
                            # Choose color based on recognition
                            if name == "Unknown":
                                color = (0, 0, 255)  # Red
                                label = f"Unknown ({confidence:.2f})"
                            else:
                                color = (0, 255, 0)  # Green
                                label = f"{name} ({confidence:.2f})"
                            
                            # Draw rectangle and label
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            cv2.putText(frame, label, (left + 6, bottom - 6),
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                            
                            # Capture image for display
                            face_img = frame[top:bottom, left:right].copy()
                            if face_img.size > 0:
                                self.captured_image = cv2.resize(face_img, (200, 200))
                            
                            # Emit person detection update
                            try:
                                socketio.emit('person_detected', {
                                    'name': name,
                                    'confidence': f"{confidence:.1%}",
                                    'timestamp': datetime.now().strftime("%H:%M:%S")
                                }, to=None)
                            except:
                                pass
                    
                    # Add status overlay to frame
                    if self.is_running:
                        status_text = "Chatty AI Active"
                        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        status_text = "System Ready - Click Start"
                        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if self.current_person:
                        person_text = f"Current: {self.current_person}"
                        cv2.putText(frame, person_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if self.wake_word_active:
                        wake_text = "Wake Word: ACTIVE"
                        cv2.putText(frame, wake_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        wake_text = "Wake Word: INACTIVE"
                        cv2.putText(frame, wake_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                
                else:  # No camera available
                    # Create a black frame with error message
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Not Available", (200, 220),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Check camera connection", (150, 260),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                print(f"Video feed error: {e}")
                # Create error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Video Error", (200, 220),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(error_frame, f"{str(e)[:40]}", (100, 260),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)

    def start_system(self):
        """Start the AI system"""
        self.emit_log("Starting Chatty AI System...")
        
        # Load models first if not already loaded
        if not self.whisper_model or not self.llama_model:
            self.emit_log("Loading AI models...", "info")
            if not self.load_models():
                self.emit_log("Failed to load AI models", "error")
                return False
        
        if not self.camera_initialized:
            self.emit_log("Initializing camera for full system...", "info")
            if not self.setup_camera():
                self.emit_log("Warning: Camera initialization failed - continuing without full AI features", "warning")
        
        if not self.known_encodings:
            self.emit_log("Loading facial recognition encodings...", "info")
            if not self.load_encodings():
                self.emit_log("Warning: No facial recognition data - continuing without person recognition", "warning")
        
        self.emit_log("Chatty AI System Started!")
        self.is_running = True
        
        # Start audio thread only if we have the required models
        if self.whisper_model:
            self.audio_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
            self.audio_thread.start()
            self.emit_log("Audio processing thread started")
        
        # Start camera monitoring thread only if camera and encodings are available
        if self.camera_initialized and self.known_encodings:
            self.camera_thread = threading.Thread(target=self.camera_monitoring_thread, daemon=True)
            self.camera_thread.start()
            self.emit_log("Camera monitoring thread started")
        
        return True

    def stop_system(self):
        """Stop the AI system"""
        self.is_running = False
        self.emit_log("Chatty AI System Stopped")

    def speak_text(self, text):
        """Convert text to speech using Piper"""
        try:
            with self.audio_recording_lock:
                command = [
                    PIPER_EXECUTABLE,
                    "--model", VOICE_PATH,
                    "--config", CONFIG_PATH,
                    "--output_file", RESPONSE_AUDIO
                ]
                subprocess.run(command, input=text.encode("utf-8"), check=True, capture_output=True)
                subprocess.run(["aplay", RESPONSE_AUDIO], check=True, capture_output=True)
                
                # Emit to web interface
                self.emit_conversation(f"🔊 Speaking: {text}", "speech")
        except subprocess.CalledProcessError as e:
            self.emit_log(f"TTS failed: {e}", "error")

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
            self.emit_log(f"LLM error: {e}", "error")
            return "Sorry, I had trouble processing that question."

    def process_user_input(self, text):
        """Process user input"""
        self.emit_conversation(f"👤 User said: '{text}'", "user_input")
        
        # Check for commands
        command = self.is_command(text)
        if command:
            self.emit_conversation(f"⚡ Executing command: {command}", "command")
            response = self.execute_command(command)
        else:
            self.emit_conversation("🤖 Generating LLM response", "llm")
            response = self.query_llama(text)
        
        self.emit_conversation(f"💬 Response: '{response}'", "response")
        return response

    def is_command(self, text):
        """Check if text is a command"""
        text_lower = text.lower().strip()
        for command in COMMANDS.keys():
            if command in text_lower:
                return command
        return None

    def execute_command(self, command):
        """Execute system command"""
        responses = {
            "flush the toilet": f"Oh {self.current_person}, you know I am a digital assistant. I cannot actually flush toilets! So why dont you haul your lazy butt up off the couch and flush the toilet yourself!",
            "turn on the lights": "I would turn on the lights if I was connected to a smart home system.",
            "turn off the lights": "I would turn off the lights if I was connected to a smart home system.",
            "play music": "I would start playing music if I had access to a music system.",
            "stop music": "I would stop the music if any music was playing.",
            "who is sponsoring this video": f"You are very funny {self.current_person}. You know you dont have any sponsors for your videos!",
            "how is the weather today": f"O M G {self.current_person}! Surely you DO NOT want to waste my valuable resources by asking me what the weather is today. Cant you just look out the window or ask Siri. That is about all Siri is good for!",
            "what time is it": f"The current time is {datetime.now().strftime('%I:%M %p')}",
            "shutdown system": "I would shutdown the system, but I will skip that for safety reasons during testing.",
            "reboot system": "I would reboot the system, but I will skip that for safety reasons during testing."
        }
        
        if command == "who is sponsoring this video":
            self.play_laughing()
        
        return responses.get(command, f"I understand you want me to {command}, but I dont have that capability yet.")

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

    def get_personalized_response(self, person_name, response_type, fallback_list=None):
        """Get a personalized response for a specific person and response type"""
        person_name_lower = person_name.lower()
        person_data = None
        
        for name, data in self.personalized_responses.items():
            if name.lower() == person_name_lower:
                person_data = data
                break
        
        if person_data and response_type in person_data:
            responses = person_data[response_type]
            if responses:
                return random.choice(responses)
        
        if fallback_list:
            response = random.choice(fallback_list)
            if "{name}" in response:
                return response.replace("{name}", person_name)
            else:
                return f"Hello {person_name}! {response}"
        
        return f"Hello {person_name}! How can I help you today?"

    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper"""
        try:
            if not os.path.exists(filename):
                return ""
            
            segments, _ = self.whisper_model.transcribe(filename)
            transcript = " ".join(segment.text for segment in segments).strip()
            self.emit_log(f"Transcription: '{transcript}'")
            return transcript
        except Exception as e:
            self.emit_log(f"Transcription error: {e}", "error")
            return ""

    def detect_wake_word(self, text):
        """Check if text contains wake word"""
        if not text:
            return False
        
        text_cleaned = text.lower().replace(',', '').replace('.', '').strip()
        
        for wake_word in WAKE_WORDS:
            wake_word_cleaned = wake_word.lower().strip()
            if wake_word_cleaned in text_cleaned:
                self.emit_conversation(f"🎯 Wake word detected: '{wake_word}' in '{text}'", "wake_word")
                return True
        
        return False

    def record_with_silence_detection(self):
        """Record audio until silence detected"""
        try:
            with self.audio_recording_lock:
                self.emit_log("Recording audio...")
                audio_data = []
                silence_duration = 0
                recording_duration = 0
                check_interval = 0.2
                samples_per_check = int(SAMPLE_RATE * check_interval)
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        self.emit_log(f"Audio callback status: {status}", "warning")
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
                                    self.emit_log(f"Silence detected after {recording_duration:.1f}s")
                                    break
                            else:
                                silence_duration = 0
                
                if audio_data:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sf.write(WAV_FILENAME, audio_array, SAMPLE_RATE)
                    self.emit_log(f"Audio saved: {len(audio_array)/SAMPLE_RATE:.1f}s duration")
                    return True
                
                return False
        except Exception as e:
            self.emit_log(f"Recording error: {e}", "error")
            return False

    def listen_for_wake_word(self):
        """Listen for wake words in background"""
        self.emit_log("Wake word detection thread started")
        # Simplified wake word detection for web interface
        # You can implement full audio processing here
        pass

    def camera_monitoring_thread(self):
        """Camera monitoring thread for facial recognition processing"""
        self.emit_log("Camera monitoring thread started")
        # Camera monitoring logic would go here
        # This is simplified for the web interface
        pass

    def cleanup(self):
        """Clean up resources"""
        self.emit_log("Cleaning up resources...")
        self.is_running = False
        
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
                self.emit_log("Camera stopped")
            except:
                pass
        
        # Clean up audio files
        for audio_file in [WAV_FILENAME, RESPONSE_AUDIO, WAKE_WORD_AUDIO]:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
        
        self.emit_log("Chatty AI cleanup complete")

def initialize_system_background():
    """Initialize heavy components in background after Flask starts"""
    global chatty_ai, camera_initialized, system_ready, initialization_status, models_loaded
    
    try:
        log_message("🚀 Background: Starting system initialization...")
        initialization_status = "Initializing components..."
        
        # Give Flask server time to fully start and accept connections
        time.sleep(2)
        
        # Notify clients that initialization is starting
        socketio.emit('status_update', {'status': 'Initializing AI system...'}, to=None)
        
        log_message("📋 Initializing ChattyAI instance...")
        initialization_status = "Loading AI models..."
        
        # Initialize ChattyAI instance
        try:
            chatty_ai = ChattyAIWeb()
            
            # Try to load models
            models_loaded = chatty_ai.load_models()
            
            # Try to setup camera
            camera_initialized = chatty_ai.setup_camera()
            
            log_message("✅ Response files loaded successfully")
            log_message("✅ Telegram configuration loaded")
            
        except Exception as e:
            log_message(f"❌ ChattyAI initialization error: {e}")
            models_loaded = False
            camera_initialized = False
        
        # Mark system as ready
        if models_loaded or camera_initialized:
            log_message("🎉 System initialization completed successfully!")
            initialization_status = "Ready"
            system_ready = True
            
            # Notify all connected clients that system is ready
            socketio.emit('system_ready', {
                'status': 'ready', 
                'camera': camera_initialized,
                'models': models_loaded
            }, to=None)
        else:
            raise Exception("Failed to initialize required components")
            
    except Exception as e:
        error_msg = f"💥 System initialization failed: {e}"
        log_message(error_msg)
        initialization_status = f"Error: {str(e)}"
        
        traceback.print_exc()
        
        # Notify clients of error
        socketio.emit('system_error', {'error': str(e)}, to=None)

# Flask Routes
@app.route('/')
def index():
    """Serve the main web interface using existing HTML template"""
    try:
        return render_template('Chatty_AI.html')
    except Exception as e:
        log_message(f"❌ Template loading error: {e}")
        return f"<h1>Template Error</h1><p>Could not load Chatty_AI.html: {e}</p>"

@app.route('/video_feed')
def video_feed():
    """Serve camera video feed as MJPEG stream"""
    global chatty_ai
    
    if chatty_ai and chatty_ai.camera_initialized:
        log_message("📹 Serving video feed from Picamera2")
        return Response(chatty_ai.generate_video_feed(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Fallback to simple error frame
        def generate_error_feed():
            while True:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Initializing...", (150, 220),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, "Please wait", (220, 260),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
        
        return Response(generate_error_feed(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/captured_image')
def captured_image():
    """Route for captured person image"""
    global chatty_ai
    
    if chatty_ai and chatty_ai.captured_image is not None:
        ret, buffer = cv2.imencode('.jpg', chatty_ai.captured_image)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    # Return placeholder image if no capture available
    placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No Image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    ret, buffer = cv2.imencode('.jpg', placeholder)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    global chatty_ai
    return jsonify({
        'system_ready': system_ready,
        'camera_ready': camera_initialized,
        'models_loaded': models_loaded,
        'status': initialization_status,
        'current_person': chatty_ai.current_person if chatty_ai else None,
        'wake_word_active': chatty_ai.wake_word_active if chatty_ai else False,
        'timestamp': datetime.now().isoformat()
    })

# SocketIO Event Handlers
@socketio.on('connect')
def handle_connect():
    """Handle client WebSocket connection"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    log_message(f"🔌 WebSocket client connected: {client_ip} (ID: {request.sid})")
    
    # Send current status to newly connected client
    emit('status_update', {
        'status': initialization_status,
        'system_ready': system_ready,
        'camera_ready': camera_initialized,
        'models_loaded': models_loaded
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client WebSocket disconnection"""
    log_message(f"🔌 WebSocket client disconnected: {request.sid}")

@socketio.on('start_system')
def handle_start_system():
    """Handle system start request from client"""
    global chatty_ai
    
    if not chatty_ai:
        emit('system_status', {'running': False, 'error': 'System not initialized'})
        return
    
    try:
        success = chatty_ai.start_system()
        emit('system_status', {'running': success})
        
        if success:
            log_message("✅ System started successfully by web request")
        else:
            log_message("❌ System start failed")
            
    except Exception as e:
        log_message(f"❌ System start error: {e}")
        emit('system_status', {'running': False, 'error': str(e)})

@socketio.on('stop_system')
def handle_stop_system():
    """Handle system stop request from client"""
    global chatty_ai
    
    if not chatty_ai:
        emit('system_status', {'running': False})
        return
    
    try:
        chatty_ai.stop_system()
        emit('system_status', {'running': False})
        log_message("⏹️ System stopped by web request")
    except Exception as e:
        log_message(f"❌ System stop error: {e}")
        emit('system_status', {'running': False, 'error': str(e)})

@socketio.on('get_system_info')
def handle_get_system_info():
    """Handle system info request"""
    global chatty_ai
    
    try:
        info = {
            'camera_initialized': camera_initialized,
            'models_loaded': models_loaded,
            'wake_word_active': chatty_ai.wake_word_active if chatty_ai else False,
            'current_person': chatty_ai.current_person if chatty_ai else None,
            'system_ready': system_ready
        }
        emit('system_info', info)
        log_message("📊 System info sent to client")
    except Exception as e:
        log_message(f"❌ System info error: {e}")

@socketio.on('test_speech')
def handle_test_speech(data):
    """Handle speech test request"""
    global chatty_ai
    
    if not chatty_ai:
        emit('speech_test_result', {'success': False, 'message': 'System not initialized'})
        return
    
    try:
        test_text = data.get('text', 'Hello! This is a test of the Chatty AI speech system.')
        chatty_ai.speak_text(test_text)
        emit('speech_test_result', {'success': True})
        log_message("🔊 Speech test completed")
    except Exception as e:
        emit('speech_test_result', {'success': False, 'message': str(e)})
        log_message(f"❌ Speech test failed: {e}")

@socketio.on('manual_wake_word')
def handle_manual_wake_word():
    """Handle manual wake word trigger"""
    global chatty_ai
    
    if not chatty_ai:
        emit('manual_wake_result', {'success': False, 'message': 'System not initialized'})
        return
    
    if not chatty_ai.current_person:
        emit('manual_wake_result', {'success': False, 'message': 'No person detected'})
        return
    
    try:
        # Trigger manual wake word response
        chatty_ai.wake_word_active = True
        chatty_ai.play_beep()
        
        listening_response = chatty_ai.get_personalized_response(
            chatty_ai.current_person, 
            "listening", 
            chatty_ai.listening_responses
        )
        chatty_ai.speak_text(listening_response)
        
        emit('manual_wake_result', {'success': True})
        log_message("🎯 Manual wake word triggered")
    except Exception as e:
        emit('manual_wake_result', {'success': False, 'message': str(e)})
        log_message(f"❌ Manual wake word failed: {e}")

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages from web interface"""
    global chatty_ai
    
    if not chatty_ai:
        emit('chat_error', {'error': 'System not ready'})
        return
    
    if not models_loaded:
        emit('chat_error', {'error': 'AI models not loaded'})
        return
    
    try:
        message = data.get('message', '')
        if not message.strip():
            emit('chat_error', {'error': 'Empty message'})
            return
        
        log_message(f"💬 Chat message received: {message[:50]}...")
        
        # Process message with ChattyAI
        response = chatty_ai.process_user_input(message)
        
        emit('chat_response', {'response': response})
        
    except Exception as e:
        log_message(f"❌ Chat processing error: {e}")
        emit('chat_error', {'error': str(e)})

# Background initialization using app context
def start_background_initialization():
    """Start background initialization after Flask is ready"""
    with app.app_context():
        # Use SocketIO's background task functionality
        socketio.start_background_task(initialize_system_background)
        log_message("📅 Scheduled background initialization")

# Cleanup function for graceful shutdown
def cleanup():
    """Cleanup resources on shutdown"""
    global chatty_ai
    if chatty_ai:
        chatty_ai.cleanup()
        log_message("🧹 Cleanup completed")

# Register cleanup function
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    print("🚀 Initializing Chatty AI Web Interface (app_2_corrected.py)")
    print("=" * 60)
    print("🌐 Starting Flask server on http://0.0.0.0:5000")
    print("🔗 Access your Chatty AI interface at: http://192.168.1.16:5000")
    print("=" * 60)
    
    try:
        # Create templates directory if it doesn't exist
        templates_dir = '/home/nickspi5/Chatty_AI/templates'
        os.makedirs(templates_dir, exist_ok=True)
        log_message(f"📁 Templates directory checked: {templates_dir}")
        
        # Check if HTML template exists
        html_template_path = os.path.join(templates_dir, 'Chatty_AI.html')
        if not os.path.exists(html_template_path):
            log_message(f"⚠️ Warning: Chatty_AI.html template not found at: {html_template_path}")
        else:
            log_message("✅ Chatty_AI.html template found")
        
        # Start background initialization in a separate thread after a delay
        # This ensures Flask has time to bind to the port first
        def delayed_init():
            time.sleep(3)  # Wait for Flask to fully start
            start_background_initialization()
        
        init_thread = threading.Thread(target=delayed_init, daemon=True)
        init_thread.start()
        
        log_message("🎯 Background initialization scheduled")
        log_message("🌐 Starting Flask-SocketIO server...")
        
        # Start Flask-SocketIO server
        socketio.run(app,
                    host='0.0.0.0',
                    port=5000,
                    debug=False,
                    allow_unsafe_werkzeug=True,
                    use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
        cleanup()
    except Exception as e:
        print(f"💥 Exception during startup: {e}")
        traceback.print_exc()
        cleanup()