#!/usr/bin/env python3
"""
Fast Chatty AI Web Version - Optimized for Speed
Based on the original terminal version with Flask web interface
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
import psutil
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from llama_cpp import Llama
from picamera2 import Picamera2
from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import base64
import io

# -------------------------------
# Configuration - Same as original
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
BORED_RESPONSES_FILE = "bored_responses.txt"
VISITOR_GREETING_RESPONSES_FILE = "visitor_greeting_responses.txt"

# Wake word phrases - Same as original
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

# Command keywords - Same as original
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

# Audio parameters - Same as original
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.035
MIN_SILENCE_DURATION = 1.5
MAX_RECORDING_DURATION = 30

# Timing parameters - Same as original
GREETING_COOLDOWN = 300  # 5 minutes
BORED_RESPONSE_INTERVAL = 30
PERSON_DETECTION_INTERVAL = 0.5
WAKE_WORD_CHECK_INTERVAL = 1.0

# Flask Configuration
app = Flask(__name__, template_folder='/home/nickspi5/Chatty_AI/templates')
app.config['SECRET_KEY'] = 'chatty_ai_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class FastChattyAI:
    def __init__(self):
        """Initialize with optimized approach - preload everything"""
        print("Initializing Fast Chatty AI...")
        
        # Core state - Same as original
        self.is_running = False
        self.current_person = None
        self.last_greeting_time = {}
        self.last_interaction_time = None
        self.person_absent_since = None
        self.last_bored_response_time = None
        self.bored_cycle = 0
        self.audio_recording_lock = threading.Lock()
        self.wake_word_active = False
        
        # Response lists
        self.jokes = []
        self.listening_responses = []
        self.waiting_responses = []
        self.warning_responses = []
        self.greeting_responses = []
        self.bored_responses = []
        self.visitor_greeting_responses = []
        
        # Telegram
        self.telegram_token = None
        self.telegram_chat_id = None
        
        # Threading
        self.camera_thread = None
        self.audio_thread = None
        
        # Models - Will be preloaded
        self.whisper_model = None
        self.llama_model = None
        
        # Facial Recognition
        self.known_encodings = []
        self.known_names = []
        
        # Camera
        self.picam2 = None
        
        # Web-specific
        self.latest_frame = None
        self.latest_person_image = None
        self.frame_lock = threading.Lock()
        
        # System monitoring
        self.system_info = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'cpu_temp': 0,
            'gpu_temp': 0
        }
        
        # Initialize everything immediately
        self.setup_directories()
        self.load_response_files()
        self.setup_logging()
        self.load_encodings()
        self.load_telegram_config()
        self.setup_camera()
        
        # CRITICAL: Preload models immediately for speed
        print("Preloading AI models for optimal performance...")
        self.preload_models()
        
        print("Fast Chatty AI initialized successfully!")
    
    def emit_log(self, message, level='info'):
        """Emit log message to web interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('log_update', {
            'message': message,
            'type': level,
            'timestamp': timestamp
        })
        
        # Also log to console for debugging
        print(f"[{timestamp}] {message}")
    
    def emit_conversation(self, message, level='info'):
        """Emit conversation message to web interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        socketio.emit('conversation_update', {
            'message': message,
            'type': level,
            'timestamp': timestamp
        })
    
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
        """Load response text files - Same as original"""
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
            
            # Load bored responses
            try:
                with open(BORED_RESPONSES_FILE, 'r') as f:
                    self.bored_responses = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                self.create_default_bored_responses()
            
            # Load visitor greeting responses
            try:
                with open(VISITOR_GREETING_RESPONSES_FILE, 'r') as f:
                    self.visitor_greeting_responses = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                self.create_default_visitor_responses()
                
            print("Response files loaded successfully")
        except FileNotFoundError as e:
            print(f"Response file not found: {e}")
            self.create_default_responses()
    
    def create_default_bored_responses(self):
        """Create default bored responses file"""
        default_bored = [
            "I'm getting a bit bored waiting here",
            "Still hanging around here waiting for you, dude",
            "I'm patiently waiting for your commands",
            "I am feeling restless waiting here",
            "Still here waiting to help you"
        ]
        
        try:
            with open(BORED_RESPONSES_FILE, 'w') as f:
                for response in default_bored:
                    f.write(response + '\n')
            self.bored_responses = default_bored
        except Exception as e:
            self.bored_responses = default_bored
    
    def create_default_visitor_responses(self):
        """Create default visitor greeting responses file"""
        default_visitor = [
            "Hello. I do not recognize you. Can I be of assistance?",
            "Good day, visitor. How may I help you today?",
            "Welcome. I do not believe we have met before. What can I do for you?",
            "Hello there. I am an AI assistant. How can I assist you?",
            "It is great to meet you. I do not recognize your face, but I am happy to help."
        ]
        
        try:
            with open(VISITOR_GREETING_RESPONSES_FILE, 'w') as f:
                for response in default_visitor:
                    f.write(response + '\n')
            self.visitor_greeting_responses = default_visitor
        except Exception as e:
            self.visitor_greeting_responses = default_visitor
    
    def create_default_responses(self):
        """Create default responses if files are missing"""
        self.jokes = ["Why don't scientists trust atoms? Because they make up everything!"]
        self.greeting_responses = ["Hey {name}! Good to see you, buddy! What's up?"]
        self.listening_responses = ["Yes {name}, I'm listening. What would you like to know?"]
        self.waiting_responses = ["I am still around if you need me, {name}"]
        self.warning_responses = ["Attention unauthorized person, you are not authorized to access this property. Leave immediately."]
        self.bored_responses = ["I'm getting a bit bored waiting here"]
        self.visitor_greeting_responses = ["Hello. I do not recognize you. Can I be of assistance?"]
    
    def preload_models(self):
        """CRITICAL: Preload models for maximum speed - Same as original approach"""
        try:
            # Load Whisper model
            self.emit_log("Loading Whisper model...", 'info')
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            self.emit_log("Whisper model loaded successfully", 'success')
            
            # Load LLaMA model
            self.emit_log("Loading LLaMA model...", 'info')
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=2048,
                temperature=0.7,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                verbose=False
            )
            self.emit_log("LLaMA model loaded successfully", 'success')
            
            return True
            
        except Exception as e:
            self.emit_log(f"Failed to load models: {e}", 'error')
            return False
    
    def load_encodings(self):
        """Load facial recognition encodings"""
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            self.emit_log(f"Loaded {len(self.known_encodings)} face encodings", 'success')
            return True
        except FileNotFoundError:
            self.emit_log(f"Encodings file '{ENCODINGS_FILE}' not found!", 'error')
            return False
        except Exception as e:
            self.emit_log(f"Failed to load encodings: {e}", 'error')
            return False
    
    def load_telegram_config(self):
        """Load Telegram configuration"""
        try:
            with open(TELEGRAM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            self.telegram_token = config.get('bot_token')
            self.telegram_chat_id = config.get('chat_id')
            self.emit_log("Telegram configuration loaded", 'success')
        except FileNotFoundError:
            self.emit_log("Telegram config not found - alerts disabled", 'warning')
        except Exception as e:
            self.emit_log(f"Failed to load Telegram config: {e}", 'error')
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            ))
            self.picam2.start()
            time.sleep(2)  # Camera warm-up
            self.emit_log("Camera initialized successfully", 'success')
            return True
        except Exception as e:
            self.emit_log(f"Failed to initialize camera: {e}", 'error')
            return False
    
    def speak_text(self, text):
        """Convert text to speech using Piper - Same as original"""
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
                self.emit_log(f"Speaking: '{text[:50]}...'", 'info')
        except subprocess.CalledProcessError as e:
            self.emit_log(f"TTS failed: {e}", 'error')
    
    def play_beep(self):
        """Play beep sound - Same as original"""
        try:
            with self.audio_recording_lock:
                subprocess.run(["aplay", BEEP_SOUND], check=True, capture_output=True)
                self.emit_log("Beep sound played", 'debug')
        except subprocess.CalledProcessError:
            pass
    
    def detect_faces(self, frame):
        """Detect and recognize faces in frame - Same as original"""
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
    
    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper - Same as original but with web logging"""
        try:
            if not os.path.exists(filename):
                return ""
            
            segments, _ = self.whisper_model.transcribe(filename)
            transcript = " ".join(segment.text for segment in segments).strip()
            self.emit_log(f"Transcription: '{transcript}'", 'info')
            return transcript
        except Exception as e:
            self.emit_log(f"Transcription error: {e}", 'error')
            return ""
    
    def detect_wake_word(self, text):
        """Check if text contains wake word - Same as original"""
        if not text:
            return False
        
        text_cleaned = text.lower().replace(',', '').replace('.', '').strip()
        
        for wake_word in WAKE_WORDS:
            wake_word_cleaned = wake_word.lower().strip()
            if wake_word_cleaned in text_cleaned:
                self.emit_log(f"Wake word detected: '{wake_word}' in '{text}'", 'success')
                return True
        
        self.emit_log(f"No wake word found in: '{text_cleaned}'", 'debug')
        return False
    
    def record_with_silence_detection(self):
        """Record audio until silence detected - Same as original with better sample rate handling"""
        try:
            # Auto-detect working sample rate
            working_sample_rate = None
            for rate in [44100, 48000, 22050, 16000]:
                try:
                    sd.check_input_settings(channels=1, samplerate=rate, dtype='float32')
                    working_sample_rate = rate
                    break
                except:
                    continue
            
            if not working_sample_rate:
                self.emit_log("No compatible sample rate found", 'error')
                return False
            
            with self.audio_recording_lock:
                self.emit_log(f"Recording with sample rate: {working_sample_rate} Hz", 'debug')
                audio_data = []
                silence_duration = 0
                recording_duration = 0
                check_interval = 0.1
                samples_per_check = int(working_sample_rate * check_interval)
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        self.emit_log(f"Audio status: {status}", 'warning')
                    audio_data.extend(indata[:, 0])
                
                with sd.InputStream(callback=audio_callback, 
                                  samplerate=working_sample_rate, 
                                  channels=1,
                                  dtype='float32'):
                    
                    while recording_duration < 30:  # Max 30 seconds
                        time.sleep(check_interval)
                        recording_duration += check_interval
                        
                        if len(audio_data) >= samples_per_check:
                            recent_audio = np.array(audio_data[-samples_per_check:])
                            rms = np.sqrt(np.mean(recent_audio**2))
                            
                            # Debug every second
                            if int(recording_duration * 10) % 10 == 0:
                                self.emit_log(f"Recording: {recording_duration:.1f}s | Audio: {rms:.4f} | Silence: {silence_duration:.1f}s", 'debug')
                            
                            if rms < SILENCE_THRESHOLD:
                                silence_duration += check_interval
                                if silence_duration >= MIN_SILENCE_DURATION:
                                    self.emit_log(f"Silence detected! Recorded {recording_duration:.1f}s", 'info')
                                    break
                            else:
                                if silence_duration > 0.5:
                                    self.emit_log(f"Speech resumed after {silence_duration:.1f}s silence", 'debug')
                                silence_duration = 0
                
                # Save audio (convert to 16kHz for Whisper if needed)
                if len(audio_data) > 0:
                    audio_array = np.array(audio_data)
                    
                    # Convert to 16kHz for Whisper if we recorded at different rate
                    if working_sample_rate != 16000:
                        from scipy import signal
                        target_length = int(len(audio_array) * 16000 / working_sample_rate)
                        audio_array = signal.resample(audio_array, target_length)
                        save_sample_rate = 16000
                    else:
                        save_sample_rate = working_sample_rate
                        
                    sf.write(WAV_FILENAME, audio_array, save_sample_rate)
                    self.emit_log(f"Audio saved: {len(audio_array)/save_sample_rate:.1f}s", 'info')
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.emit_log(f"Recording error: {e}", 'error')
            return False
    
    def record_wake_word_check(self):
        """Record short audio clip for wake word detection - Same as original"""
        try:
            if not self.audio_recording_lock.acquire(blocking=False):
                return False
            
            try:
                # Record 5 seconds for wake word detection
                audio_data = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                
                # Check if audio contains sound above threshold
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > SILENCE_THRESHOLD * 2:  # Higher threshold for wake word
                    sf.write(WAKE_WORD_AUDIO, audio_data, SAMPLE_RATE)
                    self.emit_log(f"Wake word audio saved, RMS: {rms:.4f}", 'debug')
                    return True
                else:
                    self.emit_log(f"Audio too quiet for wake word, RMS: {rms:.4f}", 'debug')
                    return False
                    
            finally:
                self.audio_recording_lock.release()
                
        except Exception as e:
            self.emit_log(f"Wake word recording error: {e}", 'error')
            return False
    
    def process_user_input(self, text):
        """Process user input - Same as original"""
        self.emit_log(f"Processing user input: '{text}'", 'info')
        command = self.is_command(text)
        if command:
            self.emit_log(f"Executing command: {command}", 'info')
            response = self.execute_command(command)
        else:
            self.emit_log("Generating LLM response", 'info')
            response = self.query_llama(text)
        
        return response
    
    def is_command(self, text):
        """Check if text is a command"""
        text_lower = text.lower().strip()
        for command in COMMANDS.keys():
            if command in text_lower:
                return command
        return None
    
    def execute_command(self, command):
        """Execute system command - Same as original"""
        if command == "flush the toilet":
            response = f"Oh {self.current_person}, you know I am a digital assistant. I cannot actually flush toilets!"
        elif command == "what time is it":
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
        else:
            response = f"I understand you want me to {command}, but I dont have that capability yet."
        
        return response
    
    def query_llama(self, prompt):
        """Generate LLM response - Same as original"""
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
            self.emit_log(f"LLM error: {e}", 'error')
            return "Sorry, I had trouble processing that question."
    
    def greet_person(self, name):
        """Greet a detected person - Same as original"""
        current_time = time.time()
        
        # Check if we should greet this person (cooldown check)
        if name in self.last_greeting_time:
            time_since_last = current_time - self.last_greeting_time[name]
            if time_since_last < GREETING_COOLDOWN:
                return False
        
        # Use generic greeting with person's name
        if self.greeting_responses:
            greeting_template = random.choice(self.greeting_responses)
            greeting = greeting_template.replace("{name}", name)
        else:
            greeting = f"Hey {name}! Good to see you, buddy! What's up?"
        
        self.speak_text(greeting)
        self.last_greeting_time[name] = current_time
        self.last_interaction_time = current_time
        
        # Enable wake word detection after greeting
        self.wake_word_active = True
        self.last_bored_response_time = current_time
        self.bored_cycle = 0
        
        self.emit_log(f"Greeted {name} - Wake word detection now active", 'success')
        return True
    
    def listen_for_wake_word(self):
        """Listen for wake words in background - Same as original"""
        self.emit_log("Wake word detection thread started", 'success')
        
        while self.is_running:
            try:
                # Only listen if someone is present and wake word detection is active
                if self.current_person and self.current_person != "Unknown" and self.wake_word_active:
                    
                    # Record audio for wake word detection
                    if self.record_wake_word_check():
                        # Transcribe and check for wake word
                        transcript = self.transcribe_audio(WAKE_WORD_AUDIO)
                        
                        if transcript and self.detect_wake_word(transcript):
                            self.emit_log("WAKE WORD DETECTED! Starting conversation...", 'success')
                            self.emit_conversation(f"Wake word detected: {transcript}", 'wake_word')
                            self.play_beep()
                            
                            # Speak listening response
                            if self.listening_responses:
                                listening_template = random.choice(self.listening_responses)
                                listening_response = listening_template.replace("{name}", self.current_person)
                            else:
                                listening_response = f"Yes {self.current_person}, I'm listening. What would you like to know?"
                            
                            self.speak_text(listening_response)
                            
                            # Record full request
                            self.emit_log("Please speak your request...", 'info')
                            if self.record_with_silence_detection():
                                user_text = self.transcribe_audio(WAV_FILENAME)
                                if user_text and len(user_text.strip()) > 2:
                                    self.emit_log(f"User said: '{user_text}'", 'info')
                                    self.emit_conversation(f"User said: {user_text}", 'user_input')
                                    response = self.process_user_input(user_text)
                                    self.emit_log(f"Response: '{response}'", 'info')
                                    self.emit_conversation(f"Response: {response}", 'response')
                                    self.speak_text(response)
                                    self.last_interaction_time = time.time()
                                    self.last_bored_response_time = time.time()
                                else:
                                    self.speak_text("I didn't catch that. Could you repeat your request?")
                                    self.emit_conversation("No clear speech detected", 'info')
                            else:
                                self.speak_text("I'm having trouble hearing you. Please try again.")
                                self.emit_conversation("Failed to record audio", 'info')
                    
                    time.sleep(WAKE_WORD_CHECK_INTERVAL)
                else:
                    # No one present or wake word not active, sleep longer
                    time.sleep(2.0)
                    
            except Exception as e:
                self.emit_log(f"Wake word detection error: {e}", 'error')
                time.sleep(2.0)
        
        self.emit_log("Wake word detection thread stopped", 'info')
    
    def camera_monitoring_loop(self):
        """Main camera monitoring loop - optimized for web"""
        while self.is_running:
            try:
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV if needed
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Store latest frame for web interface
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Process facial recognition
                name, face_location, confidence = self.detect_faces(frame)
                current_time = time.time()
                
                # Process facial recognition logic - Same as original
                if name and face_location:
                    # Person detected
                    if name != self.current_person:
                        # New person or person changed
                        self.current_person = name
                        self.person_absent_since = None
                        self.wake_word_active = False  # Reset wake word state
                        
                        # Update web interface
                        socketio.emit('person_detected', {
                            'name': name,
                            'confidence': f"{confidence:.1%}"
                        })
                        
                        if name == "Unknown":
                            self.handle_unknown_person(frame, confidence)
                        else:
                            # Save photo and send telegram alert for known person
                            photo_path = self.save_security_photo(frame, name, confidence)
                            self.send_telegram_alert(name, confidence, photo_path)
                            
                            # Store person image for web interface
                            with self.frame_lock:
                                self.latest_person_image = frame.copy()
                            
                            # Greet known person (this will activate wake word detection)
                            self.greet_person(name)
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
                            self.wake_word_active = False
                            self.last_bored_response_time = None
                            self.bored_cycle = 0
                            
                            # Update web interface
                            socketio.emit('person_detected', {
                                'name': 'No person detected',
                                'confidence': '--'
                            })
                            
                            self.emit_log("Person left - resetting state", 'info')
                
                time.sleep(PERSON_DETECTION_INTERVAL)
                
            except Exception as e:
                self.emit_log(f"Camera loop error: {e}", 'error')
                time.sleep(1)
    
    def save_security_photo(self, frame, person_name, confidence):
        """Save security photo with timestamp - Same as original"""
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
        """Send Telegram alert - Same as original"""
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
            self.emit_log(f"Telegram alert failed: {e}", 'error')
            return False
    
    def handle_unknown_person(self, frame, confidence):
        """Handle unknown person detection - Same as original"""
        if self.is_daytime_hours():
            # 6:00AM - 12:00PM: Assume visitor, be friendly
            if self.visitor_greeting_responses:
                visitor_greeting = random.choice(self.visitor_greeting_responses)
            else:
                visitor_greeting = "Hello. I do not recognize you. Can I be of assistance?"
            self.speak_text(visitor_greeting)
            self.emit_log("Unknown person detected during daytime - treated as visitor", 'info')
        else:
            # 12:01PM - 5:59AM: Assume intruder, give warning
            if self.warning_responses:
                warning = random.choice(self.warning_responses)
            else:
                warning = "Attention unauthorized person, you are not authorized to access this property. Leave immediately."
            self.speak_text(warning)
            self.emit_log("Unknown person detected during nighttime/evening - treated as intruder", 'warning')
        
        photo_path = self.save_security_photo(frame, "Unknown", confidence)
        self.send_telegram_alert("Unknown", confidence, photo_path)
    
    def is_daytime_hours(self):
        """Check if current time is between 6:00AM and 12:00PM"""
        current_hour = datetime.now().hour
        return 6 <= current_hour <= 12
    
    def get_system_info(self):
        """Get current system performance information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Try to get CPU temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = int(f.read()) / 1000.0
            except:
                cpu_temp = 0
            
            # GPU temp (if available)
            try:
                result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_temp = float(result.stdout.split('=')[1].split("'")[0])
                else:
                    gpu_temp = 0
            except:
                gpu_temp = 0
            
            self.system_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'cpu_temp': cpu_temp,
                'gpu_temp': gpu_temp,
                'camera_initialized': self.picam2 is not None,
                'models_loaded': self.whisper_model is not None and self.llama_model is not None,
                'wake_word_active': self.wake_word_active,
                'current_person': self.current_person if self.current_person else "No Entity Detected"
            }
            
            return self.system_info
            
        except Exception as e:
            self.emit_log(f"Error getting system info: {e}", 'error')
            return self.system_info
    
    def start_system(self):
        """Start the AI system"""
        if self.is_running:
            return False
        
        self.emit_log("Starting Chatty AI system...", 'info')
        
        # Verify models are loaded
        if not self.whisper_model or not self.llama_model:
            self.emit_log("Models not loaded - attempting to reload...", 'warning')
            if not self.preload_models():
                self.emit_log("Failed to load models - cannot start system", 'error')
                return False
        
        # Verify camera is ready
        if not self.picam2:
            self.emit_log("Camera not initialized - attempting to setup...", 'warning')
            if not self.setup_camera():
                self.emit_log("Failed to setup camera - cannot start system", 'error')
                return False
        
        self.is_running = True
        
        # Start background threads
        self.camera_thread = threading.Thread(target=self.camera_monitoring_loop, daemon=True)
        self.camera_thread.start()
        
        self.audio_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
        self.audio_thread.start()
        
        self.emit_log("Chatty AI system started successfully", 'success')
        return True
    
    def stop_system(self):
        """Stop the AI system"""
        if not self.is_running:
            return False
        
        self.emit_log("Stopping Chatty AI system...", 'info')
        self.is_running = False
        
        # Wait for threads to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=3)
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=3)
        
        self.emit_log("Chatty AI system stopped", 'info')
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.emit_log("Cleaning up resources...", 'info')
        self.is_running = False
        
        if self.picam2:
            try:
                self.picam2.stop()
            except:
                pass
        
        # Clean up audio files
        for audio_file in [WAV_FILENAME, RESPONSE_AUDIO, WAKE_WORD_AUDIO]:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
        
        self.emit_log("Cleanup complete", 'info')

# Global instance
chatty_ai = FastChattyAI()

# Flask Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('Chatty_AI_HighTech.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            try:
                with chatty_ai.frame_lock:
                    if chatty_ai.latest_frame is not None:
                        frame = chatty_ai.latest_frame.copy()
                    else:
                        # Create a black frame if no camera data
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Video feed error: {e}")
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/captured_image')
def captured_image():
    """Captured person image route"""
    try:
        with chatty_ai.frame_lock:
            if chatty_ai.latest_person_image is not None:
                frame = chatty_ai.latest_person_image.copy()
            else:
                # Create a placeholder frame
                frame = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(frame, "No Person", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            return Response(frame_bytes, mimetype='image/jpeg')
        else:
            # Return empty image if encoding fails
            return Response(b'', mimetype='image/jpeg')
            
    except Exception as e:
        print(f"Captured image error: {e}")
        return Response(b'', mimetype='image/jpeg')

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("Client connected")
    emit('status_update', {
        'is_running': chatty_ai.is_running,
        'status': 'running' if chatty_ai.is_running else 'stopped',
        'message': 'Connected to Chatty AI'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("Client disconnected")

@socketio.on('start_system')
def handle_start_system():
    """Handle system start request"""
    success = chatty_ai.start_system()
    emit('status_update', {
        'is_running': chatty_ai.is_running,
        'status': 'running' if success else 'error',
        'message': 'System started successfully' if success else 'Failed to start system'
    })

@socketio.on('stop_system')
def handle_stop_system():
    """Handle system stop request"""
    success = chatty_ai.stop_system()
    emit('status_update', {
        'is_running': chatty_ai.is_running,
        'status': 'stopped' if success else 'error',
        'message': 'System stopped successfully' if success else 'Failed to stop system'
    })

@socketio.on('get_system_info')
def handle_get_system_info():
    """Handle system info request"""
    info = chatty_ai.get_system_info()
    emit('system_info', info)

if __name__ == '__main__':
    print("Fast Chatty AI Web Server Starting...")
    print("=" * 60)
    print("✨ Optimized for maximum speed and performance")
    print("🚀 Models preloaded for instant response")
    print("🎯 Wake word detection optimized")
    print("📱 Web interface available")
    print("=" * 60)
    
    try:
        # Start the SocketIO server
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        chatty_ai.cleanup()
    except Exception as e:
        print(f"Server error: {e}")
        chatty_ai.cleanup()