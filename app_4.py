#!/usr/bin/env python3
"""
🤖 Chatty AI Web Interface - Enhanced with Telegram & Wake Word Fixes
==================================================================
A complete AI assistant with face recognition, voice interaction, and Telegram alerts
Maintains all original functionality from chatty_ai.py

Author: Nick  
Version: 4.1 (Fixed - Original Behavior Restored)
"""

import os
import sys
import time
import json
import threading
import logging
import queue
from datetime import datetime
import cv2
import numpy as np
import struct
import io
from PIL import Image
import requests

# Flask and SocketIO
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

# Audio processing
import pyaudio
import speech_recognition as sr
import pyttsx3

# AI Models
import whisper
from llama_cpp import Llama

# Computer vision
import face_recognition
from picamera2 import Picamera2

# Utilities
import psutil
import GPUtil

# ============================================================================
# ENHANCED TELEGRAM HELPER CLASS
# ============================================================================

class TelegramHelper:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        logging.info(f"📱 Telegram Helper initialized")
        
    def send_photo_with_text(self, image, caption=""):
        """Send photo with caption to Telegram"""
        try:
            # Convert image to bytes if it's a numpy array
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(image)
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format='JPEG', quality=85)
                img_bytes.seek(0)
            else:
                img_bytes = image
            
            # Prepare the request
            url = f"{self.base_url}/sendPhoto"
            
            files = {
                'photo': ('detection.jpg', img_bytes, 'image/jpeg')
            }
            
            data = {
                'chat_id': self.chat_id,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            
            # Send the request with timeout
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                logging.info(f"✅ Telegram photo sent successfully")
                return True
            else:
                logging.error(f"❌ Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Wake word detection start error: {e}")
            return False
    
    def handle_wake_word_detected(self):
        """Handle wake word detection event - ORIGINAL FUNCTIONALITY"""
        try:
            logging.info("🎤 Processing wake word detection...")
            
            # Emit wake word event to web interface
            self.socketio.emit('wake_word_detected', {
                'message': 'Wake word detected - listening for command...',
                'timestamp': time.time(),
                'person': self.last_detected_person
            })
            
            # Add conversation log entry
            timestamp = datetime.now().strftime("%H:%M:%S")
            wake_message = f"{timestamp} 🎤 Wake word detected - Ready to listen"
            
            self.socketio.emit('conversation_update', {
                'message': wake_message,
                'type': 'wake_word'
            })
            
            # Start speech recognition (ORIGINAL BEHAVIOR)
            self._start_speech_recognition()
            
        except Exception as e:
            logging.error(f"❌ Wake word handling error: {e}")
    
    def _start_speech_recognition(self):
        """Start speech recognition after wake word - ORIGINAL FUNCTIONALITY"""
        try:
            logging.info("🎤 Starting speech recognition...")
            
            # Use speech recognition to capture voice command
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
            logging.info("🎤 Listening for voice command...")
            
            # Emit listening status to web interface
            self.socketio.emit('conversation_update', {
                'message': f"{datetime.now().strftime('%H:%M:%S')} 🎤 Listening for your command...",
                'type': 'listening'
            })
            
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            # Recognize speech
            try:
                user_input = self.recognizer.recognize_google(audio)
                logging.info(f"🎤 Recognized: {user_input}")
                
                # Emit recognized text to web interface
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.socketio.emit('conversation_update', {
                    'message': f"{timestamp} 👤 You said: {user_input}",
                    'type': 'user_input'
                })
                
                # Process the command with AI
                self._process_voice_command(user_input)
                
            except sr.UnknownValueError:
                logging.info("🎤 Could not understand audio")
                self.speak_text("Sorry, I didn't understand that. Could you please repeat?")
            except sr.RequestError as e:
                logging.error(f"🎤 Speech recognition error: {e}")
                self.speak_text("Sorry, I'm having trouble with speech recognition.")
                
        except Exception as e:
            logging.error(f"🎤 Speech recognition error: {e}")
    
    def _process_voice_command(self, user_input):
        """Process voice command with AI - ORIGINAL FUNCTIONALITY"""
        try:
            logging.info(f"🧠 Processing command: {user_input}")
            
            # Get personalized response
            person_name = self.last_detected_person
            response = self._get_ai_response(user_input, person_name)
            
            # Speak the response
            logging.info(f"🤖 AI Response: {response}")
            self.speak_text(response)
            
            # Emit to web interface
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.socketio.emit('conversation_update', {
                'message': f"{timestamp} 🤖 Chatty AI: {response}",
                'type': 'ai_response'
            })
            
        except Exception as e:
            logging.error(f"🧠 Error processing voice command: {e}")
            self.speak_text("Sorry, I'm having trouble processing that request.")
    
    def _get_ai_response(self, user_input, person_name):
        """Generate AI response using LLaMA - ORIGINAL FUNCTIONALITY"""
        try:
            # Create personalized prompt
            if person_name in self.response_data:
                personality = f"You are speaking to {person_name}. "
                context = self.response_data[person_name].get('context', '')
                if context:
                    personality += f"Context: {context}. "
            else:
                personality = f"You are speaking to {person_name}. "
            
            prompt = f"{personality}Please respond naturally and helpfully to: {user_input}"
            
            # Generate response with LLaMA
            if self.llama_model:
                response = self.llama_model(
                    prompt,
                    max_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["Human:", "User:", "\n\n"]
                )
                
                ai_response = response['choices'][0]['text'].strip()
                
                # Clean up response
                if ai_response.startswith("AI:") or ai_response.startswith("Assistant:"):
                    ai_response = ai_response.split(":", 1)[1].strip()
                
                return ai_response if ai_response else "I understand what you're saying."
            else:
                # Fallback response if model not available
                return "I understand what you're saying, but I'm having trouble generating a detailed response right now."
                
        except Exception as e:
            logging.error(f"AI response generation error: {e}")
            return "I understand what you're saying."
    
    def _camera_monitoring_thread(self):
        """Camera monitoring and face detection thread - ORIGINAL BEHAVIOR"""
        logging.info("Camera monitoring thread started")
        
        face_detection_interval = 0.5  # Check for faces every 0.5 seconds
        last_face_check = 0
        
        while self.system_running and self.camera_ready:
            try:
                if self.picam2:
                    # Capture frame
                    frame = self.picam2.capture_array()
                    
                    if frame is not None:
                        # Convert RGBA to RGB if needed
                        if len(frame.shape) == 3 and frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        
                        # Convert to BGR for OpenCV
                        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Perform face detection periodically
                        current_time = time.time()
                        if current_time - last_face_check >= face_detection_interval:
                            last_face_check = current_time
                            self._detect_faces(frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logging.error(f"Camera monitoring error: {e}")
                time.sleep(1)
        
        logging.info("Camera monitoring thread stopped")
    
    def _detect_faces(self, rgb_frame):
        """Detect and recognize faces in the frame - ORIGINAL BEHAVIOR WITH FIXES"""
        try:
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    name = "Unknown"
                    confidence = 0.0
                    
                    if matches and len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            # Convert distance to confidence percentage
                            confidence = max(0, (1 - face_distances[best_match_index]) * 100)
                    
                    # Only process registered persons with good confidence
                    if name != "Unknown" and confidence > 60.0:
                        # Convert RGB frame to BGR for Telegram
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        
                        # Handle person detection with ORIGINAL BEHAVIOR
                        self.handle_person_detected(name, confidence, bgr_frame)
            else:
                # No faces detected - check if someone left the view
                current_time = time.time()
                for person_name in list(self.person_last_seen.keys()):
                    time_since_seen = current_time - self.person_last_seen[person_name]
                    if time_since_seen > 5.0:  # Person not seen for 5 seconds
                        if self.last_detected_person == person_name:
                            self.last_detected_person = "Unknown"
                            self.last_confidence = 0.0
                            logging.info(f"👤 {person_name} left the view")
                        del self.person_last_seen[person_name]
                        
        except Exception as e:
            logging.error(f"Face detection error: {e}")
    
    def start_system(self):
        """Start the complete Chatty AI system - ORIGINAL BEHAVIOR"""
        try:
            logging.info("Loading facial recognition encodings...")
            
            if self.system_running:
                logging.warning("System already running")
                return True
            
            # Load face encodings
            if not self.load_face_encodings():
                logging.error("Failed to load face encodings")
                return False
            
            logging.info("Chatty AI System Started!")
            
            # Initialize models if not already loaded
            if not self.models_loaded:
                if not self.initialize_models():
                    logging.error("Failed to initialize AI models")
                    return False
            
            # Initialize camera if not ready
            if not self.camera_ready:
                if not self.initialize_camera():
                    logging.error("Failed to initialize camera")
                    return False
            
            # Start camera monitoring thread
            self.system_running = True
            
            if not self.camera_thread or not self.camera_thread.is_alive():
                self.camera_thread = threading.Thread(target=self._camera_monitoring_thread, daemon=True)
                self.camera_thread.start()
                logging.info("Camera monitoring thread started")
            
            # Start wake word detection thread (this gets activated when person is detected)
            logging.info("Wake word detection thread started")
            
            # Start audio processing thread  
            logging.info("Audio processing thread started")
            
            # Start system info updates
            self._start_system_info_updates()
            
            logging.info("✅ System started successfully by web request")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start system: {e}")
            self.system_running = False
            return False
    
    def stop_system(self):
        """Stop the Chatty AI system"""
        try:
            logging.info("Stopping Chatty AI System...")
            
            self.system_running = False
            
            # Stop wake word detection
            if self.wake_word_detector:
                self.wake_word_detector.stop_listening()
            
            # Wait for camera thread to finish
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=5)
            
            logging.info("System stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping system: {e}")
    
    def _start_system_info_updates(self):
        """Start periodic system info updates"""
        def update_loop():
            while self.system_running:
                try:
                    self.socketio.emit('system_info', {
                        'message': 'System info sent to client',
                        'timestamp': time.time()
                    })
                    time.sleep(10)  # Update every 10 seconds
                except:
                    break
        
        info_thread = threading.Thread(target=update_loop, daemon=True)
        info_thread.start()
    
    def cleanup(self):
        """Cleanup all resources"""
        try:
            logging.info("Cleaning up resources...")
            
            self.stop_system()
            
            if self.wake_word_detector:
                self.wake_word_detector.cleanup()
            
            if self.picam2:
                self.picam2.close()
            
            if self.tts_engine:
                self.tts_engine.stop()
            
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

# ============================================================================
# BACKGROUND INITIALIZATION - ORIGINAL BEHAVIOR
# ============================================================================

def background_initialization(chatty_instance):
    """Initialize system components in background"""
    try:
        logging.info("🚀 Background: Starting system initialization...")
        logging.info("📅 Scheduled background initialization")
        time.sleep(2)  # Brief delay
        
        logging.info("📋 Initializing ChattyAI instance...")
        print("📋 Initializing ChattyAI instance...")
        
        # Test Telegram if configured
        if chatty_instance.telegram_helper:
            if chatty_instance.telegram_helper.test_connection():
                logging.info("✅ Telegram connection verified")
            else:
                logging.error("❌ Telegram connection failed")
        
        logging.info("🎉 System initialization completed successfully!")
        
    except Exception as e:
        logging.error(f"Background initialization error: {e}")

# ============================================================================
# MAIN APPLICATION SETUP - ORIGINAL BEHAVIOR
# ============================================================================

def create_app():
    """Create and configure the Flask application"""
    try:
        # Create ChattyAI instance
        chatty_ai = ChattyAIWeb()
        
        # Schedule background initialization
        init_thread = threading.Thread(target=background_initialization, args=(chatty_ai,), daemon=True)
        init_thread.start()
        
        return chatty_ai.app, chatty_ai.socketio, chatty_ai
        
    except Exception as e:
        logging.error(f"Failed to create Flask application: {e}")
        sys.exit(1)

# ============================================================================
# MAIN EXECUTION - ORIGINAL BEHAVIOR RESTORED
# ============================================================================

if __name__ == '__main__':
    print("🚀 Initializing Chatty AI Web Interface (app_2_corrected.py)")
    print("=" * 60)
    
    try:
        # Create the application
        app, socketio, chatty_ai = create_app()
        
        # Get local IP for display
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        print(f"🌐 Starting Flask server on http://0.0.0.0:5000")
        print(f"🔗 Access your Chatty AI interface at: http://{local_ip}:5000")
        print("=" * 60)
        
        # Check templates directory
        templates_dir = os.path.join(os.getcwd(), 'templates')
        logging.info(f"📁 Templates directory checked: {templates_dir}")
        
        if os.path.exists(os.path.join(templates_dir, 'Chatty_AI.html')):
            logging.info("✅ Chatty_AI.html template found")
        else:
            logging.error("❌ Chatty_AI.html template not found!")
        
        logging.info("🎯 Background initialization scheduled")
        logging.info("🌐 Starting Flask-SocketIO server...")
        
        # Setup cleanup on exit
        import atexit
        atexit.register(chatty_ai.cleanup)
        
        # Run the application with original settings
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
        if 'chatty_ai' in locals():
            chatty_ai.cleanup()
        print("👋 Chatty AI shutting down gracefully")
        
    except Exception as e:
        logging.error(f"Application startup error: {e}")
        sys.exit(1) e:
            logging.error(f"❌ Failed to send Telegram photo: {e}")
            return False
    
    def test_connection(self):
        """Test if bot token and chat ID are working"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                logging.info(f"✅ Telegram bot connected: @{bot_info['result']['username']}")
                return True
            else:
                logging.error(f"❌ Invalid bot token")
                return False
                
        except Exception as e:
            logging.error(f"❌ Telegram connection test failed: {e}")
            return False

# ============================================================================
# ENHANCED WAKE WORD DETECTOR CLASS  
# ============================================================================

class WakeWordDetector:
    def __init__(self):
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.wake_word_callback = None
        self.audio_stream = None
        self.audio_thread = None
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.pyaudio_instance = None
        self.last_wake_time = 0
        self.wake_cooldown = 3.0  # 3 seconds cooldown
        
    def initialize_audio(self):
        """Initialize PyAudio with error handling"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Get default input device
            try:
                default_device = self.pyaudio_instance.get_default_input_device_info()
                logging.info(f"🎤 Using default microphone: {default_device['name']}")
            except:
                logging.warning("🎤 No default input device, using first available")
                default_device = {'index': 0}
            
            # Try to open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=default_device.get('index'),
                frames_per_buffer=self.CHUNK
            )
            
            logging.info("✅ Audio stream initialized for wake word detection")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize audio: {e}")
            return False
    
    def start_listening(self, callback):
        """Start wake word detection"""
        try:
            if self.is_listening:
                return True
            
            self.wake_word_callback = callback
            
            # Initialize audio if needed
            if not self.audio_stream:
                if not self.initialize_audio():
                    return False
            
            self.is_listening = True
            
            # Start processing thread
            self.audio_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.audio_thread.start()
            
            logging.info("✅ Wake word detection started and listening")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to start wake word detection: {e}")
            return False
    
    def _process_audio(self):
        """Process audio for wake word detection"""
        wake_word_threshold = 2500  # Adjust based on your environment
        consecutive_detections = 0
        required_detections = 8
        
        while self.is_listening:
            try:
                if self.audio_stream and self.audio_stream.is_active():
                    try:
                        audio_data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                    except:
                        time.sleep(0.1)
                        continue
                else:
                    time.sleep(0.1)
                    continue
                
                # Convert audio data to numpy array
                try:
                    audio_array = struct.unpack(f'{len(audio_data)//2}h', audio_data)
                except:
                    continue
                
                # Simple volume-based detection
                if len(audio_array) > 0:
                    volume = max(abs(x) for x in audio_array if x != 0)
                    
                    if volume > wake_word_threshold:
                        consecutive_detections += 1
                        
                        if consecutive_detections >= required_detections:
                            current_time = time.time()
                            if current_time - self.last_wake_time > self.wake_cooldown:
                                logging.info(f"🎤 WAKE WORD DETECTED!")
                                consecutive_detections = 0
                                self.last_wake_time = current_time
                                
                                # Call the callback
                                if self.wake_word_callback:
                                    try:
                                        threading.Thread(target=self.wake_word_callback, daemon=True).start()
                                    except Exception as e:
                                        logging.error(f"Wake word callback error: {e}")
                            else:
                                consecutive_detections = 0
                    else:
                        consecutive_detections = max(0, consecutive_detections - 1)
                
            except Exception as e:
                logging.error(f"🎤 Audio processing error: {e}")
                time.sleep(0.1)
    
    def stop_listening(self):
        """Stop wake word detection"""
        try:
            self.is_listening = False
            
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
            
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
            
        except Exception as e:
            logging.error(f"❌ Error stopping wake word detection: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_listening()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
        except Exception as e:
            logging.error(f"🎤 Cleanup error: {e}")
    
    def get_status(self):
        """Get current status"""
        return "Active" if self.is_listening else "Inactive"

# ============================================================================
# MAIN CHATTY AI CLASS - RESTORED ORIGINAL BEHAVIOR
# ============================================================================

class ChattyAIWeb:
    def __init__(self):
        # Setup logging
        log_format = '[%(asctime)s] %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
        
        logging.info("🚀 Initializing Chatty AI Web Interface (app_2_corrected.py)")
        
        # Initialize Flask and SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'chatty_ai_secret_key_2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)
        
        # System state
        self.system_running = False
        self.models_loaded = False
        self.camera_ready = False
        
        # AI Models (will be loaded later)
        self.whisper_model = None
        self.llama_model = None
        
        # Text-to-speech engine
        self.tts_engine = None
        
        # Camera
        self.picam2 = None
        self.current_frame = None
        
        # Face recognition
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_detected_person = "Unknown"
        self.last_confidence = 0.0
        
        # Person detection tracking (ORIGINAL BEHAVIOR RESTORED)
        self.person_detection_times = {}  # Track when each person was last seen
        self.person_greeting_times = {}   # Track when each person was last greeted  
        self.greeting_cooldown = 300      # 5 minutes = 300 seconds
        self.person_last_seen = {}        # Track continuous presence
        
        # Enhanced components
        self.telegram_helper = None
        self.wake_word_detector = None
        
        # Configuration
        self.response_data = {}
        self.telegram_config = {}
        
        # Threads
        self.camera_thread = None
        self.audio_thread = None
        self.wake_word_thread = None
        
        # Audio processing
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Load configurations
        self._load_configurations()
        
        # Setup Flask routes and SocketIO handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        logging.info("ChattyAIWeb instance created - models will be loaded when system starts")
    
    def _load_configurations(self):
        """Load response data and telegram configuration"""
        try:
            # Load response data
            if os.path.exists('response_files.json'):
                with open('response_files.json', 'r') as f:
                    self.response_data = json.load(f)
                logging.info("Response files loaded successfully")
                logging.info(f"Personalized responses loaded for {len(self.response_data)} people")
            else:
                logging.warning("response_files.json not found")
                self.response_data = {}
            
            # Load telegram configuration
            if os.path.exists('telegram_config.json'):
                with open('telegram_config.json', 'r') as f:
                    self.telegram_config = json.load(f)
                logging.info("Telegram configuration loaded")
                
                # Initialize Telegram helper with fixed implementation
                if self.telegram_config.get('bot_token') and self.telegram_config.get('chat_id'):
                    self.telegram_helper = TelegramHelper(
                        self.telegram_config['bot_token'],
                        self.telegram_config['chat_id']
                    )
                    # Test connection in background
                    threading.Thread(target=self._test_telegram_connection, daemon=True).start()
                else:
                    logging.error("Telegram configuration incomplete")
            else:
                logging.warning("telegram_config.json not found")
                self.telegram_config = {}
                
        except Exception as e:
            logging.error(f"Error loading configurations: {e}")
    
    def _test_telegram_connection(self):
        """Test Telegram connection in background"""
        try:
            if self.telegram_helper:
                if self.telegram_helper.test_connection():
                    logging.info("✅ Telegram connection verified")
                else:
                    logging.error("❌ Telegram connection test failed")
        except Exception as e:
            logging.error(f"❌ Telegram test error: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('Chatty_AI.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            logging.info("📹 Serving video feed from Picamera2")
            return Response(
                self._generate_video_feed(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/captured_image')
        def captured_image():
            if self.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if ret:
                    return Response(
                        buffer.tobytes(),
                        mimetype='image/jpeg'
                    )
            # Return a blank image if no frame available
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            session_id = request.sid
            logging.info(f"🔌 WebSocket client connected: {client_ip} (ID: {session_id})")
            
            # Send initial system info
            self._emit_system_info()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logging.info(f"🔌 WebSocket client disconnected: {request.sid}")
        
        @self.socketio.on('start_system')
        def handle_start_system():
            logging.info("Starting Chatty AI System...")
            success = self.start_system()
            emit('system_status', {
                'status': 'started' if success else 'error',
                'message': 'System started successfully' if success else 'Failed to start system'
            })
        
        @self.socketio.on('stop_system')
        def handle_stop_system():
            logging.info("Stopping Chatty AI System...")
            self.stop_system()
            emit('system_status', {
                'status': 'stopped',
                'message': 'System stopped successfully'
            })
        
        @self.socketio.on('request_system_info')
        def handle_system_info_request():
            self._emit_system_info()
    
    def _emit_system_info(self):
        """Emit current system information to connected clients"""
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            system_info = {
                'socket_status': 'Connected',
                'camera_status': 'Ready' if self.camera_ready else 'Not Ready',
                'models_status': 'Loaded' if self.models_loaded else 'Not Loaded',
                'wake_word_status': self.wake_word_detector.get_status() if self.wake_word_detector else 'Inactive',
                'current_person': self.last_detected_person,
                'person_confidence': f"{self.last_confidence:.1f}%",
                'system_running': self.system_running,
                'timestamp': time.time()
            }
            
            self.socketio.emit('system_info', system_info)
            
        except Exception as e:
            logging.error(f"Error emitting system info: {e}")
    
    def _generate_video_feed(self):
        """Generate video frames for streaming"""
        logging.info("Video feed generation started")
        
        while True:
            try:
                if self.current_frame is not None:
                    # Create a copy for processing
                    frame = self.current_frame.copy()
                    
                    # Add overlay information
                    if self.last_detected_person != "Unknown":
                        cv2.putText(frame, f"{self.last_detected_person} ({self.last_confidence:.1f}%)",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add system status
                    status_text = "ACTIVE" if self.system_running else "INACTIVE"
                    color = (0, 255, 0) if self.system_running else (0, 0, 255)
                    cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Encode frame
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Generate blank frame
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Camera Not Ready", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', blank)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logging.error(f"Video feed error: {e}")
                time.sleep(1)
    
    def initialize_models(self):
        """Initialize AI models"""
        try:
            logging.info("Loading AI models...")
            
            # Load Whisper model
            try:
                self.whisper_model = whisper.load_model("base")
                logging.info("Whisper model loaded")
            except Exception as e:
                logging.error(f"Failed to load Whisper model: {e}")
                return False
            
            # Load LLaMA model
            try:
                model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
                if os.path.exists(model_path):
                    self.llama_model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=4,
                        verbose=False
                    )
                    logging.info("LLaMA model loaded")
                else:
                    logging.error(f"LLaMA model file not found: {model_path}")
                    return False
            except Exception as e:
                logging.error(f"Failed to load LLaMA model: {e}")
                return False
            
            # Initialize TTS engine
            try:
                self.tts_engine = pyttsx3.init()
                # Configure TTS settings
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)  # Use first voice
                self.tts_engine.setProperty('rate', 150)  # Speech rate
                self.tts_engine.setProperty('volume', 0.9)  # Volume
                logging.info("TTS engine initialized")
            except Exception as e:
                logging.error(f"Failed to initialize TTS: {e}")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            logging.error(f"Model initialization error: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera with multiple attempts"""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logging.info(f"Camera initialization attempt {attempt}/{max_attempts}")
                
                if self.picam2 is not None:
                    self.picam2.close()
                    time.sleep(1)
                
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                
                # Test capture
                time.sleep(2)
                test_frame = self.picam2.capture_array()
                
                if test_frame is not None:
                    # Convert RGBA to RGB if needed
                    if len(test_frame.shape) == 3 and test_frame.shape[2] == 4:
                        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGBA2RGB)
                    
                    self.current_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)
                    
                    logging.info(f"Picamera2 initialized successfully - Frame shape: {test_frame.shape}")
                    self.camera_ready = True
                    return True
                    
            except Exception as e:
                logging.error(f"Camera initialization attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    time.sleep(2)
                else:
                    self.camera_ready = False
                    return False
        
        return False
    
    def load_face_encodings(self):
        """Load known face encodings"""
        try:
            logging.info("Loading facial recognition encodings...")
            
            encodings_path = "face_encodings"
            if not os.path.exists(encodings_path):
                logging.warning(f"Face encodings directory not found: {encodings_path}")
                return False
            
            self.known_face_encodings = []
            self.known_face_names = []
            
            for filename in os.listdir(encodings_path):
                if filename.endswith('.npy'):
                    name = os.path.splitext(filename)[0]
                    encoding_path = os.path.join(encodings_path, filename)
                    
                    try:
                        encoding = np.load(encoding_path)
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                    except Exception as e:
                        logging.error(f"Failed to load encoding for {name}: {e}")
            
            logging.info(f"Loaded {len(self.known_face_encodings)} face encodings")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load face encodings: {e}")
            return False
    
    def speak_text(self, text):
        """Speak text using TTS engine - ORIGINAL FUNCTIONALITY"""
        try:
            if self.tts_engine:
                logging.info(f"🔊 Speaking: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                logging.error("TTS engine not initialized")
        except Exception as e:
            logging.error(f"TTS error: {e}")
    
    def send_telegram_alert(self, person_name, confidence, image):
        """Send Telegram alert - FIXED VERSION"""
        try:
            if not self.telegram_helper:
                logging.error("Telegram not initialized")
                return False
            
            caption = f"🚨 <b>Person Detected</b>\n\n" \
                     f"👤 <b>Person:</b> {person_name}\n" \
                     f"📊 <b>Confidence:</b> {confidence:.1f}%\n" \
                     f"🕐 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            
            success = self.telegram_helper.send_photo_with_text(image, caption)
            
            if success:
                logging.info(f"✅ Telegram alert sent: {person_name}")
            else:
                logging.error(f"❌ Failed to send Telegram alert")
                
            return success
            
        except Exception as e:
            logging.error(f"❌ Error sending Telegram alert: {e}")
            return False
    
    def handle_person_detected(self, person_name, confidence, image):
        """Handle person detection - ORIGINAL BEHAVIOR RESTORED"""
        try:
            current_time = time.time()
            
            # Update person tracking
            self.last_detected_person = person_name
            self.last_confidence = confidence
            self.person_last_seen[person_name] = current_time
            
            # Check if this person should be greeted (ORIGINAL 5-MINUTE LOGIC)
            last_greeting_time = self.person_greeting_times.get(person_name, 0)
            time_since_last_greeting = current_time - last_greeting_time
            
            should_greet = time_since_last_greeting >= self.greeting_cooldown
            
            if should_greet:
                # Update greeting time
                self.person_greeting_times[person_name] = current_time
                
                # 1. SEND TELEGRAM ALERT (ORIGINAL BEHAVIOR)
                logging.info(f"📱 Sending Telegram alert for {person_name}")
                telegram_thread = threading.Thread(
                    target=self.send_telegram_alert,
                    args=(person_name, confidence, image),
                    daemon=True
                )
                telegram_thread.start()
                
                # 2. SPEAK GREETING (ORIGINAL BEHAVIOR) 
                greeting_text = self._get_personalized_greeting(person_name)
                logging.info(f"🔊 Greeting {person_name}: {greeting_text}")
                speak_thread = threading.Thread(
                    target=self.speak_text,
                    args=(greeting_text,),
                    daemon=True
                )
                speak_thread.start()
                
                # 3. START WAKE WORD DETECTION (ORIGINAL BEHAVIOR)
                if not self.wake_word_detector or not self.wake_word_detector.is_listening:
                    logging.info("🎤 Starting wake word detection for new person")
                    self._start_wake_word_detection()
                
                # 4. EMIT TO WEB INTERFACE
                timestamp = datetime.now().strftime("%H:%M:%S")
                greeting_message = f"{timestamp} 👋 Greeting: {person_name} - {greeting_text}"
                
                self.socketio.emit('conversation_update', {
                    'message': greeting_message,
                    'type': 'greeting'
                })
                
                # 5. LOG THE DETECTION
                detection_message = f"{timestamp} 👤 Person detected: {person_name} ({confidence:.1f}%)"
                self.socketio.emit('conversation_update', {
                    'message': detection_message,
                    'type': 'detection'
                })
                
                logging.info(f"🎉 Full greeting sequence completed for {person_name}")
            
            else:
                # Just log the continued presence (ORIGINAL BEHAVIOR)
                timestamp = datetime.now().strftime("%H:%M:%S")
                detection_message = f"{timestamp} 👤 Person detected: {person_name} ({confidence:.1f}%)"
                
                self.socketio.emit('conversation_update', {
                    'message': detection_message,
                    'type': 'detection'
                })
                
                logging.info(detection_message)
                
        except Exception as e:
            logging.error(f"❌ Error handling person detection: {e}")
    
    def _get_personalized_greeting(self, person_name):
        """Get personalized greeting for person - ORIGINAL FUNCTIONALITY"""
        try:
            if person_name in self.response_data:
                return self.response_data[person_name].get('greeting', f"Hello {person_name}!")
            else:
                return f"Hello {person_name}! Nice to see you!"
        except:
            return f"Hello {person_name}!"
    
    def _start_wake_word_detection(self):
        """Start wake word detection - FIXED VERSION"""
        try:
            if not self.wake_word_detector:
                self.wake_word_detector = WakeWordDetector()
                if not self.wake_word_detector.initialize_audio():
                    logging.error("Failed to initialize wake word detector")
                    return False
            
            def wake_word_callback():
                logging.info("🎤 WAKE WORD DETECTED - Starting conversation mode")
                self.handle_wake_word_detected()
            
            success = self.wake_word_detector.start_listening(wake_word_callback)
            
            if success:
                logging.info("✅ Wake word detection started")
            else:
                logging.error("❌ Failed to start wake word detection")
                
            return success
            
        except Exception as