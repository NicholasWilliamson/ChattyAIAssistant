#!/usr/bin/env python3
"""
chatty_ai_web.py - Complete AI Assistant with Facial Recognition and Wake Word Detection
Web UI version with real-time monitoring and control interface
"""

import os
import sys
import time
import logging
import threading
import psutil
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import random
import re
import cv2
import face_recognition
import pickle
import json
import requests
import io
import signal
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
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
BORED_RESPONSES_FILE = "bored_responses.txt"
VISITOR_GREETING_RESPONSES_FILE = "visitor_greeting_responses.txt"

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
BORED_RESPONSE_INTERVAL = 30  # Configurable duration for bored responses
PERSON_DETECTION_INTERVAL = 0.5  # Check for people every 0.5 seconds
WAKE_WORD_CHECK_INTERVAL = 1.0  # Check for wake words every 1 second

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatty_ai_web.log')
    ]
)
logger = logging.getLogger(__name__)

class ChattyAIWebServer:
    def __init__(self):
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='templates')
        self.app.config['SECRET_KEY'] = 'chatty_ai_web_secret_key_2025'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, 
                               cors_allowed_origins="*",
                               async_mode='threading',
                               ping_timeout=60,
                               ping_interval=25,
                               logger=False,
                               engineio_logger=False)
        
        # AI Models
        self.whisper_model = None
        self.llama_model = None
        
        # Facial Recognition
        self.known_encodings = []
        self.known_names = []
        
        # Camera
        self.picam2 = None
        self.current_frame = None
        self.captured_person_frame = None
        
        # System state
        self.system_running = False
        self.clients = set()
        self.camera_initialized = False
        self.models_loaded = False
        self.wake_word_active = False
        
        # AI Assistant state
        self.current_person = None
        self.last_greeting_time = {}
        self.last_interaction_time = None
        self.person_absent_since = None
        self.last_bored_response_time = None
        self.bored_cycle = 0
        self.audio_recording_lock = threading.Lock()
        
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
        self.system_monitor_thread = None
        self.monitor_running = False
        
        # Initialize directories and logging
        self.setup_directories()
        self.setup_ai_logging()
        
        # Setup routes and socket events
        self.setup_routes()
        self.setup_socket_events()
        
        logger.info("ChattyAI Web Server initialized")

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(SECURITY_PHOTOS_DIR, exist_ok=True)
        os.makedirs(SECURITY_LOGS_DIR, exist_ok=True)
        os.makedirs('logs', exist_ok=True)  # For web server logs

    def setup_ai_logging(self):
        """Setup logging for AI detections"""
        log_file = os.path.join(SECURITY_LOGS_DIR, "chatty_ai.log")
        self.ai_logger = logging.getLogger('chatty_ai')
        self.ai_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.ai_logger.handlers[:]:
            self.ai_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        self.ai_logger.addHandler(file_handler)

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
            
            self.emit_log("Response files loaded successfully", 'success')
            return True
        except FileNotFoundError as e:
            self.emit_log(f"Response file not found: {e}", 'error')
            self.create_default_responses()
            return False

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
        self.warning_responses = ["Attention unauthorized person, you are not authorized to access this property. Leave immediately. I am contacting the authorities to report your intrusion."]
        self.bored_responses = ["I'm getting a bit bored waiting here"]
        self.visitor_greeting_responses = ["Hello. I do not recognize you. Can I be of assistance?"]

    def load_models(self):
        """Load AI models"""
        try:
            self.emit_log("Loading AI models...", 'info')
            
            # Load Whisper model
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            self.emit_log("Whisper model loaded successfully", 'success')
            
            # Load LLaMA model
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=2048,
                temperature=0.7,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                verbose=False
            )
            self.emit_log("LLaMA model loaded successfully", 'success')
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            self.emit_log(f"Failed to load AI models: {e}", 'error')
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
            return True
        except FileNotFoundError:
            self.emit_log("Telegram config not found - alerts disabled", 'warning')
            return False
        except Exception as e:
            self.emit_log(f"Failed to load Telegram config: {e}", 'error')
            return False

    def setup_camera(self):
        """Initialize camera"""
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            ))
            self.picam2.start()
            time.sleep(2)  # Camera warm-up
            self.camera_initialized = True
            self.emit_log("Camera initialized successfully", 'success')
            return True
        except Exception as e:
            self.emit_log(f"Failed to initialize camera: {e}", 'error')
            return False

    def get_system_info(self):
        """Get current system performance information"""
        try:
            # Get CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            
            # Get temperature info (Raspberry Pi specific)
            cpu_temp = self.get_cpu_temperature()
            gpu_temp = self.get_gpu_temperature()
            
            return {
                'cpu_percent': round(cpu_percent, 1),
                'memory_percent': round(memory_percent, 1),
                'memory_used': memory_used,
                'memory_total': memory_total,
                'cpu_temp': round(cpu_temp, 2) if cpu_temp else 0,
                'gpu_temp': round(gpu_temp, 1) if gpu_temp else 0,
                'timestamp': datetime.now().isoformat(),
                'camera_initialized': self.camera_initialized,
                'models_loaded': self.models_loaded,
                'wake_word_active': self.wake_word_active,
                'current_person': self.current_person if self.current_person else "No person detected"
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used': 0,
                'memory_total': 0,
                'cpu_temp': 0,
                'gpu_temp': 0,
                'timestamp': datetime.now().isoformat(),
                'camera_initialized': False,
                'models_loaded': False,
                'wake_word_active': False,
                'current_person': 'Error'
            }

    def get_cpu_temperature(self):
        """Get CPU temperature for Raspberry Pi"""
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                if temp_str.startswith('temp=') and temp_str.endswith("'C"):
                    return float(temp_str[5:-2])
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        
        # Fallback: try thermal zone
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except (FileNotFoundError, ValueError, IOError):
            pass
        
        return None

    def get_gpu_temperature(self):
        """Get GPU temperature for Raspberry Pi"""
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                if temp_str.startswith('temp=') and temp_str.endswith("'C"):
                    return float(temp_str[5:-2]) + 1.0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        
        return None

    def emit_log(self, message, log_type='info'):
        """Emit log message to web clients"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': message,
            'type': log_type
        }
        
        if self.clients:
            self.socketio.emit('log_update', log_entry)
        
        # Also log to file
        if log_type == 'error':
            logger.error(message)
        elif log_type == 'warning':
            logger.warning(message)
        elif log_type == 'success':
            logger.info(f"✅ {message}")
        else:
            logger.info(message)

    def emit_conversation(self, message, conv_type='info'):
        """Emit conversation message to web clients"""
        conv_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': message,
            'type': conv_type
        }
        
        if self.clients:
            self.socketio.emit('conversation_update', conv_entry)

    def system_monitor_loop(self):
        """Background thread for system monitoring"""
        while self.monitor_running:
            try:
                if self.clients:  # Only send if there are connected clients
                    system_info = self.get_system_info()
                    self.socketio.emit('system_info', system_info)
                
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
                time.sleep(5)

    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            try:
                return render_template('Chatty_AI.html')
            except Exception as e:
                logger.error(f"Error rendering template: {e}")
                return f"Error: Could not find Chatty_AI.html in templates folder. Error: {e}", 500

        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            def generate():
                while True:
                    try:
                        if self.current_frame is not None and self.system_running:
                            # Convert frame to JPEG
                            is_success, buffer = cv2.imencode(".jpg", self.current_frame)
                            if is_success:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + 
                                      frame_bytes + b'\r\n')
                        else:
                            # Create placeholder image
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, "Camera Feed", (200, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            if not self.system_running:
                                cv2.putText(placeholder, "System Stopped", (190, 280), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            is_success, buffer = cv2.imencode(".jpg", placeholder)
                            if is_success:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + 
                                      frame_bytes + b'\r\n')
                        
                        time.sleep(0.1)  # ~10 FPS
                    except Exception as e:
                        logger.error(f"Error in video feed: {e}")
                        break
                        
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/captured_image')
        def captured_image():
            """Return the captured person image"""
            try:
                if self.captured_person_frame is not None:
                    is_success, buffer = cv2.imencode(".jpg", self.captured_person_frame)
                    if is_success:
                        return Response(buffer.tobytes(), mimetype='image/jpeg')
                
                # Return placeholder image
                placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Person", (80, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(placeholder, "Detected", (90, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                is_success, buffer = cv2.imencode(".jpg", placeholder)
                if is_success:
                    return Response(buffer.tobytes(), mimetype='image/jpeg')
                else:
                    return Response(b'', mimetype='image/jpeg')
                    
            except Exception as e:
                logger.error(f"Error serving captured image: {e}")
                return Response(b'', mimetype='image/jpeg')

        @self.app.route('/templates/<path:filename>')
        def serve_template_files(filename):
            """Serve template files (images, etc.)"""
            try:
                return send_from_directory('templates', filename)
            except Exception as e:
                logger.error(f"Error serving template file {filename}: {e}")
                return "File not found", 404

        @self.app.route('/api/system_info')
        def api_system_info():
            """API endpoint for system information"""
            return jsonify(self.get_system_info())

    def setup_socket_events(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.clients.add(client_id)
            logger.info(f"Client connected: {client_id}. Total clients: {len(self.clients)}")
            
            # Send initial status
            emit('status_update', {
                'status': 'running' if self.system_running else 'stopped',
                'is_running': self.system_running,
                'message': 'Connected to server successfully'
            })
            
            # Send initial system info
            try:
                system_info = self.get_system_info()
                emit('system_info', system_info)
            except Exception as e:
                logger.error(f"Error sending initial system info: {e}")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.clients.discard(client_id)
            logger.info(f"Client disconnected: {client_id}. Total clients: {len(self.clients)}")

        @self.socketio.on('start_system')
        def handle_start_system():
            """Handle start system request"""
            client_id = request.sid
            logger.info(f"Received start system request from client: {client_id}")
            
            try:
                if not self.system_running:
                    self.emit_log("Starting Chatty AI system...", 'info')
                    
                    # Load response files
                    if not self.load_response_files():
                        emit('status_update', {
                            'status': 'error',
                            'is_running': False,
                            'message': 'Failed to load response files'
                        })
                        return
                    
                    # Load AI models
                    if not self.load_models():
                        emit('status_update', {
                            'status': 'error',
                            'is_running': False,
                            'message': 'Failed to load AI models'
                        })
                        return
                    
                    # Load face encodings
                    if not self.load_encodings():
                        emit('status_update', {
                            'status': 'error',
                            'is_running': False,
                            'message': 'Failed to load face encodings'
                        })
                        return
                    
                    # Load Telegram config
                    self.load_telegram_config()
                    
                    # Setup camera
                    if not self.setup_camera():
                        emit('status_update', {
                            'status': 'error',
                            'is_running': False,
                            'message': 'Failed to initialize camera'
                        })
                        return
                    
                    # Start system monitoring
                    if not self.monitor_running:
                        self.monitor_running = True
                        self.system_monitor_thread = threading.Thread(
                            target=self.system_monitor_loop, daemon=True)
                        self.system_monitor_thread.start()
                        logger.info("System monitoring thread started")
                    
                    # Start camera monitoring
                    self.camera_thread = threading.Thread(target=self.camera_monitoring_loop, daemon=True)
                    self.camera_thread.start()
                    
                    # Start wake word detection
                    self.audio_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
                    self.audio_thread.start()
                    
                    self.system_running = True
                    self.emit_log("Chatty AI system started successfully", 'success')
                    
                    response_data = {
                        'status': 'running',
                        'is_running': True,
                        'message': 'Chatty AI system started successfully'
                    }
                    
                    # Send to all clients
                    self.socketio.emit('status_update', response_data)
                    logger.info("System started successfully - status sent to all clients")
                    
                else:
                    emit('status_update', {
                        'status': 'running',
                        'is_running': True,
                        'message': 'System is already running'
                    })
                    logger.info("Start system requested but system already running")
                    
            except Exception as e:
                logger.error(f"Error starting system: {e}")
                emit('status_update', {
                    'status': 'error',
                    'is_running': False,
                    'message': f'Error starting system: {e}'
                })

        @self.socketio.on('stop_system')
        def handle_stop_system():
            """Handle stop system request"""
            client_id = request.sid
            logger.info(f"Received stop system request from client: {client_id}")
            try:
                if self.system_running:
                    self.system_running = False
                    self.monitor_running = False
                    
                    # Reset AI Assistant state
                    self.current_person = None
                    self.wake_word_active = False
                    self.last_interaction_time = None
                    self.person_absent_since = None
                    self.last_bored_response_time = None
                    self.bored_cycle = 0
                    
                    # Stop camera
                    if self.picam2:
                        try:
                            self.picam2.stop()
                            self.picam2 = None
                            self.camera_initialized = False
                        except:
                            pass
                    
                    self.models_loaded = False
                    
                    self.emit_log("Chatty AI system stopped", 'info')
                    
                    response_data = {
                        'status': 'stopped',
                        'is_running': False,
                        'message': 'Chatty AI system stopped'
                    }
                    
                    # Send to all clients
                    self.socketio.emit('status_update', response_data)
                    logger.info("System stopped successfully - status sent to all clients")
                    
                else:
                    emit('status_update', {
                        'status': 'stopped',
                        'is_running': False,
                        'message': 'System is already stopped'
                    })
                    logger.info("Stop system requested but system already stopped")
                    
            except Exception as e:
                logger.error(f"Error stopping system: {e}")
                emit('status_update', {
                    'status': 'error',
                    'is_running': self.system_running,
                    'message': f'Error stopping system: {e}'
                })

        @self.socketio.on('get_system_info')
        def handle_get_system_info():
            """Handle system info request"""
            try:
                system_info = self.get_system_info()
                emit('system_info', system_info)
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
                emit('system_info', {
                    'error': f'Error getting system info: {e}'
                })

        @self.socketio.on('test_speech')
        def handle_test_speech(data):
            """Handle speech test request"""
            try:
                text = data.get('text', 'Test speech from Chatty AI')
                logger.info(f"Speech test requested: {text}")
                
                if self.system_running:
                    self.speak_text(text)
                    self.emit_conversation(f"🔊 Speech test: {text}", 'speech')
                    
                    emit('speech_test_result', {
                        'success': True,
                        'message': 'Speech test completed successfully'
                    })
                else:
                    emit('speech_test_result', {
                        'success': False,
                        'message': 'System must be running for speech test'
                    })
                
            except Exception as e:
                logger.error(f"Error in speech test: {e}")
                emit('speech_test_result', {
                    'success': False,
                    'message': f'Speech test failed: {e}'
                })

        @self.socketio.on('manual_wake_word')
        def handle_manual_wake_word():
            """Handle manual wake word trigger"""
            try:
                if self.system_running and self.current_person:
                    logger.info("Manual wake word triggered")
                    self.emit_conversation(f"⚡ Manual wake word activated for {self.current_person}", 'wake_word')
                    
                    # Trigger wake word response
                    self.handle_wake_word_detected()
                    
                    emit('manual_wake_result', {
                        'success': True,
                        'message': 'Manual wake word triggered successfully'
                    })
                else:
                    emit('manual_wake_result', {
                        'success': False,
                        'message': 'System must be running with person detected for manual wake word'
                    })
                
            except Exception as e:
                logger.error(f"Error in manual wake word: {e}")
                emit('manual_wake_result', {
                    'success': False,
                    'message': f'Manual wake word failed: {e}'
                })

    # AI Assistant Core Functions
    def is_daytime_hours(self):
        """Check if current time is between 6:00AM and 12:00PM (daytime visitor hours)"""
        current_hour = datetime.now().hour
        return 6 <= current_hour <= 12

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
            self.emit_log(f"TTS failed: {e}", 'error')
    
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
        self.ai_logger.info(f"Security photo saved: {filename} | Person: {person_name} | Confidence: {confidence:.2f}")
        
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
            self.emit_log(f"Telegram alert failed: {e}", 'error')
            return False

    def greet_person(self, name):
        """Greet a detected person with generic greeting using their name"""
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
        self.emit_conversation(f"👋 Greeted {name}: {greeting}", 'response')
        
        self.last_greeting_time[name] = current_time
        self.last_interaction_time = current_time
        
        # Enable wake word detection after greeting
        self.wake_word_active = True
        self.last_bored_response_time = current_time  # Reset bored response timer
        self.bored_cycle = 0  # Reset bored cycle
        self.emit_log(f"Greeted {name} - Wake word detection now active", 'success')
        return True

    def handle_unknown_person(self, frame, confidence):
        """Handle unknown person detection with time-based responses"""
        if self.is_daytime_hours():
            # 6:00AM - 12:00PM: Assume visitor, be friendly
            if self.visitor_greeting_responses:
                visitor_greeting = random.choice(self.visitor_greeting_responses)
            else:
                visitor_greeting = "Hello. I do not recognize you. Can I be of assistance?"
            self.speak_text(visitor_greeting)
            self.emit_conversation(f"👤 Unknown visitor greeted: {visitor_greeting}", 'response')
            self.emit_log("Unknown person detected during daytime - treated as visitor", 'warning')
        else:
            # 12:01PM - 5:59AM: Assume intruder, give warning
            if self.warning_responses:
                warning = random.choice(self.warning_responses)
            else:
                warning = "Attention unauthorized person, you are not authorized to access this property. Leave immediately. I am contacting the authorities to report your intrusion."
            self.speak_text(warning)
            self.emit_conversation(f"🚨 Intruder warning: {warning}", 'response')
            self.emit_log("Unknown person detected during nighttime/evening - treated as intruder", 'error')
        
        photo_path = self.save_security_photo(frame, "Unknown", confidence)
        self.send_telegram_alert("Unknown", confidence, photo_path)

    def get_llm_joke(self):
        """Ask the local LLM for a joke"""
        try:
            prompt = "Tell me a short, clean joke. Just the joke, nothing else."
            formatted_prompt = f"You are a comedian. Tell a brief, funny joke.\nUser: {prompt}\nAssistant: "
            
            result = self.llama_model(formatted_prompt, max_tokens=80)
            if "choices" in result and result["choices"]:
                joke = result["choices"][0]["text"].strip()
                # Clean up the response
                joke = re.sub(r"\(.*?\)", "", joke)
                joke = re.sub(r"(User:|Assistant:)", "", joke)
                joke = joke.strip()
                
                # Take only the first few sentences
                sentences = joke.split('.')
                if len(sentences) > 2:
                    joke = '. '.join(sentences[:2]) + '.'
                
                return joke if joke else "Why did the computer go to therapy? Because it had too many bytes!"
            else:
                return "Why did the computer go to therapy? Because it had too many bytes!"
        except Exception as e:
            self.emit_log(f"LLM joke error: {e}", 'error')
            return "Why did the computer go to therapy? Because it had too many bytes!"

    def get_llm_fun_fact(self):
        """Ask the local LLM for a fun fact"""
        try:
            prompt = "Tell me an interesting fun fact. Keep it brief and fascinating."
            formatted_prompt = f"You are a knowledgeable teacher. Share one interesting fact.\nUser: {prompt}\nAssistant: "
            
            result = self.llama_model(formatted_prompt, max_tokens=100)
            if "choices" in result and result["choices"]:
                fact = result["choices"][0]["text"].strip()
                # Clean up the response
                fact = re.sub(r"\(.*?\)", "", fact)
                fact = re.sub(r"(User:|Assistant:)", "", fact)
                fact = fact.strip()
                
                # Take only the first few sentences
                sentences = fact.split('.')
                if len(sentences) > 3:
                    fact = '. '.join(sentences[:3]) + '.'
                
                return fact if fact else "Did you know that octopuses have three hearts and blue blood?"
            else:
                return "Did you know that octopuses have three hearts and blue blood?"
        except Exception as e:
            self.emit_log(f"LLM fun fact error: {e}", 'error')
            return "Did you know that octopuses have three hearts and blue blood?"

    def check_for_bored_response(self, name):
        """Check if it's time to give a bored response with joke or fun fact from LLM"""
        if not self.wake_word_active or not self.last_bored_response_time:
            return False
        
        current_time = time.time()
        time_since_bored = current_time - self.last_bored_response_time
        
        if time_since_bored >= BORED_RESPONSE_INTERVAL:
            if self.bored_cycle == 0:
                # Give bored response + joke from LLM
                if self.bored_responses:
                    bored_template = random.choice(self.bored_responses)
                    bored_msg = bored_template.replace("{name}", name)
                else:
                    bored_msg = f"Yo {name}, still hanging around here waiting for you, dude!"
                
                joke = self.get_llm_joke()
                # Create complete message with 2-second pause represented as spoken text
                full_message = f"{bored_msg} Let me tell you a joke! ... ... {joke}"
                self.speak_text(full_message)
                self.emit_conversation(f"😴 Bored response + joke for {name}: {full_message}", 'response')
                self.bored_cycle = 1
                self.emit_log(f"Gave {name} a bored response with LLM joke", 'info')
            else:
                # Give waiting response + fun fact from LLM
                if self.waiting_responses:
                    waiting_template = random.choice(self.waiting_responses)
                    waiting_msg = waiting_template.replace("{name}", name)
                else:
                    waiting_msg = f"I am still around if you need me, {name}"
                
                fun_fact = self.get_llm_fun_fact()
                # Create complete message with 2-second pause represented as spoken text
                full_message = f"{waiting_msg} Let me tell you a fun fact! ... ... {fun_fact}"
                self.speak_text(full_message)
                self.emit_conversation(f"💡 Waiting response + fun fact for {name}: {full_message}", 'response')
                self.bored_cycle = 0
                self.emit_log(f"Gave {name} a waiting response with LLM fun fact", 'info')
            
            self.last_bored_response_time = current_time
            return True
        
        return False

    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper"""
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
        """Check if text contains wake word"""
        if not text:
            return False
            
        text_cleaned = text.lower().replace(',', '').replace('.', '').strip()
        
        for wake_word in WAKE_WORDS:
            wake_word_cleaned = wake_word.lower().strip()
            if wake_word_cleaned in text_cleaned:
                self.emit_log(f"Wake word detected: '{wake_word}' in '{text}'", 'success')
                return True
        return False

    def record_with_silence_detection(self):
        """Record audio until silence detected"""
        try:
            with self.audio_recording_lock:
                self.emit_log("Recording user audio...", 'info')
                audio_data = []
                silence_duration = 0
                recording_duration = 0
                check_interval = 0.2
                samples_per_check = int(SAMPLE_RATE * check_interval)
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        self.emit_log(f"Audio callback status: {status}", 'warning')
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
                                    self.emit_log(f"Silence detected after {recording_duration:.1f}s", 'info')
                                    break
                            else:
                                silence_duration = 0
                
                if audio_data:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sf.write(WAV_FILENAME, audio_array, SAMPLE_RATE)
                    self.emit_log(f"Audio saved: {len(audio_array)/SAMPLE_RATE:.1f}s duration", 'success')
                    return True
                
                return False
                
        except Exception as e:
            self.emit_log(f"Recording error: {e}", 'error')
            return False

    def record_wake_word_check(self):
        """Record short audio clip for wake word detection"""
        try:
            if not self.audio_recording_lock.acquire(blocking=False):
                return False  # Audio system is busy
            
            try:
                # Record 5 seconds of audio for wake word detection
                audio_data = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                
                # Check if audio contains sound above threshold
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > SILENCE_THRESHOLD * 2:  # Higher threshold for wake word
                    sf.write(WAKE_WORD_AUDIO, audio_data, SAMPLE_RATE)
                    return True
                else:
                    return False
                    
            finally:
                self.audio_recording_lock.release()
                
        except Exception as e:
            self.emit_log(f"Wake word recording error: {e}", 'error')
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
            current_time = datetime.now().strftime("%I:%M %p")
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
            self.emit_log(f"LLM error: {e}", 'error')
            return "Sorry, I had trouble processing that question."

    def process_user_input(self, text):
        """Process user input"""
        self.emit_log(f"Processing user input: '{text}'", 'info')
        command = self.is_command(text)
        if command:
            self.emit_log(f"Executing command: {command}", 'info')
            response = self.execute_command(command)
        else:
            self.emit_log("Generating LLM response", 'info')
            response = self.query_llama(text)
        
        return response

    def handle_wake_word_detected(self):
        """Handle wake word detection and start conversation"""
        self.play_beep()
        
        # Speak generic listening response with person's name
        if self.listening_responses:
            listening_template = random.choice(self.listening_responses)
            listening_response = listening_template.replace("{name}", self.current_person)
        else:
            listening_response = f"Yes {self.current_person}, I'm listening. What would you like to know?"
        
        self.speak_text(listening_response)
        self.emit_conversation(f"👂 Listening: {listening_response}", 'response')
        
        # Record full request
        self.emit_log("Please speak your request...", 'info')
        if self.record_with_silence_detection():
            user_text = self.transcribe_audio(WAV_FILENAME)
            if user_text and len(user_text.strip()) > 2:
                self.emit_conversation(f"🎤 User said: '{user_text}'", 'user_input')
                response = self.process_user_input(user_text)
                self.emit_conversation(f"🤖 Response: '{response}'", 'response')
                self.speak_text(response)
                self.last_interaction_time = time.time()
                # Reset bored response timer only after successful interaction
                self.last_bored_response_time = time.time()
            else:
                self.emit_log("No clear speech detected", 'warning')
                self.speak_text("I didn't catch that. Could you repeat your request?")
                self.emit_conversation("❓ No clear speech detected", 'info')
        else:
            self.emit_log("Failed to record user request", 'error')
            self.speak_text("I'm having trouble hearing you. Please try again.")
            self.emit_conversation("🚫 Failed to record audio", 'info')

    def listen_for_wake_word(self):
        """Listen for wake words in background"""
        self.emit_log("Wake word detection thread started", 'success')
        
        while self.system_running:
            try:
                # Only listen if someone is present and wake word detection is active
                if self.current_person and self.current_person != "Unknown" and self.wake_word_active:
                    # Check for bored response first
                    if self.check_for_bored_response(self.current_person):
                        # Bored response was given, continue to next iteration
                        time.sleep(WAKE_WORD_CHECK_INTERVAL)
                        continue
                    
                    # Record audio for wake word detection
                    if self.record_wake_word_check():
                        # Transcribe and check for wake word
                        transcript = self.transcribe_audio(WAKE_WORD_AUDIO)
                        
                        if transcript and self.detect_wake_word(transcript):
                            self.emit_log("WAKE WORD DETECTED! Starting conversation...", 'success')
                            self.emit_conversation(f"🌟 Wake word detected: {transcript}", 'wake_word')
                            
                            # Handle the wake word detection
                            self.handle_wake_word_detected()
                    
                    time.sleep(WAKE_WORD_CHECK_INTERVAL)
                else:
                    # No one present or wake word not active, sleep longer
                    time.sleep(2.0)
                
            except Exception as e:
                self.emit_log(f"Wake word detection error: {e}", 'error')
                time.sleep(2.0)
        
        self.emit_log("Wake word detection thread stopped", 'info')

    def camera_monitoring_loop(self):
        """Main camera monitoring loop for web interface"""
        self.emit_log("Camera monitoring thread started", 'success')
        
        while self.system_running:
            try:
                if not self.picam2:
                    time.sleep(1)
                    continue
                    
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process facial recognition
                name, face_location, confidence = self.detect_faces(frame)
                
                current_time = time.time()
                
                # Draw face rectangles and labels on frame for web display
                display_frame = frame.copy()
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
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # Draw label text
                    cv2.putText(display_frame, label, (left + 6, bottom - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Add status information to frame
                status_text = "Chatty AI Web Interface Active"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show current person status
                if self.current_person:
                    person_text = f"Current Person: {self.current_person}"
                    cv2.putText(display_frame, person_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show wake word status
                if self.wake_word_active:
                    wake_word_text = "Wake Word Detection: ACTIVE"
                    cv2.putText(display_frame, wake_word_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    wake_word_text = "Wake Word Detection: INACTIVE"
                    cv2.putText(display_frame, wake_word_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Show bored response timer
                if self.last_bored_response_time and self.wake_word_active:
                    time_since_bored = current_time - self.last_bored_response_time
                    timer_text = f"Bored Timer: {int(BORED_RESPONSE_INTERVAL - time_since_bored)}s"
                    cv2.putText(display_frame, timer_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Update current frame for web streaming
                self.current_frame = display_frame
                
                # Process facial recognition logic
                if name and face_location:
                    # Person detected
                    if name != self.current_person:
                        # New person or person changed
                        self.current_person = name
                        self.person_absent_since = None
                        self.wake_word_active = False  # Reset wake word state
                        
                        # Create captured person frame (crop face area)
                        if face_location:
                            top, right, bottom, left = face_location
                            # Expand the crop area a bit
                            margin = 50
                            crop_top = max(0, top - margin)
                            crop_bottom = min(frame.shape[0], bottom + margin)
                            crop_left = max(0, left - margin)
                            crop_right = min(frame.shape[1], right + margin)
                            
                            cropped_face = frame[crop_top:crop_bottom, crop_left:crop_right]
                            # Resize to standard size
                            self.captured_person_frame = cv2.resize(cropped_face, (300, 300))
                        
                        # Emit person detection to web clients
                        person_data = {
                            'name': name,
                            'confidence': f"{confidence:.1%}" if name != "Unknown" else "N/A",
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        }
                        
                        if self.clients:
                            self.socketio.emit('person_detected', person_data)
                        
                        if name == "Unknown":
                            self.handle_unknown_person(frame, confidence)
                        else:
                            # Save photo and send telegram alert for known person
                            photo_path = self.save_security_photo(frame, name, confidence)
                            self.send_telegram_alert(name, confidence, photo_path)
                            
                            # Greet known person (this will activate wake word detection)
                            self.greet_person(name)
                
                else:
                    # No person detected
                    if self.current_person:
                        if not self.person_absent_since:
                            self.person_absent_since = current_time
                        elif current_time - self.person_absent_since >= GREETING_COOLDOWN:
                            # Person has been absent for 5+ minutes, reset
                            old_person = self.current_person
                            self.current_person = None
                            self.person_absent_since = None
                            self.last_interaction_time = None
                            self.wake_word_active = False
                            self.last_bored_response_time = None  # Reset bored response timer
                            self.bored_cycle = 0  # Reset bored cycle
                            self.captured_person_frame = None
                            
                            # Emit person left to web clients
                            person_data = {
                                'name': "No person detected",
                                'confidence': "--",
                                'timestamp': datetime.now().strftime('%H:%M:%S')
                            }
                            
                            if self.clients:
                                self.socketio.emit('person_detected', person_data)
                            
                            self.emit_log(f"{old_person} left - resetting state", 'info')
                
                time.sleep(PERSON_DETECTION_INTERVAL)
                
            except Exception as e:
                self.emit_log(f"Camera loop error: {e}", 'error')
                time.sleep(1)
        
        self.emit_log("Camera monitoring thread stopped", 'info')

    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        self.system_running = False
        self.monitor_running = False
        
        # Stop camera
        if self.picam2:
            try:
                self.picam2.stop()
                logger.info("Camera stopped")
            except:
                pass
        
        # Clean up audio files
        for audio_file in [WAV_FILENAME, RESPONSE_AUDIO, WAKE_WORD_AUDIO]:
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass

    def get_local_ip(self):
        """Get local IP address"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web server"""
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Print startup banner
            print("\n" + "="*60)
            print("🚀 Starting Chatty AI Web Interface")
            print("="*60)
            print("Web Interface Features:")
            print("• Complete AI Assistant Integration")
            print("• Live Video Feed with Facial Recognition")
            print("• Real-time System Monitoring")
            print("• Wake Word Detection")
            print("• Speech Synthesis & Processing")
            print("• Security Monitoring & Telegram Alerts")
            print("• LLM-Generated Entertainment")
            print("="*60)
            print(f"Access the web interface at: http://localhost:{port}")
            print(f"Or from other devices at: http://{self.get_local_ip()}:{port}")
            print("Press Ctrl+C to stop the server")
            print("="*60)
            print("\nDEBUG INFO:")
            print(f"• Wake words: {len(WAKE_WORDS)} phrases loaded")
            print(f"• Bored response interval: {BORED_RESPONSE_INTERVAL} seconds")
            print(f"• Audio sample rate: {SAMPLE_RATE} Hz")
            print(f"• Silence threshold: {SILENCE_THRESHOLD}")
            print(f"• Daytime visitor hours: 6:00AM - 12:00PM")
            print(f"• Intruder detection hours: 12:01PM - 5:59AM")
            print("="*60)
            
            # Start the server with better error handling
            logger.info("Starting Flask-SocketIO server...")
            self.socketio.run(self.app, 
                            host=host, 
                            port=port, 
                            debug=debug,
                            use_reloader=False,
                            log_output=True)
                            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            self.cleanup()
            raise
        finally:
            self.cleanup()

def main():
    """Main function"""
    try:
        # Check audio devices
        try:
            print("Available audio devices:")
            print(sd.query_devices())
            print("=" * 60)
        except Exception as e:
            print(f"Could not query audio devices: {e}")
        
        # Create and run server
        server = ChattyAIWebServer()
        server.run(debug=False)
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()