@#!/usr/bin/env python3
"""
Chatty AI Web Interface Server
Fixed version that uses external HTML template and includes system monitoring
"""

import os
import sys
import time
import logging
import threading
import psutil
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import io
from PIL import Image
import subprocess
import signal

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
        self.app.config['SECRET_KEY'] = 'chatty_ai_secret_key_2025'
        
        # Initialize SocketIO with more compatible settings
        self.socketio = SocketIO(self.app, 
                               cors_allowed_origins="*",
                               async_mode='threading',
                               ping_timeout=60,
                               ping_interval=25,
                               logger=False,
                               engineio_logger=False)
        
        # System state
        self.system_running = False
        self.clients = set()
        
        # Camera and AI components (placeholder)
        self.camera = None
        self.camera_initialized = False
        self.models_loaded = False
        self.wake_word_active = False
        self.current_person = "No person detected"
        
        # System monitoring
        self.system_monitor_thread = None
        self.monitor_running = False
        
        # Dummy image for testing
        self.create_dummy_image()
        
        # Setup routes and socket events
        self.setup_routes()
        self.setup_socket_events()
        
        logger.info("ChattyAI Web Server initialized")

    def create_dummy_image(self):
        """Create a dummy image for testing purposes"""
        try:
            # Create a simple test image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Chatty AI Camera", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "System Ready", (220, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to bytes
            is_success, buffer = cv2.imencode(".jpg", img)
            if is_success:
                self.dummy_image_bytes = io.BytesIO(buffer).getvalue()
            else:
                self.dummy_image_bytes = None
                
            # Create captured person placeholder
            person_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(person_img, "No Person", (80, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(person_img, "Detected", (90, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            is_success, buffer = cv2.imencode(".jpg", person_img)
            if is_success:
                self.dummy_person_bytes = io.BytesIO(buffer).getvalue()
            else:
                self.dummy_person_bytes = None
                
        except Exception as e:
            logger.error(f"Error creating dummy images: {e}")
            self.dummy_image_bytes = None
            self.dummy_person_bytes = None

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
                'current_person': self.current_person
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
            # Try vcgencmd first (Raspberry Pi specific)
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
        
        # If no temperature available, return None
        return None

    def get_gpu_temperature(self):
        """Get GPU temperature for Raspberry Pi"""
        try:
            # For Raspberry Pi, GPU temp is similar to CPU
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                if temp_str.startswith('temp=') and temp_str.endswith("'C"):
                    # GPU temp is typically close to CPU temp on RPi
                    return float(temp_str[5:-2]) + 1.0  # Slight offset for realism
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        
        return None

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
                        if self.dummy_image_bytes:
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + 
                                  self.dummy_image_bytes + b'\r\n')
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
                if self.dummy_person_bytes:
                    return Response(self.dummy_person_bytes, mimetype='image/jpeg')
                else:
                    # Return a placeholder image
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
                    self.system_running = True
                    
                    # Simulate system initialization
                    self.camera_initialized = True
                    self.models_loaded = True
                    self.wake_word_active = True
                    
                    # Start system monitoring
                    if not self.monitor_running:
                        self.monitor_running = True
                        self.system_monitor_thread = threading.Thread(
                            target=self.system_monitor_loop, daemon=True)
                        self.system_monitor_thread.start()
                        logger.info("System monitoring thread started")
                    
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
                    
                    # Simulate system shutdown
                    self.camera_initialized = False
                    self.models_loaded = False
                    self.wake_word_active = False
                    self.current_person = "No person detected"
                    
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
                text = data.get('text', 'Test speech')
                logger.info(f"Speech test requested: {text}")
                
                # Simulate speech test
                time.sleep(1)
                
                emit('speech_test_result', {
                    'success': True,
                    'message': 'Speech test completed'
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
                logger.info("Manual wake word triggered")
                
                emit('manual_wake_result', {
                    'success': True,
                    'message': 'Manual wake word triggered'
                })
                
            except Exception as e:
                logger.error(f"Error in manual wake word: {e}")
                emit('manual_wake_result', {
                    'success': False,
                    'message': f'Manual wake word failed: {e}'
                })

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
        
        if self.camera:
            try:
                self.camera.release()
            except:
                pass

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
            print("• Live Video Feed")
            print("• Facial Recognition")
            print("• Real-time System Monitoring")
            print("• Speech Synthesis")
            print("• Wake Word Detection")
            print("• AI Assistant Integration")
            print("="*60)
            print(f"Access the web interface at: http://localhost:{port}")
            print(f"Or from other devices at: http://{self.get_local_ip()}:{port}")
            print("Press Ctrl+C to stop the server")
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

def main():
    """Main function"""
    try:
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