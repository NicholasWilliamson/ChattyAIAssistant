#!/usr/bin/env python3
"""
Chatty AI Web Interface - Fixed Socket Configuration
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import json
import threading
import time
import logging
from datetime import datetime
import psutil
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chatty_ai_secret_key_2024'

# Initialize SocketIO with proper configuration for Raspberry Pi
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25
)

class ChattyAISystem:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.system_status = "stopped"
        self.connected_clients = 0
        
    def start_camera(self):
        """Initialize camera"""
        try:
            # Try different camera indices for Raspberry Pi
            for camera_index in [0, 1, 2]:
                self.camera = cv2.VideoCapture(camera_index)
                if self.camera.isOpened():
                    # Set camera properties for Raspberry Pi
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 15)
                    logger.info(f"Camera initialized successfully on index {camera_index}")
                    return True
                else:
                    self.camera.release()
                    
            logger.error("Failed to initialize any camera")
            return False
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def get_system_info(self):
        """Get current system information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get GPU temperature for Raspberry Pi
            try:
                result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                      capture_output=True, text=True)
                gpu_temp = result.stdout.strip().replace('temp=', '').replace("'C", '')
                gpu_temp = float(gpu_temp)
            except:
                gpu_temp = 0
            
            # Get CPU temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000.0
            except:
                cpu_temp = 0
                
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_total': memory.total,
                'cpu_temp': cpu_temp,
                'gpu_temp': gpu_temp,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return None
    
    def start_system(self):
        """Start the AI system"""
        try:
            logger.info("Starting Chatty AI system...")
            
            if not self.start_camera():
                return False, "Failed to initialize camera"
            
            self.is_running = True
            self.system_status = "running"
            
            # Start video streaming thread
            self.video_thread = threading.Thread(target=self.video_stream_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            # Start system monitoring thread
            self.monitor_thread = threading.Thread(target=self.system_monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Chatty AI system started successfully")
            return True, "System started successfully"
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False, f"Error starting system: {e}"
    
    def stop_system(self):
        """Stop the AI system"""
        try:
            logger.info("Stopping Chatty AI system...")
            
            self.is_running = False
            self.system_status = "stopped"
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            logger.info("Chatty AI system stopped")
            return True, "System stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            return False, f"Error stopping system: {e}"
    
    def video_stream_loop(self):
        """Video streaming loop"""
        while self.is_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Resize frame for better performance
                    frame = cv2.resize(frame, (320, 240))
                    
                    # Add timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit to connected clients
                    socketio.emit('video_frame', {'frame': frame_data})
                    
                time.sleep(0.1)  # ~10 FPS for Raspberry Pi
                
            except Exception as e:
                logger.error(f"Video stream error: {e}")
                break
    
    def system_monitor_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                system_info = self.get_system_info()
                if system_info:
                    socketio.emit('system_info', system_info)
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                break

# Initialize system
chatty_system = ChattyAISystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': chatty_system.system_status,
        'is_running': chatty_system.is_running,
        'connected_clients': chatty_system.connected_clients
    })

@app.route('/api/system-info')
def get_system_info():
    """Get system information"""
    info = chatty_system.get_system_info()
    if info:
        return jsonify(info)
    else:
        return jsonify({'error': 'Unable to get system info'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    chatty_system.connected_clients += 1
    logger.info(f"Client connected. Total clients: {chatty_system.connected_clients}")
    
    # Send initial status
    emit('status_update', {
        'status': chatty_system.system_status,
        'is_running': chatty_system.is_running,
        'message': 'Connected to server successfully'
    })
    
    # Send initial system info
    system_info = chatty_system.get_system_info()
    if system_info:
        emit('system_info', system_info)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    chatty_system.connected_clients -= 1
    logger.info(f"Client disconnected. Total clients: {chatty_system.connected_clients}")

@socketio.on('start_system')
def handle_start_system():
    """Handle start system request"""
    logger.info("Received start system request")
    success, message = chatty_system.start_system()
    
    emit('system_response', {
        'success': success,
        'message': message,
        'status': chatty_system.system_status
    })
    
    # Broadcast status update to all clients
    socketio.emit('status_update', {
        'status': chatty_system.system_status,
        'is_running': chatty_system.is_running,
        'message': message
    })

@socketio.on('stop_system')
def handle_stop_system():
    """Handle stop system request"""
    logger.info("Received stop system request")
    success, message = chatty_system.stop_system()
    
    emit('system_response', {
        'success': success,
        'message': message,
        'status': chatty_system.system_status
    })
    
    # Broadcast status update to all clients
    socketio.emit('status_update', {
        'status': chatty_system.system_status,
        'is_running': chatty_system.is_running,
        'message': message
    })

@socketio.on('get_system_info')
def handle_get_system_info():
    """Handle system info request"""
    system_info = chatty_system.get_system_info()
    if system_info:
        emit('system_info', system_info)
    else:
        emit('system_error', {'message': 'Unable to get system information'})

if __name__ == '__main__':
    print("\n🚀 Starting Chatty AI Web Interface")
    print("=" * 60)
    print("Web Interface Features:")
    print("• Live Video Feed")
    print("• Facial Recognition") 
    print("• Real-time System Monitoring")
    print("• Speech Synthesis")
    print("• Wake Word Detection")
    print("• AI Assistant Integration")
    print("=" * 60)
    print("Access the web interface at: http://localhost:5000")
    print("Or from other devices at: http://192.168.1.16:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Run with host='0.0.0.0' to allow external connections
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Chatty AI...")
        chatty_system.stop_system()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        chatty_system.stop_system()