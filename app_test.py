#!/usr/bin/env python3
"""
Chatty AI Web Interface - Complete Fixed Version
"""

from flask import Flask, render_template_string, jsonify, request
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

# HTML Template embedded in the Python file
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatty AI - Web Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .panel h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
            color: #fff;
        }
        
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        #videoFeed {
            max-width: 100%;
            max-height: 100%;
            border-radius: 10px;
        }
        
        .video-placeholder {
            color: #888;
            font-size: 1.1em;
            text-align: center;
        }
        
        .controls {
            margin-top: 15px;
            text-align: center;
        }
        
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .btn.stop {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-online {
            background: #4CAF50;
        }
        
        .status-offline {
            background: #f44336;
        }
        
        .status-connecting {
            background: #ff9800;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .system-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .info-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .info-item .label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .info-item .value {
            font-size: 1.4em;
            font-weight: bold;
        }
        
        .logs {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .log-entry {
            margin-bottom: 5px;
            padding: 2px 0;
        }
        
        .log-info { color: #4CAF50; }
        .log-error { color: #f44336; }
        .log-warning { color: #ff9800; }
        .log-system { color: #2196F3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Chatty AI</h1>
            <p>Advanced AI Assistant with Real-time Monitoring</p>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h3>📹 Live Video Feed</h3>
                <div class="video-container">
                    <img id="videoFeed" style="display: none;" alt="Video Feed">
                    <div id="videoPlaceholder" class="video-placeholder">
                        Camera feed will appear here when system is started
                    </div>
                </div>
                <div class="controls">
                    <button id="startBtn" class="btn">🚀 Start System</button>
                    <button id="stopBtn" class="btn stop" disabled>🛑 Stop System</button>
                </div>
            </div>
            
            <div class="panel">
                <h3>📊 System Information</h3>
                <div class="system-info">
                    <div class="info-item">
                        <div class="label">CPU Usage</div>
                        <div class="value" id="cpuUsage">--</div>
                    </div>
                    <div class="info-item">
                        <div class="label">Memory Usage</div>
                        <div class="value" id="memoryUsage">--</div>
                    </div>
                    <div class="info-item">
                        <div class="label">CPU Temperature</div>
                        <div class="value" id="cpuTemp">--</div>
                    </div>
                    <div class="info-item">
                        <div class="label">GPU Temperature</div>
                        <div class="value" id="gpuTemp">--</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h3>
                <span id="statusIndicator" class="status-indicator status-connecting"></span>
                System Status: <span id="systemStatus">Connecting...</span>
            </h3>
            <div id="systemLogs" class="logs"></div>
        </div>
    </div>

    <script>
        class ChattyAI {
            constructor() {
                this.socket = null;
                this.isConnected = false;
                this.systemRunning = false;
                this.init();
            }
            
            init() {
                this.initElements();
                this.connect();
                this.setupEventListeners();
            }
            
            initElements() {
                this.startBtn = document.getElementById('startBtn');
                this.stopBtn = document.getElementById('stopBtn');
                this.videoFeed = document.getElementById('videoFeed');
                this.videoPlaceholder = document.getElementById('videoPlaceholder');
                this.systemStatus = document.getElementById('systemStatus');
                this.statusIndicator = document.getElementById('statusIndicator');
                this.systemLogs = document.getElementById('systemLogs');
                this.cpuUsage = document.getElementById('cpuUsage');
                this.memoryUsage = document.getElementById('memoryUsage');
                this.cpuTemp = document.getElementById('cpuTemp');
                this.gpuTemp = document.getElementById('gpuTemp');
            }
            
            connect() {
                this.log('🔗 Attempting to connect to server...', 'system');
                
                this.socket = io({
                    transports: ['websocket', 'polling'],
                    timeout: 20000,
                    forceNew: true
                });
                
                this.socket.on('connect', () => {
                    this.isConnected = true;
                    this.updateStatus('Connected', 'online');
                    this.log('✅ Connected to server successfully', 'info');
                });
                
                this.socket.on('disconnect', () => {
                    this.isConnected = false;
                    this.updateStatus('Disconnected', 'offline');
                    this.log('❌ Connection lost - attempting to reconnect...', 'warning');
                });
                
                this.socket.on('connect_error', (error) => {
                    this.log(`❌ Connection error: ${error.message}`, 'error');
                    this.updateStatus('Connection Error', 'offline');
                });
                
                this.socket.on('status_update', (data) => {
                    this.systemRunning = data.is_running;
                    this.updateButtons();
                    this.log(`📊 ${data.message}`, 'info');
                });
                
                this.socket.on('system_response', (data) => {
                    if (data.success) {
                        this.log(`✅ ${data.message}`, 'info');
                    } else {
                        this.log(`❌ ${data.message}`, 'error');
                    }
                });
                
                this.socket.on('video_frame', (data) => {
                    if (data.frame) {
                        this.videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
                        this.videoFeed.style.display = 'block';
                        this.videoPlaceholder.style.display = 'none';
                    }
                });
                
                this.socket.on('system_info', (data) => {
                    this.updateSystemInfo(data);
                });
                
                this.socket.on('system_error', (data) => {
                    this.log(`❌ ${data.message}`, 'error');
                });
            }
            
            setupEventListeners() {
                this.startBtn.addEventListener('click', () => {
                    if (this.isConnected) {
                        this.log('🚀 Starting system...', 'system');
                        this.socket.emit('start_system');
                    } else {
                        this.log('❌ Cannot start system - not connected to server', 'error');
                    }
                });
                
                this.stopBtn.addEventListener('click', () => {
                    if (this.isConnected) {
                        this.log('🛑 Stopping system...', 'system');
                        this.socket.emit('stop_system');
                        this.videoFeed.style.display = 'none';
                        this.videoPlaceholder.style.display = 'block';
                    } else {
                        this.log('❌ Cannot stop system - not connected to server', 'error');
                    }
                });
            }
            
            updateStatus(status, type) {
                this.systemStatus.textContent = status;
                this.statusIndicator.className = `status-indicator status-${type}`;
            }
            
            updateButtons() {
                this.startBtn.disabled = !this.isConnected || this.systemRunning;
                this.stopBtn.disabled = !this.isConnected || !this.systemRunning;
            }
            
            updateSystemInfo(data) {
                this.cpuUsage.textContent = `${data.cpu_percent.toFixed(1)}%`;
                this.memoryUsage.textContent = `${data.memory_percent.toFixed(1)}%`;
                this.cpuTemp.textContent = `${data.cpu_temp.toFixed(1)}°C`;
                this.gpuTemp.textContent = `${data.gpu_temp.toFixed(1)}°C`;
            }
            
            log(message, type = 'info') {
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${type}`;
                logEntry.textContent = `${timestamp} ${message}`;
                
                this.systemLogs.appendChild(logEntry);
                this.systemLogs.scrollTop = this.systemLogs.scrollHeight;
                
                // Keep only last 50 log entries
                while (this.systemLogs.children.length > 50) {
                    this.systemLogs.removeChild(this.systemLogs.firstChild);
                }
            }
        }
        
        // Initialize the application when page loads
        document.addEventListener('DOMContentLoaded', () => {
            window.chattyAI = new ChattyAI();
            
            // Initial log message
            setTimeout(() => {
                window.chattyAI.log('🌟 Chatty AI Web Interface initialized', 'system');
                window.chattyAI.log('💡 Click "Start System" to begin AI operations', 'info');
            }, 500);
        });
    </script>
</body>
</html>
"""

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
    return render_template_string(HTML_TEMPLATE)

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