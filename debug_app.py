#!/usr/bin/env python3
"""
debug_app.py - Debug version of Chatty AI Web Application
This version includes extensive debugging to identify issues
"""

import os
import sys
import traceback
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
from datetime import datetime
import threading

# Add current directory to path
sys.path.append('/home/nickspi5/Chatty_AI')

# Flask application setup
app = Flask(__name__, 
            template_folder='/home/nickspi5/Chatty_AI/templates',
            static_folder='/home/nickspi5/Chatty_AI/templates')
app.config['SECRET_KEY'] = 'chatty_ai_debug_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class DebugChattyAI:
    def __init__(self):
        print("🔧 Initializing DebugChattyAI...")
        self.is_running = False
        self.picam2 = None
        self.current_frame = None
        self.captured_image = None
        self.setup_camera()

    def emit_log(self, message, log_type="info"):
        """Emit log message to web interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {log_type.upper()}: {message}")
        try:
            socketio.emit('log_update', {
                'timestamp': timestamp,
                'message': message,
                'type': log_type
            })
        except Exception as e:
            print(f"Failed to emit log: {e}")

    def setup_camera(self):
        """Initialize camera with detailed debugging"""
        try:
            print("🎥 Setting up camera...")
            
            # Check if we can import Picamera2
            try:
                from picamera2 import Picamera2
                print("✅ Picamera2 imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import Picamera2: {e}")
                return False
            
            # Try to initialize camera
            self.picam2 = Picamera2()
            print("✅ Picamera2 instance created")
            
            # Configure camera
            config = self.picam2.create_preview_configuration(
                main={"format": 'XRGB8888', "size": (640, 480)}
            )
            self.picam2.configure(config)
            print("✅ Camera configured")
            
            # Start camera
            self.picam2.start()
            print("✅ Camera started")
            
            # Test capture
            time.sleep(2)
            test_frame = self.picam2.capture_array()
            print(f"✅ Test capture successful - Frame shape: {test_frame.shape}")
            
            self.emit_log("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            self.emit_log(f"Camera setup failed: {e}", "error")
            return False

    def generate_video_feed(self):
        """Generate video frames for streaming with debugging"""
        print("🎬 Starting video feed generation...")
        
        frame_count = 0
        while True:
            try:
                frame_count += 1
                
                if self.picam2:
                    # Capture frame
                    frame = self.picam2.capture_array()
                    
                    if frame_count % 30 == 0:  # Log every 30th frame
                        print(f"📸 Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")
                    
                    # Convert color space if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    
                    # Add debug overlay
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(frame, f"Debug Mode - {timestamp}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Shape: {frame.shape}", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Store current frame
                    self.current_frame = frame
                    
                else:
                    # Create test pattern if no camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "NO CAMERA AVAILABLE", (200, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print(f"❌ Failed to encode frame {frame_count}")
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"❌ Video feed error at frame {frame_count}: {e}")
                print(f"❌ Traceback: {traceback.format_exc()}")
                
                # Create error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"ERROR: {str(e)[:30]}", (50, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)

    def start_system(self):
        """Start the debug system"""
        self.emit_log("Debug system started")
        self.is_running = True
        return True

    def stop_system(self):
        """Stop the debug system"""
        self.emit_log("Debug system stopped")
        self.is_running = False

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up debug system...")
        self.is_running = False
        if self.picam2:
            try:
                self.picam2.stop()
                print("✅ Camera stopped")
            except Exception as e:
                print(f"⚠️  Error stopping camera: {e}")

# Create global instance
debug_chatty = DebugChattyAI()

# Flask routes
@app.route('/')
def index():
    """Main page"""
    try:
        print("📄 Serving main page...")
        return render_template('Chatty_AI.html')
    except Exception as e:
        print(f"❌ Error serving main page: {e}")
        return f"Error loading page: {e}"

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files from templates directory"""
    try:
        print(f"📁 Serving static file: {filename}")
        file_path = f"/home/nickspi5/Chatty_AI/templates/{filename}"
        
        if os.path.exists(file_path):
            print(f"✅ File found: {file_path}")
            return app.send_static_file(filename)
        else:
            print(f"❌ File not found: {file_path}")
            return f"File not found: {filename}", 404
    except Exception as e:
        print(f"❌ Error serving static file {filename}: {e}")
        return f"Error: {e}", 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        print("🎥 Starting video feed...")
        return Response(debug_chatty.generate_video_feed(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"❌ Video feed error: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return f"Video feed error: {e}", 500

@app.route('/captured_image')
def captured_image():
    """Route for captured person image"""
    try:
        if debug_chatty.captured_image is not None:
            ret, buffer = cv2.imencode('.jpg', debug_chatty.captured_image)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
        
        # Return placeholder image
        placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        ret, buffer = cv2.imencode('.jpg', placeholder)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
        
    except Exception as e:
        print(f"❌ Captured image error: {e}")
        return "Error", 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('🔌 Client connected')
    emit('status', {'message': 'Connected to Debug Chatty AI'})

@socketio.on('start_system')
def handle_start_system():
    """Handle system start request"""
    print("🚀 Start system requested")
    success = debug_chatty.start_system()
    emit('system_status', {'running': success})

@socketio.on('stop_system')
def handle_stop_system():
    """Handle system stop request"""
    print("🛑 Stop system requested")
    debug_chatty.stop_system()
    emit('system_status', {'running': False})

if __name__ == '__main__':
    print("🐛 Chatty AI Debug Mode")
    print("=" * 50)
    
    # System checks
    print("🔍 Running system checks...")
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Check working directory
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if templates directory exists
    templates_dir = "/home/nickspi5/Chatty_AI/templates"
    if os.path.exists(templates_dir):
        print(f"✅ Templates directory found: {templates_dir}")
        files = os.listdir(templates_dir)
        print(f"📄 Files in templates: {files}")
    else:
        print(f"❌ Templates directory not found: {templates_dir}")
    
    # Check camera permissions
    try:
        import picamera2
        print("✅ Picamera2 module available")
    except ImportError as e:
        print(f"❌ Picamera2 import error: {e}")
    
    # Check OpenCV
    print(f"👁️ OpenCV version: {cv2.__version__}")
    
    # Check if running as root/sudo
    if os.geteuid() == 0:
        print("⚠️  Running as root - this may cause issues")
    else:
        print(f"👤 Running as user: {os.getenv('USER')}")
    
    print("=" * 50)
    
    try:
        print("🌐 Starting debug Flask server...")
        print(f"🔗 Access at: http://localhost:5000")
        
        # Start server
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Debug server stopped by user")
        debug_chatty.cleanup()
    except Exception as e:
        print(f"❌ Debug server error: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        debug_chatty.cleanup()