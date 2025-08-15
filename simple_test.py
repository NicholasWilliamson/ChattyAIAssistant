#!/usr/bin/env python3
"""
Simple Socket.IO test server to diagnose connection issues
"""

import logging
import psutil
from datetime import datetime
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='templates')
app.config['SECRET_KEY'] = 'test_key_123'

# Create SocketIO with minimal configuration
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   ping_timeout=30,
                   ping_interval=10)

# Global variables
system_running = False
connected_clients = set()

def get_system_info():
    """Get basic system info"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Try to get temperature
        try:
            import subprocess
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                cpu_temp = float(temp_str[5:-2]) if temp_str.startswith('temp=') else 50.0
            else:
                cpu_temp = 50.0
        except:
            cpu_temp = 50.0
            
        return {
            'cpu_percent': round(cpu_percent, 1),
            'memory_percent': round(memory.percent, 1),
            'memory_used': memory.used,
            'memory_total': memory.total,
            'cpu_temp': round(cpu_temp, 1),
            'gpu_temp': round(cpu_temp + 1, 1),
            'timestamp': datetime.now().isoformat()
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
            'timestamp': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('Chatty_AI.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return f"Template error: {e}. Make sure Chatty_AI.html is in the templates folder.", 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients.add(client_id)
    logger.info(f"Client connected: {client_id}. Total: {len(connected_clients)}")
    
    # Send welcome message
    emit('status_update', {
        'status': 'running' if system_running else 'stopped',
        'is_running': system_running,
        'message': 'Connected to server successfully'
    })
    
    # Send initial system info
    emit('system_info', get_system_info())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    connected_clients.discard(client_id)
    logger.info(f"Client disconnected: {client_id}. Total: {len(connected_clients)}")

@socketio.on('start_system')
def handle_start_system():
    """Handle start system request"""
    global system_running
    client_id = request.sid
    logger.info(f"Start system requested by: {client_id}")
    
    try:
        system_running = True
        
        # Send response to all clients
        socketio.emit('status_update', {
            'status': 'running',
            'is_running': True,
            'message': 'Chatty AI system started successfully'
        })
        
        logger.info("System started - response sent to all clients")
        
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        emit('status_update', {
            'status': 'error',
            'is_running': False,
            'message': f'Error: {e}'
        })

@socketio.on('stop_system')
def handle_stop_system():
    """Handle stop system request"""
    global system_running
    client_id = request.sid
    logger.info(f"Stop system requested by: {client_id}")
    
    try:
        system_running = False
        
        # Send response to all clients
        socketio.emit('status_update', {
            'status': 'stopped',
            'is_running': False,
            'message': 'Chatty AI system stopped'
        })
        
        logger.info("System stopped - response sent to all clients")
        
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        emit('status_update', {
            'status': 'error',
            'is_running': system_running,
            'message': f'Error: {e}'
        })

@socketio.on('get_system_info')
def handle_get_system_info():
    """Handle system info request"""
    try:
        system_info = get_system_info()
        emit('system_info', system_info)
        logger.info("System info sent to client")
    except Exception as e:
        logger.error(f"Error getting system info: {e}")

@socketio.on_error_default
def default_error_handler(e):
    """Default error handler"""
    logger.error(f"Socket.IO error: {e}")

if __name__ == '__main__':
    print("="*60)
    print("🧪 Simple Socket.IO Test Server")
    print("="*60)
    print("Testing Socket.IO connection...")
    print("Access at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    try:
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=True,
                    use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")