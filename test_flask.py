#!/usr/bin/env python3
"""
Minimal Flask test to verify server functionality
Run this to test if Flask-SocketIO works on your Pi
"""

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "<h1>Flask Test Server Running!</h1><p>Port 5000 is working correctly.</p>"

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    print("🚀 Starting minimal Flask test server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)