#!/usr/bin/env python3
"""
wsgi.py - WSGI Application Entry Point for Chatty AI
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import app and socketio (defined at module level in chatty_ai.py)
from chatty_ai import app, socketio, FastChattyAI

# Create the FastChattyAI instance to initialize everything
chatty = FastChattyAI()

if __name__ == "__main__":
    # Run with Flask-SocketIO's built-in server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)