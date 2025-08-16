#!/usr/bin/env python3
"""
wsgi.py - WSGI Application Entry Point for Chatty AI
This file is used by Gunicorn to serve the application
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the correct module (chatty_ai, not chatty_ai_web)
from chatty_ai import ChattyAIWebServer

# Create the application instance
chatty_server = ChattyAIWebServer()

# WSGI application
app = chatty_server.app
socketio = chatty_server.socketio

# For Gunicorn with eventlet worker
if __name__ != "__main__":
    # When running under Gunicorn, we need to use the socketio app
    app = socketio

if __name__ == "__main__":
    # Direct execution (development mode)
    chatty_server.run(debug=False)