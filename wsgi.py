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

# For Flask-SocketIO with Gunicorn, we need the socketio instance
# but we need to make sure it's properly configured as a WSGI app
try:
    # Method 1: Try to get the WSGI app from socketio
    app = chatty_server.socketio.wsgi_app
except AttributeError:
    # Method 2: Use the socketio instance directly (for newer versions)
    app = chatty_server.socketio

if __name__ == "__main__":
    # Direct execution (development mode)
    chatty_server.run(debug=False)