#!/usr/bin/env python3
"""
wsgi.py - WSGI Application Entry Point for Chatty AI
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatty_ai import ChattyAIWebServer

# Create the server instance
chatty_server = ChattyAIWebServer()

# The WSGI application should be the Flask app, not the SocketIO object
# Flask-SocketIO patches the Flask app to handle Socket.IO requests
app = chatty_server.app

if __name__ == "__main__":
    chatty_server.run(debug=False)