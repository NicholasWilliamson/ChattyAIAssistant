#!/usr/bin/env python3
"""
wsgi.py - WSGI Application Entry Point for Chatty AI
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch the async_mode before importing
import chatty_ai
# Monkey patch to change async_mode before class instantiation
original_init = chatty_ai.ChattyAIWebServer.__init__

def patched_init(self):
    original_init(self)
    # Change async_mode to eventlet after initialization
    self.socketio.async_mode = 'eventlet'

chatty_ai.ChattyAIWebServer.__init__ = patched_init

# Now import and create the server
from chatty_ai import ChattyAIWebServer

chatty_server = ChattyAIWebServer()
app = chatty_server.socketio

if __name__ == "__main__":
    chatty_server.run(debug=False)