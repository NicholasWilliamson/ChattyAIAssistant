#!/usr/bin/env python3
"""
wsgi.py - WSGI Application Entry Point for Chatty AI
"""
from chatty_ai import ChattyAIWebServer

# Create the server instance
server = ChattyAIWebServer()

# Get the Flask app and SocketIO instance for Gunicorn
app = server.app
socketio = server.socketio

if __name__ == "__main__":
    # For direct running (not through Gunicorn)
    server.run(host='0.0.0.0', port=5000, debug=False)
