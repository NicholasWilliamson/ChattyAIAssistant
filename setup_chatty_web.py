#!/usr/bin/env python3
"""
setup_chatty_web.py - Setup script for Chatty AI Web Interface
This script will help you set up all the necessary files and dependencies
"""

import os
import subprocess
import sys
import json

def create_file_with_content(filename, content):
    """Create a file with specified content"""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {filename}: {e}")
        return False

def check_and_create_response_files():
    """Create response files if they don't exist"""
    print("📝 Setting up response files...")
    
    files_content = {
        "jokes.txt": """Why don't scientists trust atoms? Because they make up everything!
What do you call a bear with no teeth? A gummy bear!
Why did the scarecrow win an award? He was outstanding in his field!
What do you call a fake noodle? An impasta!
Why don't eggs tell jokes? They'd crack each other up!""",
        
        "listening_responses.txt": """I'm listening, what would you like to know?
Yes, how can I help you?
What can I do for you?
I'm all ears, what's on your mind?
Ready for your instructions!""",
        
        "waiting_responses.txt": """I'm still here if you need anything
Let me know if you need help
Still waiting to assist you
Your AI assistant is standing by
I'm here whenever you're ready""",
        
        "warning_responses.txt": """Warning: Unknown person detected. Please identify yourself.
Alert: Unrecognized individual detected.
Security notice: Unknown person in area.
Attention: Unauthorized person detected.""",
        
        "greeting_responses.txt": """Hello! How can I help you today?
Welcome! What can I do for you?
Hi there! Ready to assist you.
Good to see you! How may I help?
Hello! Your AI assistant is ready."""
    }
    
    for filename, content in files_content.items():
        if not os.path.exists(filename):
            create_file_with_content(filename, content)
        else:
            print(f"✅ Found existing: {filename}")

def create_personalized_responses():
    """Create personalized responses JSON file"""
    print("🤖 Setting up personalized responses...")
    
    if not os.path.exists("personalized_responses.json"):
        personalized_data = {
            "Nick": {
                "greetings": [
                    "Hello Nick, my master! It is so lovely to see you again. Thank you for creating me. How may I assist you?",
                    "Welcome back Nick! Your brilliant creation is ready to serve. What can I help you with today?",
                    "Nick! My creator has returned! I've been waiting patiently for your commands.",
                    "Master Nick! Your faithful AI assistant is here and ready to help."
                ],
                "listening": [
                    "Yes Nick, I'm listening. What would you like to know?",
                    "I'm all ears, Nick. What's on your mind?",
                    "Ready for your instructions, Nick!",
                    "Your wish is my command, Nick. What do you need?"
                ],
                "waiting": [
                    "Nick, I'm still here if you need anything",
                    "Your faithful AI assistant is standing by, Nick",
                    "Still waiting to help you, Nick",
                    "Ready when you are, Master Nick"
                ],
                "bored_responses": [
                    "Nick, I'm still here waiting to help you.",
                    "Your AI is patiently waiting for your next command, Nick.",
                    "Still here and ready to assist, Nick."
                ],
                "joke_responses": [
                    "Here's a joke for you: Why do programmers prefer dark mode? Because light attracts bugs!",
                    "Want to hear something funny? There are only 10 types of people in the world: those who understand binary and those who don't!",
                    "Here's one: Why do Java developers wear glasses? Because they don't C#!"
                ],
                "fun_fact_responses": [
                    "Did you know that the first computer bug was an actual bug found in a computer in 1947?",
                    "Fun fact: Python was named after Monty Python, not the snake!",
                    "Here's something cool: The term 'debugging' was coined by Grace Hopper!"
                ]
            }
        }
        
        with open("personalized_responses.json", 'w') as f:
            json.dump(personalized_data, f, indent=4)
        print("✅ Created: personalized_responses.json")
    else:
        print("✅ Found existing: personalized_responses.json")

def create_telegram_config():
    """Create example Telegram config"""
    print("📱 Setting up Telegram configuration...")
    
    if not os.path.exists("telegram_config.json"):
        telegram_config = {
            "bot_token": "YOUR_BOT_TOKEN_HERE",
            "chat_id": "YOUR_CHAT_ID_HERE"
        }
        
        with open("telegram_config.json", 'w') as f:
            json.dump(telegram_config, f, indent=4)
        print("✅ Created: telegram_config.json (template)")
        print("   ⚠️  Edit this file with your actual Telegram bot token and chat ID")
    else:
        print("✅ Found existing: telegram_config.json")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "templates",
        "security_photos", 
        "security_logs",
        "audio_files",
        "voices/en_US-amy-low",
        "tinyllama-models"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Found existing directory: {directory}")

def check_required_files():
    """Check for required files and give guidance"""
    print("🔍 Checking for required files...")
    
    required_files = {
        "encodings.pickle": "Face recognition encodings - you need to run encode_faces.py first",
        "templates/Chatty_AI.html": "Main web interface template",
        "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": "LLaMA model file",
        "voices/en_US-amy-low/en_US-amy-low.onnx": "Piper voice model",
        "voices/en_US-amy-low/en_US-amy-low.onnx.json": "Piper voice config",
        "piper/piper": "Piper TTS executable"
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ Found: {filename}")
        else:
            print(f"❌ Missing: {filename} - {description}")
            missing_files.append((filename, description))
    
    if missing_files:
        print("\n⚠️  Missing files detected:")
        for filename, description in missing_files:
            print(f"   - {filename}: {description}")
        
        print("\n📋 Next steps to complete setup:")
        print("   1. Download required model files")
        print("   2. Run encode_faces.py to create face encodings")
        print("   3. Ensure Piper TTS is installed")
        print("   4. Make sure HTML template exists")
    
    return len(missing_files) == 0

def create_sample_html():
    """Create a basic HTML template if it doesn't exist"""
    print("🌐 Checking web template...")
    
    html_path = "templates/Chatty_AI.html"
    if not os.path.exists(html_path):
        print("📄 Creating basic HTML template...")
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatty AI Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #333; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .video-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .logs { background: white; padding: 20px; border-radius: 8px; height: 300px; overflow-y: auto; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        #video-feed { max-width: 100%; border: 2px solid #ddd; border-radius: 8px; }
        .log-entry { margin: 5px 0; padding: 5px; border-left: 3px solid #ccc; }
        .log-info { border-left-color: #2196F3; }
        .log-error { border-left-color: #f44336; }
        .log-warning { border-left-color: #FF9800; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Chatty AI Web Interface</h1>
            <p>AI Assistant with Face Recognition and Voice Interaction</p>
        </div>
        
        <div class="controls">
            <h2>System Controls</h2>
            <button id="start-btn" class="start-btn" onclick="startSystem()">Start System</button>
            <button id="stop-btn" class="stop-btn" onclick="stopSystem()">Stop System</button>
            <span id="status">Disconnected</span>
        </div>
        
        <div class="video-section">
            <h2>Camera Feed</h2>
            <img id="video-feed" src="/video_feed" alt="Camera feed will appear here">
        </div>
        
        <div class="logs">
            <h2>System Logs</h2>
            <div id="log-content"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('status').textContent = 'Connected';
        });
        
        socket.on('log_update', function(data) {
            const logContent = document.getElementById('log-content');
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${data.type}`;
            logEntry.innerHTML = `<strong>[${data.timestamp}]</strong> ${data.message}`;
            logContent.appendChild(logEntry);
            logContent.scrollTop = logContent.scrollHeight;
        });
        
        function startSystem() {
            socket.emit('start_system');
        }
        
        function stopSystem() {
            socket.emit('stop_system');
        }
    </script>
</body>
</html>'''
        
        create_file_with_content(html_path, html_content)
    else:
        print("✅ Found existing: templates/Chatty_AI.html")

def main():
    """Main setup function"""
    print("🚀 Chatty AI Web Interface Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("⚠️  Warning: app.py not found in current directory")
        print("Please run this setup script from your Chatty AI directory")
        return
    
    # Create directories
    create_directories()
    
    # Create response files
    check_and_create_response_files()
    
    # Create personalized responses
    create_personalized_responses()
    
    # Create Telegram config
    create_telegram_config()
    
    # Create basic HTML template
    create_sample_html()
    
    # Check for required files
    all_files_present = check_required_files()
    
    print("\n" + "=" * 50)
    if all_files_present:
        print("🎉 Setup complete! All required files are present.")
        print("✅ You can now run: python3 app.py")
    else:
        print("⚠️  Setup partially complete. Some files are still missing.")
        print("📋 Please address the missing files listed above.")
    
    print("\n📚 Additional setup notes:")
    print("   - Edit telegram_config.json with your bot details")
    print("   - Run encode_faces.py to create face recognition data")
    print("   - Make sure your camera is enabled and working")
    print("   - Test camera with: python3 test_camera.py")
    
    print("=" * 50)

if __name__ == "__main__":
    main()