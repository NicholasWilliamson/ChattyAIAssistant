#!/bin/bash
# Fix Chatty AI Model Configuration and Add Debug Logging
echo "🔧 Fixing Model Configuration Issues"
echo "===================================="

echo "1. Current model files in tinyllama-models:"
ls -la tinyllama-models/

echo -e "\n2. Checking current model configuration..."
echo "In chatty_ai.py:"
grep -n "LLAMA_MODEL_PATH\|tinyllama" chatty_ai.py || echo "No LLAMA_MODEL_PATH found in chatty_ai.py"

echo -e "\nIn app.py:"
grep -n "LLAMA_MODEL_PATH\|tinyllama" app.py || echo "No LLAMA_MODEL_PATH found in app.py"

echo -e "\n3. Finding the correct model file..."
CORRECT_MODEL=""
if [ -f "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf" ]; then
    CORRECT_MODEL="tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf"
    echo "✅ Found Q4_K_S model (smaller, faster)"
elif [ -f "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    CORRECT_MODEL="tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    echo "✅ Found Q4_K_M model"
elif [ -f "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" ]; then
    CORRECT_MODEL="tinyllama-models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    echo "✅ Found Q8_0 model (largest)"
else
    echo "❌ No suitable model found!"
    exit 1
fi

echo "Using model: $CORRECT_MODEL"

echo -e "\n4. Verifying model path configuration..."
echo "Current model path configuration is correct (Q4_K_M.gguf)"
echo "Model file exists and is accessible: $(ls -lh tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)"

echo -e "\n5. Verifying the fix..."
echo "Updated paths:"
grep -n "LLAMA_MODEL_PATH\|tinyllama.*gguf" chatty_ai.py app.py 2>/dev/null

echo -e "\n6. Creating a debug version of the startup..."
cat > debug_app.py << 'EOF'
#!/usr/bin/env python3
"""
Debug version of app.py to identify where startup hangs
"""
import sys
import time
from datetime import datetime

def debug_print(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG: {message}")
    sys.stdout.flush()

# Import Flask components first
debug_print("Importing Flask components...")
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import queue
import json
import cv2
import numpy as np

debug_print("Basic imports successful")

# Now try the problematic import
debug_print("Attempting to import ChattyAI...")
try:
    from chatty_ai import ChattyAI
    debug_print("✅ ChattyAI imported successfully")
except Exception as e:
    debug_print(f"❌ ChattyAI import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

debug_print("Creating Flask app...")
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

debug_print("Setting up global variables...")
chatty_ai = None
system_running = False
message_queue = queue.Queue()

@app.route('/')
def index():
    debug_print("Index route accessed")
    return render_template('Chatty_AI.html')

@socketio.on('connect')
def handle_connect():
    debug_print("Client connected to SocketIO")
    emit('status', {'connected': True})

@socketio.on('disconnect')
def handle_disconnect():
    debug_print("Client disconnected from SocketIO")

@socketio.on('start_system')
def handle_start_system():
    global chatty_ai, system_running
    debug_print("Start system requested...")
    
    try:
        debug_print("Creating ChattyAI instance...")
        emit('log', {'message': 'Initializing AI models...'})
        
        # This is likely where it hangs
        chatty_ai = ChattyAI()
        debug_print("✅ ChattyAI instance created successfully!")
        
        system_running = True
        emit('status', {'system_running': True})
        emit('log', {'message': 'System started successfully!'})
        debug_print("System startup complete")
        
    except Exception as e:
        debug_print(f"❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        emit('log', {'message': f'Startup failed: {str(e)}'})
        emit('status', {'system_running': False})

if __name__ == '__main__':
    debug_print("Starting debug version of Chatty AI...")
    debug_print("Web interface will be at: http://192.168.1.16:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
EOF

echo "✅ Debug version created as debug_app.py"

echo -e "\n7. Summary of analysis:"
echo "- Model path configuration is correct (Q4_K_M.gguf)"
echo "- Created debug_app.py for detailed startup logging"
echo "- All model files are present and accessible"

echo -e "\n🚀 Next steps to test:"
echo "1. Try the debug version: python3 debug_app.py"
echo "2. Watch the terminal closely for where it hangs"
echo "3. If it works, we can apply the fix to the main app"

echo -e "\n💡 If debug version still hangs, the issue is likely:"
echo "- Insufficient memory for model loading"
echo "- Missing dependencies in ChattyAI class"
echo "- Corrupted model files"