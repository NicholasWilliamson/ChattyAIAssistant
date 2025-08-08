#!/usr/bin/env python3
"""
Debug script to identify where Chatty AI startup is hanging
"""
import sys
import time
import traceback
from datetime import datetime

def debug_print(message):
    """Print debug message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG: {message}")
    sys.stdout.flush()

def test_imports():
    """Test all required imports"""
    debug_print("Testing imports...")
    
    try:
        import flask
        debug_print("✅ Flask imported")
    except ImportError as e:
        debug_print(f"❌ Flask import failed: {e}")
        return False
        
    try:
        import flask_socketio
        debug_print("✅ Flask-SocketIO imported")
    except ImportError as e:
        debug_print(f"❌ Flask-SocketIO import failed: {e}")
        return False
    
    try:
        import cv2
        debug_print("✅ OpenCV imported")
    except ImportError as e:
        debug_print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        debug_print("✅ NumPy imported")
    except ImportError as e:
        debug_print(f"❌ NumPy import failed: {e}")
        return False
    
    # Test AI-related imports (these might cause hanging)
    try:
        debug_print("Testing PyTorch import (this may take time)...")
        import torch
        debug_print(f"✅ PyTorch imported - version: {torch.__version__}")
    except ImportError as e:
        debug_print(f"❌ PyTorch import failed: {e}")
    except Exception as e:
        debug_print(f"❌ PyTorch import error: {e}")
    
    try:
        debug_print("Testing Transformers import (this may take time)...")
        import transformers
        debug_print("✅ Transformers imported")
    except ImportError as e:
        debug_print(f"❌ Transformers import failed: {e}")
    except Exception as e:
        debug_print(f"❌ Transformers import error: {e}")
    
    return True

def check_memory():
    """Check available memory"""
    debug_print("Checking system memory...")
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        
        mem_total = 0
        mem_free = 0
        mem_available = 0
        
        for line in lines:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1])
            elif line.startswith('MemFree:'):
                mem_free = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1])
        
        debug_print(f"Memory Total: {mem_total/1024:.1f} MB")
        debug_print(f"Memory Free: {mem_free/1024:.1f} MB")
        debug_print(f"Memory Available: {mem_available/1024:.1f} MB")
        
        if mem_available < 500000:  # Less than 500MB
            debug_print("⚠️  WARNING: Low memory available - may cause hanging!")
            return False
            
        return True
        
    except Exception as e:
        debug_print(f"❌ Could not check memory: {e}")
        return False

def test_model_loading():
    """Test if model loading is the issue"""
    debug_print("Testing model directory...")
    
    import os
    if not os.path.exists('models'):
        debug_print("❌ Models directory not found!")
        return False
    
    model_files = os.listdir('models')
    debug_print(f"Found {len(model_files)} files in models directory")
    
    for file in model_files[:5]:  # Show first 5 files
        file_path = os.path.join('models', file)
        try:
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            debug_print(f"  - {file}: {size:.1f} MB")
        except:
            debug_print(f"  - {file}: Could not get size")
    
    return True

def test_camera():
    """Test camera initialization"""
    debug_print("Testing camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            debug_print("✅ Camera accessible")
            cap.release()
            return True
        else:
            debug_print("❌ Camera not accessible")
            return False
    except Exception as e:
        debug_print(f"❌ Camera test failed: {e}")
        return False

def main():
    debug_print("Starting Chatty AI diagnostic...")
    debug_print("=" * 50)
    
    try:
        # Test 1: Check memory
        if not check_memory():
            debug_print("Memory check failed - this may cause issues")
        
        # Test 2: Test imports
        if not test_imports():
            debug_print("Import test failed")
            return
        
        # Test 3: Test models
        if not test_model_loading():
            debug_print("Model loading test failed")
        
        # Test 4: Test camera
        if not test_camera():
            debug_print("Camera test failed")
        
        debug_print("=" * 50)
        debug_print("Diagnostic complete. Now attempting to import main application...")
        
        # Try to import the main ChattyAI class to see where it hangs
        debug_print("Importing main application modules...")
        sys.path.append('.')
        
        try:
            debug_print("Attempting to import ChattyAI class...")
            # This is likely where it's hanging - we'll see exactly where
            from chatty_ai import ChattyAI
            debug_print("✅ ChattyAI imported successfully!")
            
            debug_print("Attempting to create ChattyAI instance...")
            chatty = ChattyAI()
            debug_print("✅ ChattyAI instance created!")
            
        except Exception as e:
            debug_print(f"❌ ChattyAI import/creation failed: {e}")
            debug_print("Traceback:")
            traceback.print_exc()
    
    except KeyboardInterrupt:
        debug_print("Diagnostic interrupted by user")
    except Exception as e:
        debug_print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()