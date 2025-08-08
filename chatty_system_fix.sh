#!/bin/bash
# Chatty AI System Connection Fix Script
echo "🔧 Fixing Chatty AI System Connection Issues"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this from the Chatty_AI directory"
    exit 1
fi

echo "1. Checking current system status..."
ps aux | grep -E "(python|flask)" | grep -v grep

echo -e "\n2. Checking memory usage..."
free -h
echo ""

echo "3. Checking for hanging processes..."
# Kill any hanging Python processes
pkill -f "python.*app.py"
sleep 2

echo "4. Checking system resources and models..."
echo "Memory before cleanup:"
free -m | head -2

echo -e "\nChecking models directory..."
if [ -d "models" ]; then
    echo "Models directory contents:"
    ls -la models/ | head -10
    du -sh models/ 2>/dev/null || echo "Cannot calculate models size"
else
    echo "❌ Models directory not found - this may be the issue!"
fi

echo -e "\n5. Checking for required dependencies..."
python3 -c "
try:
    import torch
    print('✅ PyTorch available:', torch.__version__)
except ImportError:
    print('❌ PyTorch not available - this may cause hanging!')

try:
    import transformers
    print('✅ Transformers available')
except ImportError:
    print('❌ Transformers not available')
    
try:
    import cv2
    print('✅ OpenCV available')
except ImportError:
    print('❌ OpenCV not available')
"

echo -e "\n6. Starting system with timeout and monitoring..."
echo "   This will restart the Flask app and monitor for hanging..."
echo ""

# Function to monitor the startup process
monitor_startup() {
    local pid=$1
    local timeout=60
    local count=0
    
    echo "Monitoring startup process (PID: $pid) with ${timeout}s timeout..."
    
    while [ $count -lt $timeout ]; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "❌ Process died unexpectedly"
            return 1
        fi
        
        # Check if Flask server is responding
        if curl -s --connect-timeout 1 http://localhost:5000 >/dev/null 2>&1; then
            echo "✅ Flask server is responding"
            return 0
        fi
        
        echo -n "."
        sleep 1
        ((count++))
    done
    
    echo "⚠️  Startup taking longer than expected..."
    return 1
}

# Start the application in background
echo "Starting python3 app.py..."
python3 app.py &
APP_PID=$!

# Monitor the startup
if monitor_startup $APP_PID; then
    echo "✅ System appears to be starting successfully"
    echo "🌐 Web interface should be available at: http://192.168.1.16:5000"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Refresh your browser page"
    echo "   2. Try clicking 'Start System' again"
    echo "   3. Watch this terminal for any error messages"
    echo ""
    echo "Press Ctrl+C to stop monitoring (app will continue running)"
    
    # Keep monitoring for issues
    while kill -0 $APP_PID 2>/dev/null; do
        sleep 5
        echo -n "."
    done
else
    echo "❌ System failed to start properly"
    echo ""
    echo "🔍 Troubleshooting suggestions:"
    echo "   1. Check if you have enough free memory (need ~1GB)"
    echo "   2. Verify all model files are present and not corrupted"
    echo "   3. Try running with: python3 app.py --debug"
    echo "   4. Check the full error log in terminal"
    
    # Kill the hanging process
    kill $APP_PID 2>/dev/null
fi