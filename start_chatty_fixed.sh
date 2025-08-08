#!/bin/bash
# Start Chatty AI with proper model paths
echo "🚀 Starting Chatty AI with Fixed Configuration"
echo "=============================================="

# Ensure we're in the right directory
cd ~/Chatty_AI || { echo "❌ Cannot find Chatty_AI directory"; exit 1; }

# Activate virtual environment
source chatty-venv/bin/activate || { echo "❌ Cannot activate virtual environment"; exit 1; }

echo "1. Stopping any existing processes..."
pkill -f python
sleep 2

echo "2. Verifying model setup..."
if [ -L "models" ] && [ -d "tinyllama-models" ]; then
    echo "✅ Model directory setup correct"
elif [ -d "tinyllama-models" ] && [ ! -e "models" ]; then
    echo "Creating models symlink..."
    ln -s tinyllama-models models
    echo "✅ Created models -> tinyllama-models symlink"
else
    echo "⚠️  Model directory setup may need attention"
fi

echo "3. Checking model file..."
if [ -f "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf" ]; then
    echo "✅ TinyLlama model file found"
else
    echo "❌ TinyLlama model file missing!"
fi

echo "4. Checking system resources..."
echo "Available memory: $(free -h | awk 'NR==2{print $7}')"

echo "5. Starting Chatty AI..."
echo "   🌐 Web interface will be at: http://192.168.1.16:5000"
echo "   📱 Click 'Start System' after the page loads"
echo "   💡 Watch this terminal for startup messages"
echo ""

# Start with unbuffered output so we can see progress
python3 -u app.py

echo ""
echo "Application stopped. Check above for any error messages."