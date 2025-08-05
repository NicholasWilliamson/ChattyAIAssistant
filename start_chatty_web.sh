#!/bin/bash

# Chatty AI Web Interface Startup Script
# Place this script in /home/nickspi5/Chatty_AI/

echo "=========================================="
echo "  Starting Chatty AI Web Interface"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from /home/nickspi5/Chatty_AI/"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [ -d "chatty-venv" ]; then
    source chatty-venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment 'chatty-venv' not found!"
    echo "Please create it first with: python3 -m venv chatty-venv"
    exit 1
fi

# Check if required files exist
echo "🔍 Checking required files..."

required_files=(
    "app.py"
    "templates/Chatty_AI.html"
    "encodings.pickle"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        missing_files=$((missing_files + 1))
    else
        echo "✅ Found: $file"
    fi
done

# Check optional files  
optional_files=(
    "templates/Chatty_AI_logo.png"
    "templates/diamond_coding_logo.png"
    "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

for file in "${optional_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Optional file missing: $file"
    else
        echo "✅ Found: $file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "❌ $missing_files required files are missing."
    echo "The web interface may not work properly."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if response files exist, create if missing
echo ""
echo "📝 Checking response files..."
response_files=(
    "jokes.txt"
    "listening_responses.txt"
    "waiting_responses.txt"
    "warning_responses.txt"
    "greeting_responses.txt"
)

for file in "${response_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  Creating missing response file: $file"
        case $file in
            "jokes.txt")
                cat > "$file" << 'EOF'
Why don't scientists trust atoms? Because they make up everything!
What do you call a bear with no teeth? A gummy bear!
Why did the scarecrow win an award? He was outstanding in his field!
EOF
                ;;
            "listening_responses.txt")
                cat > "$file" << 'EOF'
I'm listening, what would you like to know?
Yes, how can I help you?
What can I do for you?
EOF
                ;;
            "waiting_responses.txt")
                cat > "$file" << 'EOF'
I'm still here if you need anything
Let me know if you need help
Still waiting to assist you
EOF
                ;;
            "warning_responses.txt")
                cat > "$file" << 'EOF'
Warning: Unknown person detected. Please identify yourself.
Alert: Unrecognized individual detected.
Security notice: Unknown person in area.
EOF
                ;;
            "greeting_responses.txt")
                cat > "$file" << 'EOF'
Hello! How can I help you today?
Welcome! What can I do for you?
Hi there! Ready to assist you.
EOF
                ;;
        esac
        echo "✅ Created: $file"
    else
        echo "✅ Found: $file"
    fi
done

# Check and kill any existing camera processes
echo ""
echo "🔧 Checking for camera conflicts..."
camera_processes=$(pgrep -f "libcamera\|picamera2\|python.*camera" || true)
if [ ! -z "$camera_processes" ]; then
    echo "⚠️  Found existing camera processes. Attempting to stop them..."
    pkill -f "libcamera\|picamera2" || true
    sleep 2
    echo "✅ Camera processes cleared"
else
    echo "✅ No camera conflicts detected"
fi

# Set permissions for audio devices
echo "🔊 Setting up audio permissions..."
if groups $USER | grep -q "\baudio\b"; then
    echo "✅ User already in audio group"
else
    echo "Adding user to audio group..."
    sudo usermod -a -G audio $USER
    echo "⚠️  You may need to log out and back in for audio group changes to take effect"
fi

# Display network information
echo ""
echo "🌐 Network Information:"
IP_ADDRESS=$(hostname -I | awk '{print $1}')
if [ ! -z "$IP_ADDRESS" ]; then
    echo "   Local IP: $IP_ADDRESS"
    echo "   Web Interface: http://$IP_ADDRESS:5000"
else
    echo "   Could not determine IP address"
fi
echo "   Localhost: http://localhost:5000"
echo "   Alternative: http://127.0.0.1:5000"

echo ""
echo "🚀 Starting Chatty AI Web Server..."
echo "   Press Ctrl+C to stop the server"
echo "   View logs above for any startup issues"
echo "=========================================="

# Start the Flask application
python3 app.py