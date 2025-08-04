#!/bin/bash

# Chatty AI Web Interface Startup Script
# Place this script in /home/nickspi5/Chatty_AI/

echo "=========================================="
echo "  Starting Chatty AI Web Interface"
echo "=========================================="

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source chatty-venv/bin/activate

# Check if required files exist
echo "🔍 Checking required files..."

required_files=(
    "app.py"
    "templates/Chatty_AI.html"
    "templates/Chatty_AI_logo.png"
    "templates/diamond_coding_logo.png"
    "encodings.pickle"
    "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
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

if [ $missing_files -gt 0 ]; then
    echo "❌ $missing_files required files are missing. Please ensure all files are in place."
    exit 1
fi

# Check if response files exist, create if missing
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
                echo "Why don't scientists trust atoms? Because they make up everything!" > "$file"
                echo "What do you call a bear with no teeth? A gummy bear!" >> "$file"
                ;;
            "listening_responses.txt")
                echo "I'm listening, what would you like to know?" > "$file"
                echo "Yes, how can I help you?" >> "$file"
                ;;
            "waiting_responses.txt")
                echo "I'm still here if you need anything" > "$file"
                echo "Let me know if you need help" >> "$file"
                ;;
            "warning_responses.txt")
                echo "Warning: Unknown person detected. Please identify yourself." > "$file"
                ;;
            "greeting_responses.txt")
                echo "Hello! How can I help you today?" > "$file"
                echo "Welcome! What can I do for you?" >> "$file"
                ;;
        esac
    fi
done

# Set permissions for audio devices
echo "🔊 Setting up audio permissions..."
sudo usermod -a -G audio $USER

# Display network information
echo "🌐 Network Information:"
IP_ADDRESS=$(hostname -I | awk '{print $1}')
echo "   Local IP: $IP_ADDRESS"
echo "   Web Interface: http://$IP_ADDRESS:5000"
echo "   Alternative: http://localhost:5000"

echo ""
echo "🚀 Starting Chatty AI Web Server..."
echo "   Press Ctrl+C to stop the server"
echo "=========================================="

# Start the Flask application
python3 debug_app.py