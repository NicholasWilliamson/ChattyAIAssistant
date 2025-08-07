#!/bin/bash

echo "🔧 Fixing Chatty AI Web Interface Setup"
echo "===================================================="

# Navigate to Chatty AI directory
cd /home/nickspi5/Chatty_AI

# 1. Fix HTML template naming
echo "📝 Fixing HTML template file name..."
if [ -f "templates/chatty_ai_html_template.html" ]; then
    mv templates/chatty_ai_html_template.html templates/Chatty_AI.html
    echo "✅ Renamed HTML template file"
else
    echo "⚠️  HTML template file not found"
fi

# 2. Copy logo files to correct location
echo "🖼️ Copying logo files to templates directory..."
if [ -f "resources/Chatty_AI_Logo.png" ]; then
    cp resources/Chatty_AI_Logo.png templates/Chatty_AI_logo.png
    echo "✅ Copied Chatty AI logo"
else
    echo "⚠️  Chatty AI logo not found in resources"
fi

if [ -f "resources/diamond_coding_Logo.png" ]; then
    cp resources/diamond_coding_Logo.png templates/diamond_coding_logo.png
    echo "✅ Copied Diamond Coding logo"
else
    echo "⚠️  Diamond Coding logo not found in resources"
fi

# 3. Set proper permissions
echo "🔐 Setting file permissions..."
chmod 644 templates/Chatty_AI.html
chmod 644 templates/*.png 2>/dev/null || echo "No PNG files to set permissions for"

# 4. Kill any existing processes
echo "🛑 Stopping any running processes..."
pkill -f "python3 app.py" 2>/dev/null || echo "No app.py processes found"
pkill -f python 2>/dev/null || echo "No python processes found"
sleep 2

# 5. Check socket.io installation
echo "📦 Checking Socket.IO installation..."
python3 -c "import flask_socketio; print('✅ Flask-SocketIO installed')" 2>/dev/null || echo "❌ Flask-SocketIO not installed"

echo "===================================================="
echo "🎉 Setup fixes completed!"
echo ""
echo "Next steps:"
echo "1. Run: python3 app.py"
echo "2. Open browser to: http://192.168.1.16:5000"
echo "===================================================="