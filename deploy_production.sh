#!/bin/bash
# Deploy Chatty AI to Production with Gunicorn WSGI Server

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/home/nickspi5/Chatty_AI"
VENV_DIR="$PROJECT_DIR/chatty-venv"
SERVICE_NAME="chatty-ai"
USER="nickspi5"

echo "🚀 Deploying Chatty AI to Production"
echo "===================================="

# Check if running as correct user
if [ "$USER" != "nickspi5" ]; then
    echo "❌ This script should be run as user 'nickspi5'"
    exit 1
fi

# Navigate to project directory
cd "$PROJECT_DIR" || {
    echo "❌ Cannot access project directory: $PROJECT_DIR"
    exit 1
}

echo "📂 Current directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at: $VENV_DIR"
    echo "Please create the virtual environment first"
    exit 1
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify Python environment
echo "🔍 Python version: $(python --version)"
echo "🔍 Pip version: $(pip --version)"

# Install/upgrade production dependencies
echo "📦 Installing production dependencies..."
pip install --upgrade pip

# Install Gunicorn and eventlet (required for Socket.IO)
pip install gunicorn eventlet

# Verify required Python packages
echo "🔍 Checking required packages..."
python -c "
import sys
required_packages = [
    'flask', 'flask_socketio', 'psutil', 'cv2', 'face_recognition',
    'faster_whisper', 'llama_cpp', 'picamera2', 'sounddevice', 
    'soundfile', 'numpy', 'requests', 'gunicorn', 'eventlet'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'cv2':
            import cv2
        else:
            __import__(package)
        print(f'✅ {package}')
    except ImportError:
        print(f'❌ {package}')
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages are installed!')
"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p security_photos
mkdir -p security_logs
mkdir -p templates

# Set proper permissions
echo "🔒 Setting file permissions..."
chmod +x chatty_ai_web.py
chmod +x gunicorn_config.py
chmod +r templates/*

# Test Gunicorn configuration
echo "🧪 Testing Gunicorn configuration..."
gunicorn --check-config --config gunicorn_config.py chatty_ai_web:app

# Create/update systemd service file
echo "⚙️  Setting up systemd service..."
SERVICE_FILE="/tmp/chatty-ai.service"

# Copy the service file to tmp (user will need to move it manually)
cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Chatty AI Web Interface
After=network.target
Wants=network-online.target

[Service]
Type=notify
User=nickspi5
Group=nickspi5
WorkingDirectory=/home/nickspi5/Chatty_AI
Environment=PATH=/home/nickspi5/Chatty_AI/chatty-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONPATH=/home/nickspi5/Chatty_AI
Environment=CHATTY_AI_ENV=production

ExecStart=/home/nickspi5/Chatty_AI/chatty-venv/bin/gunicorn --config gunicorn_config.py chatty_ai_web:app
ExecReload=/bin/kill -s HUP $MAINPID

Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/home/nickspi5/Chatty_AI
ReadWritePaths=/tmp
ReadWritePaths=/var/log

MemoryMax=2G
CPUQuota=80%

StandardOutput=journal
StandardError=journal
SyslogIdentifier=chatty-ai

KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "📄 Service file created at: $SERVICE_FILE"

# Create startup script for manual running
echo "📝 Creating startup script..."
cat > start_production.sh << EOF
#!/bin/bash
# Start Chatty AI in production mode

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

echo "🚀 Starting Chatty AI Production Server..."
echo "Access at: http://\$(hostname -I | awk '{print \$1}'):5000"
echo "Press Ctrl+C to stop"

gunicorn --config gunicorn_config.py chatty_ai_web:app
EOF

chmod +x start_production.sh

# Create stop script
echo "🛑 Creating stop script..."
cat > stop_production.sh << EOF
#!/bin/bash
# Stop Chatty AI production server

echo "🛑 Stopping Chatty AI Production Server..."
pkill -f "gunicorn.*chatty_ai_web"
echo "Server stopped"
EOF

chmod +x stop_production.sh

# Final instructions
echo ""
echo "✅ Production deployment prepared successfully!"
echo "============================================="
echo ""
echo "📋 Next steps:"
echo "1. Copy service file to systemd directory:"
echo "   sudo cp $SERVICE_FILE /etc/systemd/system/"
echo ""
echo "2. Reload systemd and enable service:"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable $SERVICE_NAME"
echo ""
echo "3. Start the service:"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "4. Check service status:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "📄 Manual startup options:"
echo "• Run production server manually: ./start_production.sh"
echo "• Stop production server: ./stop_production.sh"
echo ""
echo "📊 Monitoring commands:"
echo "• View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "• Restart service: sudo systemctl restart $SERVICE_NAME"
echo "• Stop service: sudo systemctl stop $SERVICE_NAME"
echo ""
echo "🌐 Server will be available at:"
echo "• Local: http://localhost:5000"
echo "• Network: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "🔧 Configuration files:"
echo "• Main app: chatty_ai_web.py"
echo "• Gunicorn config: gunicorn_config.py"
echo "• Service file: $SERVICE_FILE"
echo ""
echo "Deployment complete! 🎉"