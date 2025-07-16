#!/bin/bash

# Setup script for Chatty_AI local voice assistant
# By Nick Williamson | Raspberry Pi 5 | Python 3.11 venv

set -e  # Exit immediately if a command fails

# Create Chatty_AI project folder
mkdir -p ~/Chatty_AI
cd ~/Chatty_AI

# Create Python virtual environment
python3 -m venv chatty-venv
source chatty-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core packages without version conflicts
pip install python-dotenv

# Install Wyoming 1.5.3 manually
pip install wyoming==1.5.3

# Install other Wyoming services without auto-dependency conflicts
pip install --no-deps wyoming-piper==1.5.3
pip install --no-deps wyoming-faster-whisper==2.0.0

# Install system dependencies for audio (if not already installed)
sudo apt-get update
sudo apt-get install -y alsa-utils python3-pyaudio

# Create .env file
cat <<EOF > .env
# Wyoming Services
WHISPER_HOST=localhost
WHISPER_PORT=10300
PIPER_HOST=localhost
PIPER_PORT=10200

# Audio Settings
AUDIO_DEVICE_INDEX=0
SAMPLE_RATE=16000
CHUNK_SIZE=1024
EOF

# Create basic config.json
cat <<EOF > config.json
{
  "speech_recognition": {
    "engine": "whisper",
    "language": "en-US",
    "timeout": 5,
    "phrase_timeout": 0.3
  },
  "text_to_speech": {
    "engine": "piper",
    "voice": "en_US-amy-low",
    "speed": 1.0
  }
}
EOF

echo "✅ Chatty_AI setup complete!"
echo "➡️  To activate the environment later: source ~/Chatty_AI/chatty-venv/bin/activate"