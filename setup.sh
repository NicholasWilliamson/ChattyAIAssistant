#!/bin/bash

set -e  # Exit on error

# Step 1: Create project directory
echo "[+] Creating ~/Chatty_AI directory..."
mkdir -p ~/Chatty_AI
cd ~/Chatty_AI

# Step 2: Create virtual environment
echo "[+] Creating Python virtual environment (chatty-venv)..."
python3 -m venv chatty-venv
source chatty-venv/bin/activate

# Step 3: Upgrade pip
echo "[+] Upgrading pip..."
pip install --upgrade pip

# Step 4: Create requirements.txt
cat <<EOF > requirements.txt
wyoming==1.5.3
wyoming-piper==1.5.3
wyoming-whisper==1.0.0
python-dotenv
EOF

# Step 5: Install Python dependencies
echo "[+] Installing Python dependencies..."
pip install -r requirements.txt

# Step 6: Install system dependencies
echo "[+] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y alsa-utils python3-pyaudio portaudio19-dev

# Step 7: Download Piper voice: en_US-amy-low
echo "[+] Downloading Piper voice: en_US-amy-low..."
mkdir -p voices/en_US-amy-low
curl -L -o voices/en_US-amy-low/en_US-amy-low.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx
curl -L -o voices/en_US-amy-low/en_US-amy-low.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.json

# Step 8: Create .env file
echo "[+] Creating .env file..."
cat <<EOF > .env
WHISPER_HOST=localhost
WHISPER_PORT=10300
PIPER_HOST=localhost
PIPER_PORT=10200
AUDIO_DEVICE_INDEX=0
SAMPLE_RATE=16000
CHUNK_SIZE=1024
EOF

# Step 9: Create config.json
echo "[+] Creating config.json..."
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

# Final Instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "To start working:"
echo "1. cd ~/Chatty_AI"
echo "2. source chatty-venv/bin/activate"
echo "3. Start Whisper STT:"
echo "     python -m wyoming_whisper --model base.en"
echo "4. In another terminal, start Piper TTS:"
echo "     python -m wyoming_piper --voice voices/en_US-amy-low/en_US-amy-low.onnx"
echo ""