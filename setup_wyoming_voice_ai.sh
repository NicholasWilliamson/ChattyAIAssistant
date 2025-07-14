#!/bin/bash

set -e

echo "🚀 Wyoming Voice Assistant Setup"

# -------------------------
# Configuration
# -------------------------
VENV_DIR="$HOME/wyoming-venv"
WHISPER_MODEL_DIR="$HOME/models/whisper/tiny"
PIPER_MODEL_DIR="$HOME/models/piper/en_US-amy-low"

# -------------------------
# 1. System Dependencies
# -------------------------
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv sox wget unzip

# -------------------------
# 2. Python Virtual Environment
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "🧪 Creating Python virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "✅ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# -------------------------
# 3. Install Python Packages (compatible versions)
# -------------------------
echo "📥 Installing Wyoming packages..."
pip install --upgrade pip

# Clean up existing packages
pip uninstall -y wyoming wyoming-satellite wyoming-faster-whisper wyoming-piper || true

# Install fixed, compatible versions
pip install \
  wyoming==1.1.0 \
  wyoming-satellite==0.9.0 \
  wyoming-faster-whisper==1.0.1 \
  wyoming-piper==1.5.3

# -------------------------
# 4. Download FasterWhisper Tiny Model
# -------------------------
echo "📁 Downloading FasterWhisper tiny STT model..."
mkdir -p "$WHISPER_MODEL_DIR"
cd "$WHISPER_MODEL_DIR"

wget -nc -O model.bin "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin"
wget -nc -O config.json "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json"
wget -nc -O tokenizer.json "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json"

# -------------------------
# 5. Download Piper Voice (Amy - US English)
# -------------------------
echo "🗣️ Downloading Piper Amy voice..."
mkdir -p "$PIPER_MODEL_DIR"
cd "$PIPER_MODEL_DIR"

wget -nc "https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-amy-low.onnx"
wget -nc "https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/en_US-amy-low.onnx.json"

# -------------------------
# 6. Start Wyoming Services
# -------------------------
echo "🚀 Starting Wyoming STT and TTS services..."

STT_CMD="$VENV_DIR/bin/wyoming-faster-whisper --model $WHISPER_MODEL_DIR --language en --uri tcp://0.0.0.0:10300"
TTS_CMD="$VENV_DIR/bin/wyoming-piper --voice $PIPER_MODEL_DIR/en_US-amy-low.onnx --uri tcp://0.0.0.0:10200"

# Try to launch in new terminal windows
if command -v gnome-terminal &> /dev/null; then
  gnome-terminal -- bash -c "$STT_CMD; exec bash" &
  sleep 1
  gnome-terminal -- bash -c "$TTS_CMD; exec bash" &
elif command -v lxterminal &> /dev/null; then
  lxterminal -e "$STT_CMD" &
  sleep 1
  lxterminal -e "$TTS_CMD" &
else
  echo "⚠️ No supported terminal found. Please run these commands manually:"
  echo "$STT_CMD"
  echo "$TTS_CMD"
fi

echo "✅ Setup complete!"
echo "🧠 STT running on tcp://0.0.0.0:10300"
echo "🗣️ TTS running on tcp://0.0.0.0:10200"