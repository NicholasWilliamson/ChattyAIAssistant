#!/bin/bash

set -e

echo "🚀 Wyoming Voice Assistant Setup (with fixed versions & Python venv)"

# -------------------------
# Config
# -------------------------
VENV_DIR="$HOME/venv"
WHISPER_MODEL_DIR="$HOME/models/whisper/tiny"
PIPER_MODEL_DIR="$HOME/models/piper/en_US-amy-low"

# -------------------------
# 1. Install System Dependencies
# -------------------------
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv sox wget unzip

# -------------------------
# 2. Create & Activate Python Virtual Environment
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "🧪 Creating Python virtual environment in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

echo "✅ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# -------------------------
# 3. Install Wyoming packages (specific versions)
# -------------------------
echo "📥 Installing specific compatible Wyoming package versions..."
pip install --upgrade pip

# Uninstall any existing conflicting packages
pip uninstall -y wyoming wyoming-satellite wyoming-faster-whisper wyoming-piper || true

# Install fixed versions
pip install \
  wyoming==1.4.1 \
  wyoming-satellite==1.0.0 \
  wyoming-faster-whisper==1.0.1 \
  wyoming-piper==1.5.3

# -------------------------
# 4. Download Whisper (STT) Model
# -------------------------
echo "📁 Downloading FasterWhisper tiny model..."
mkdir -p "$WHISPER_MODEL_DIR"
cd "$WHISPER_MODEL_DIR"

wget -O model.bin "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin" || { echo "❌ Failed to download model.bin"; exit 1; }
wget -O config.json "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json" || { echo "❌ Failed to download config.json"; exit 1; }
wget -O tokenizer.json "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json" || { echo "❌ Failed to download tokenizer.json"; exit 1; }

# -------------------------
# 5. Download Piper (TTS) Voice
# -------------------------
echo "📁 Downloading Piper TTS model: en_US-amy-low..."
mkdir -p "$PIPER_MODEL_DIR"
cd "$PIPER_MODEL_DIR"

wget -O en_US-amy-low.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/amy-low/en_US-amy-low.onnx" || { echo "❌ Failed to download ONNX voice"; exit 1; }
wget -O en_US-amy-low.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/amy-low/en_US-amy-low.onnx.json" || { echo "❌ Failed to download voice config"; exit 1; }

# -------------------------
# 6. Start Services
# -------------------------
echo "🚀 Starting Wyoming STT and TTS services..."

# Commands to run the services
STT_CMD="$VENV_DIR/bin/wyoming-faster-whisper --model $WHISPER_MODEL_DIR --language en --uri tcp://0.0.0.0:10300"
TTS_CMD="$VENV_DIR/bin/wyoming-piper --voice $PIPER_MODEL_DIR/en_US-amy-low.onnx --uri tcp://0.0.0.0:10200"

# Start in available terminal emulator
if command -v gnome-terminal &> /dev/null; then
  gnome-terminal -- bash -c "$STT_CMD; exec bash" &
  sleep 1
  gnome-terminal -- bash -c "$TTS_CMD; exec bash" &
elif command -v lxterminal &> /dev/null; then
  lxterminal -e "$STT_CMD" &
  sleep 1
  lxterminal -e "$TTS_CMD" &
else
  echo "⚠️ No supported terminal found. Start these manually:"
  echo "$STT_CMD"
  echo "$TTS_CMD"
fi

echo "✅ Setup complete!"
echo "🧠 STT running on tcp://0.0.0.0:10300"
echo "🗣️ TTS running on tcp://0.0.0.0:10200"