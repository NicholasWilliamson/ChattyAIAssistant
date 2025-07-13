#!/bin/bash

set -e

echo "🚀 Starting Wyoming AI Voice Assistant setup..."

# -------------------------
# Install Python & Pip
# -------------------------
echo "📦 Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip sox wget unzip

# -------------------------
# Install Wyoming Satellite and Dependencies
# -------------------------
echo "📥 Installing Wyoming Satellite + services..."
pip3 install --upgrade pip
pip3 install wyoming-satellite wyoming-faster-whisper wyoming-piper

# -------------------------
# Create folders
# -------------------------
echo "📁 Creating model folders..."
mkdir -p ~/models/whisper/tiny
mkdir -p ~/models/piper/en_US-amy-low

# -------------------------
# Download FasterWhisper (tiny) STT Model from Systran
# -------------------------
echo "⬇️ Downloading FasterWhisper tiny model (STT)..."
cd ~/models/whisper/tiny
wget -q https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin -O model.bin
wget -q https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json -O config.json
wget -q https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json -O tokenizer.json

# -------------------------
# Download Piper TTS Voice (Amy - US English)
# -------------------------
echo "⬇️ Downloading Piper voice model: en_US-amy-low..."
cd ~/models/piper/en_US-amy-low
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/amy-low/en_US-amy-low.onnx -O en_US-amy-low.onnx
wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/amy-low/en_US-amy-low.onnx.json -O en_US-amy-low.onnx.json

# -------------------------
# Launch STT and TTS Services
# -------------------------
echo "🚀 Starting Wyoming STT (FasterWhisper) and TTS (Piper)..."

# Start STT (port 10300)
gnome-terminal -- bash -c "wyoming-faster-whisper --model ~/models/whisper/tiny --language en --uri tcp://0.0.0.0:10300; exec bash"

# Start TTS (port 10200)
gnome-terminal -- bash -c "wyoming-piper --voice ~/models/piper/en_US-amy-low/en_US-amy-low.onnx --uri tcp://0.0.0.0:10200; exec bash"

echo "✅ Setup complete! STT running on port 10300, TTS on port 10200"