import subprocess
import requests
import time
from faster_whisper import WhisperModel

# Configuration
PIPER_PATH = "./piper/piper"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
OUTPUT_WAV = "output.wav"
RECORD_WAV = "record.wav"
LOCALAI_URL = "http://localhost:8080/v1/chat/completions"

# Step 1: Record audio (5 seconds)
print("🎤 Recording 5 seconds of audio...")
subprocess.run([
    "arecord", "-f", "cd", "-t", "wav", "-d", "5", "-r", "16000", "-c", "1", RECORD_WAV
])
print("✅ Recording complete.")

# Step 2: Transcribe with Faster-Whisper
print("🧠 Transcribing with Faster-Whisper...")
model = WhisperModel("base.en", compute_type="int8")  # Use "tiny", "base", etc.
segments, _ = model.transcribe(RECORD_WAV)
transcription = "".join([seg.text for seg in segments])
print(f"📝 You said: {transcription}")

# Step 3: Send to LocalAI
print("🤖 Sending to TinyLLaMA via LocalAI...")
payload = {
    "model": "tinyllama",  # or the name you've set in your LocalAI config
    "messages": [
        {"role": "system", "content": "You are Chatty AI, a friendly AI assistant."},
        {"role": "user", "content": transcription}
    ],
    "temperature": 0.7
}

response = requests.post(LOCALAI_URL, json=payload)
llm_reply = response.json()["choices"][0]["message"]["content"]
print(f"💬 Chatty AI says: {llm_reply}")

# Step 4: Convert response to speech using Piper
print("🗣️ Converting reply to speech...")
subprocess.run([
    PIPER_PATH,
    "--model", VOICE_PATH,
    "--config", CONFIG_PATH,
    "--output_file", OUTPUT_WAV
], input=llm_reply.encode("utf-8"))

# Step 5: Play audio
print("🔊 Playing Chatty's reply...")
subprocess.run(["aplay", OUTPUT_WAV])