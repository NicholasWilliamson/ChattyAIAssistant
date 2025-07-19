#!/usr/bin/env python3
"""
run_chatty_ai.py
Record voice, transcribe using Whisper, reply with TinyLLaMA, and speak with Piper.
"""

import os
import subprocess
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from llama_cpp import Llama

# -------------------------------
# Config
# -------------------------------
WHISPER_MODEL_SIZE = "base"
LLAMA_MODEL_PATH = "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
WAV_FILENAME = "user_input.wav"
RESPONSE_AUDIO = "output.wav"

# -------------------------------
# Record 5 seconds of audio
# -------------------------------
def record_audio(filename=WAV_FILENAME, duration=5, samplerate=16000, channels=1):
    print("🎤 Recording 5s of audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved audio to: {filename}")

# -------------------------------
# Transcribe using Whisper
# -------------------------------
def transcribe_audio(filename):
    print("🧠 Transcribing with Whisper...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(filename)
    transcript = " ".join(segment.text for segment in segments).strip()
    print(f"📝 Transcript: {transcript}")
    return transcript

# -------------------------------
# Generate LLM response
# -------------------------------
def query_llama(prompt):
    print("🤖 Generating response...")
    try:
        llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, temperature=0.7, repeat_penalty=1.1, n_gpu_layers=0)
    except Exception as e:
        print(f"❌ Failed to load TinyLLaMA: {e}")
        return "Sorry, I couldn't load the AI model."

    formatted_prompt = (
        "[INST] <<SYS>>You are a helpful assistant.<</SYS>> "
        f"{prompt} [/INST]"
    )

    try:
        result = llm(formatted_prompt, max_tokens=100)
        if "choices" in result and result["choices"]:
            reply_text = result["choices"][0]["text"].strip()
            print(f"💬 TinyLLaMA says: {reply_text}")
            speak_text(reply_text)
            return reply_text
        else:
            print("⚠️ No response from model.")
            return "I did not understand that."
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return "An error occurred while generating a reply."

# -------------------------------
# Speak with Piper
# -------------------------------
def speak_text(text):
    print("🔊 Speaking with Piper...")
    try:
        command = [
            PIPER_EXECUTABLE,
            "--model", VOICE_PATH,
            "--config", CONFIG_PATH,
            "--output_file", RESPONSE_AUDIO
        ]
        subprocess.run(command, input=text.encode("utf-8"), check=True)
        subprocess.run(["aplay", RESPONSE_AUDIO])
    except subprocess.CalledProcessError as e:
        print("❌ Piper playback failed:", e)

# -------------------------------
# Main Orchestration
# -------------------------------
def main():
    record_audio()
    user_text = transcribe_audio(WAV_FILENAME)
    if not user_text:
        print("❌ No voice input detected.")
        return
    query_llama(user_text)

if __name__ == "__main__":
    main()