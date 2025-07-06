#!/usr/bin/env python3
"""
run_testllama_voice.py
Record voice, transcribe to text using Vosk, generate reply using TinyLLaMA, and speak it with espeak-ng.
"""

import os
import wave
import json
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
from vosk import Model as VoskModel, KaldiRecognizer
from llama_cpp import Llama

# -------------------------------
# Config
# -------------------------------
VOSK_MODEL_PATH = "vosk-models/vosk-model-small-en-us-0.15"
WAV_FILENAME = "user_input.wav"
LLAMA_MODEL_PATH = "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"

# -------------------------------
# Record 5 seconds audio
# -------------------------------
def record_audio(filename=WAV_FILENAME, duration=5, samplerate=16000, channels=1):
    print("🎤 Recording 5s of audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved audio to: {filename}")

# -------------------------------
# Transcribe audio to text
# -------------------------------
def transcribe_audio(filename):
    print("🧠 Transcribing...")
    wf = wave.open(filename, "rb")
    model = VoskModel(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get("text", ""))
    results.append(json.loads(rec.FinalResult()).get("text", ""))
    transcript = " ".join(filter(None, results)).strip()
    print(f"📝 Transcript: {transcript}")
    return transcript

# -------------------------------
# Generate reply using TinyLLaMA
# -------------------------------
def query_llama(prompt):
    print("🤖 Generating response...")
    llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048, temperature=0.7, repeat_penalty=1.1)
    prompt_text = f"[INST] <<SYS>>You are a helpful assistant.<</SYS>> {prompt} [/INST]"
    response = llm(prompt_text, max_tokens=100)
    reply = response["choices"][0]["text"].strip()
    print(f"💬 TinyLLaMA says: {reply}")
    return reply

# -------------------------------
# Speak using espeak-ng
# -------------------------------
def speak(text):
    print("🔊 Speaking...")
    subprocess.run(["espeak-ng", "-v", "en-us+f3", "-s", "150", "-p", "70", text])

# -------------------------------
# Main Orchestration
# -------------------------------
def main():
    record_audio()
    user_text = transcribe_audio(WAV_FILENAME)
    if not user_text:
        print("❌ No voice input detected.")
        return
    reply = query_llama(user_text)
    speak(reply)

if __name__ == "__main__":
    main()