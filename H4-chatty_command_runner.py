#!/usr/bin/env python3
"""
H4-chatty_command_runner.py

Records 5 seconds of audio, transcribes it to text using Vosk,
matches a command using fuzzy matching, speaks the response using eSpeak,
and sends notifications to Discord and Telegram.
"""

import os
import wave
import json
import subprocess
from vosk import Model, KaldiRecognizer
from difflib import get_close_matches
from server.notify import send_discord_message, send_telegram_message

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
VOSK_MODEL_PATH = "model"
WAV_FILENAME = "test.wav"
RECORD_DURATION = 5  # seconds
SAMPLE_RATE = 16000
CHANNELS = 1

# ------------------------------------------------------------------------------
# Utility: Record audio using sounddevice
# ------------------------------------------------------------------------------
import sounddevice as sd
def record_audio(filename, duration=RECORD_DURATION, samplerate=SAMPLE_RATE, channels=CHANNELS):
    print(f"🎤 Recording {duration}s of audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    # Save to WAV
    import soundfile as sf
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved to '{filename}'")

# ------------------------------------------------------------------------------
# Utility: Speak a phrase using eSpeak-NG
# ------------------------------------------------------------------------------
def speak(text, voice="en-us", speed=150, pitch=70, filename="response.wav"):
    cmd = ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "--stdout"]
    with open(filename, "wb") as f:
        subprocess.run(cmd, input=text.encode("utf-8"), stdout=f, check=True)
    subprocess.run(["aplay", filename], check=True)

# ------------------------------------------------------------------------------
# Define Commands and Actions
# ------------------------------------------------------------------------------
def cmd_flush_toilet():
    message = "🚽 Chatty has just flushed the toilet!"
    speak("Hello Nick, I am executing your command and flushing the toilet.")
    print(message)
    send_discord_message(message)
    send_telegram_message(message)

def cmd_turn_on_light():
    message = "💡 Chatty has turned on the light!"
    speak("Hello Nick, I am executing your command and turning the light on now.")
    print(message)
    send_discord_message(message)
    send_telegram_message(message)

def cmd_turn_off_light():
    message = "💡 Chatty has turned off the light!"
    speak("Hello Nick, I am executing your command and turning the light off now.")
    print(message)
    send_discord_message(message)
    send_telegram_message(message)

def cmd_start_recording():
    message = "🎙️ Chatty has started recording."
    speak("Hello Nick, I am executing your command and starting recording.")
    print(message) now
    send_discord_message(message)
    send_telegram_message(message)

def cmd_stop_recording():
    message = "🛑 Chatty has stopped recording."
    speak("Hello Nick, I am executing your command and stopping recording now.")
    print(message)
    send_discord_message(message)
    send_telegram_message(message)

COMMANDS = {
    "flush the toilet":     cmd_flush_toilet,
    "turn on the light":    cmd_turn_on_light,
    "turn off the light":   cmd_turn_off_light,
    "start recording":      cmd_start_recording,
    "stop recording":       cmd_stop_recording,
}

# ------------------------------------------------------------------------------
# Match Command with Fuzzy Matching
# ------------------------------------------------------------------------------
def match_command(transcript: str, commands: dict, cutoff: float = 0.6):
    matches = get_close_matches(transcript.lower(), commands.keys(), n=1, cutoff=cutoff)
    return matches[0] if matches else None

# ------------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------------
def main():
    # Step 1: Record audio
    record_audio(WAV_FILENAME)

    # Step 2: Load Vosk model
    print(f"📦 Loading Vosk model from '{VOSK_MODEL_PATH}'...")
    model = Model(VOSK_MODEL_PATH)

    # Step 3: Open the WAV file
    wf = wave.open(WAV_FILENAME, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio must be mono 16-bit PCM WAV format.")

    recognizer = KaldiRecognizer(model, wf.getframerate())

    # Step 4: Transcribe
    print("🧠 Transcribing audio...")
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            res = json.loads(recognizer.Result())
            results.append(res.get("text", ""))
    final_res = json.loads(recognizer.FinalResult())
    results.append(final_res.get("text", ""))

    transcript = " ".join(filter(None, results)).strip()
    print("\n📝 Transcript:", transcript or "[empty]")

    # Step 5: Match and run command
    if transcript:
        matched_cmd = match_command(transcript, COMMANDS, cutoff=0.6)
        if matched_cmd:
            print(f"✅ Matched command: '{matched_cmd}'")
            COMMANDS[matched_cmd]()  # Call the function
        else:
            speak("Sorry, I didn't understand that command.")
            print("❌ No command matched.")
    else:
        speak("I didn't catch anything. Please try again.")
        print("❌ Empty transcript.")

if __name__ == "__main__":
    main()