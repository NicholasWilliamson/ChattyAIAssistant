import subprocess

VOICE = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.json"
TEXT = "Hello Nick, this is a test of Piper text to speech on your Raspberry Pi 5"
OUTPUT = "output.wav"

# Run Piper
command = [
    "./piper",
    "--model", VOICE,
    "--config", CONFIG,
    "--output_file", OUTPUT,
    "--sentence", TEXT
]

subprocess.run(command)
