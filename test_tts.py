import subprocess

VOICE = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
TEXT = "Hello Nick, this is a test of Piper text to speech on your Raspberry Pi 5"
OUTPUT = "output.wav"

# Run Piper
command = [
    "./piper/piper",
    "--model", VOICE,
    "--config", CONFIG,
    "--output_file", OUTPUT
]

subprocess.run(command, input=TEXT.encode("utf-8"))
