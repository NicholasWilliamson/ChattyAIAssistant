import subprocess

text = "Hello Nick, this is a test of Piper text to speech on your Raspberry Pi 5."
model_path = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
config_path = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.json"
output_path = "output.wav"

# Run Piper
subprocess.run([
    "piper",
    "-m", model_path,
    "-c", config_path,
    "-f", output_path,
    "-i", "-",  # Read input from stdin
], input=text.encode(), check=True)

# Play result (optional)
subprocess.run(["aplay", output_path])