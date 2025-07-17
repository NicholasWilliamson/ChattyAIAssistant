from piper_tts import PiperVoice

voice = PiperVoice.load(
    model_path="~/Chatty_AI/voices/en_US-amy-low.onnx",
    config_path="~/Chatty_AI/voices/en_US-amy-low.onnx.json"
)

audio = voice.speak("Hello Nick, your Chatty AI Assistant is now speaking!", output_path="test.wav")