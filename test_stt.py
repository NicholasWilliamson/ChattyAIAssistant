from faster_whisper import WhisperModel

# Load the Whisper model
model_size = "tiny"  # or "base", "small", "medium", "large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Path to test audio file (WAV/MP3)
audio_path = "test_audio.wav"

# Transcribe audio
segments, _ = model.transcribe(audio_path)

print("Transcription:")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")