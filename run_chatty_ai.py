import subprocess
import time
import re
from faster_whisper import WhisperModel
from llama_cpp import Llama
import os

# === CONFIGURATION ===

# Piper voice
PIPER_VOICE = "en_US-lessac-medium.onnx"

# Model path (adjust if needed)
LLM_MODEL_PATH = "/home/nick/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"

# === INITIALIZATION ===

# Initialize Whisper model (use 'base', 'small', etc.)
whisper_model = WhisperModel("base", compute_type="int8")

# Initialize TinyLLaMA via llama-cpp
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_batch=64,
    verbose=False
)

# === FUNCTIONS ===

def record_audio(filename="input.wav", duration=5):
    print("🎙️ Recording...")
    subprocess.run([
        "arecord", "-D", "plughw:1", "-f", "cd", "-t", "wav",
        "-d", str(duration), "-r", "16000", "-c", "1", filename
    ])
    print("✅ Audio recorded.")

def transcribe_audio(filename="input.wav"):
    print("🧠 Transcribing with Whisper...")
    segments, _ = whisper_model.transcribe(filename)
    transcription = "".join([segment.text for segment in segments]).strip()
    print(f"📝 Transcribed: {transcription}")
    return transcription

def query_llm(prompt, max_tokens=128):
    formatted_prompt = f"[INST] {prompt.strip()} [/INST]"
    print(f"🤖 Sending to LLM: {formatted_prompt}")

    response = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        stop=["User:", "###"]
    )

    raw_text = response["choices"][0]["text"]
    cleaned = re.sub(r"(User:|Assistant:)", "", raw_text).strip()
    print(f"🤖 Response: {cleaned}")
    return cleaned

def speak_text(text):
    print(f"🔊 Speaking: {text}")
    try:
        subprocess.run(["piper", "--model", PIPER_VOICE, "--output-raw", "--sentence-silence", "0.3"],
                       input=text.encode("utf-8"), stdout=subprocess.PIPE, check=True)
        subprocess.run(["aplay", "-q"], input=subprocess.PIPE)
    except Exception as e:
        print(f"❌ Piper error: {e}")

def save_log(transcript, response, log_path="chatty_log.txt"):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} You: {transcript}\n")
        f.write(f"{timestamp} Chatty: {response}\n\n")

# === MAIN LOOP ===

def main():
    record_audio()
    user_input = transcribe_audio()

    if not user_input:
        print("❗ No speech detected.")
        return

    reply = query_llm(user_input)

    if not reply or len(reply.split()) < 2:
        print("🤷 Response too short or empty. Skipping.")
        return

    speak_text(reply)

    # Optional logging
    save_log(user_input, reply)

if __name__ == "__main__":
    while True:
        print("\n🎤 Say something to Chatty AI!")
        main()
        print("\n⏱️ Waiting 2 seconds before next input...\n")
        time.sleep(2)