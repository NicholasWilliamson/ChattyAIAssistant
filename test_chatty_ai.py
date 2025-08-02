#!/usr/bin/env python3
"""
test_chatty_ai.py - Enhanced with Wake Word Detection
Record voice, transcribe using Whisper, reply with TinyLLaMA, and speak with Piper.
Includes wake word detection, silence detection, and command vs question processing.
"""

import os
import subprocess
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import random
import re
from faster_whisper import WhisperModel
from llama_cpp import Llama

# -------------------------------
# Config
# -------------------------------
WHISPER_MODEL_SIZE = "base"
LLAMA_MODEL_PATH = "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
BEEP_SOUND = "/home/nickspi5/Chatty_AI/audio_files/beep.wav"
LAUGHING_SOUND = "/home/nickspi5/Chatty_AI/audio_files/laughing.wav"
WAV_FILENAME = "user_input.wav"
RESPONSE_AUDIO = "output.wav"
WAKE_WORD_AUDIO = "wake_word_check.wav"

# Wake word phrases (case insensitive)
WAKE_WORDS = [
    "hey chatty",
    "hello chatty", 
    "sup chatty",
    "sub-chatty",
    "how's it chatty",
    "howzit chatty",
    "hi chatty",
    "yo chatty",
    "hey chuddy",
    "hello chuddy",
    "sup chuddy",
    "sub-chuddy",
    "how's it chuddy",
    "howzit chuddy",
    "hi chuddy",
    "yo chuddy",
    "hey cheddy",
    "hello cheddy",
    "sup cheddy",
    "sub-cheddy",
    "how's it cheddy",
    "howzit cheddy",
    "hi cheddy",
    "yo cheddy",
    "hey chetty",
    "Hello, Chetty",
    "sup chetty",
    "sub-chetty",
    "how's it chetty",
    "howzit chetty",
    "hi chetty",
    "yo chetty",
    "hey cherry",
    "hello cherry",
    "sup cherry",
    "sub-cherry",
    "how's it cherry",
    "howzit cherry",
    "hi cherry",
    "yo cherry"
]

# Wake word acknowledgment responses
WAKE_RESPONSES = [
    "Hi, I am listening for your request",
    "Hello! What can I help you with?",
    "Yes, I'm here. What do you need?",
    "Hi there! I'm ready to assist you",
    "Hello! How can I help you today?",
    "I'm listening. What's your question?",
    "Yes? What would you like to know?",
    "Hi! What can I do for you?"
]

# Command keywords and their functions
COMMANDS = {
    "flush the toilet": "toilet_flush",
    "turn on the lights": "lights_on", 
    "turn off the lights": "lights_off",
    "play music": "play_music",
    "stop music": "stop_music",
    "what time is it": "get_time",
    "shutdown system": "shutdown_system",
    "who is sponsoring this video": "who_is_sponsoring_this_video",
    "how is the weather today": "how_is_the_weather_today",
    "reboot system": "reboot_system"
}

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.1  # 100ms chunks
SILENCE_THRESHOLD = 0.035  # Raised threshold to account for background noise
MIN_SILENCE_DURATION = 1.5  # 1.5 seconds of silence to stop recording
MAX_RECORDING_DURATION = 20  # Maximum 20 seconds per recording

class ChattyAI:
    def __init__(self):
        self.whisper_model = None
        self.llama_model = None
        self.is_listening = False
        self.is_recording = False
        self.audio_buffer = []
        self.load_models()
    
    def load_models(self):
        """Load Whisper and LLaMA models"""
        print("🔄 Loading AI models...")
        try:
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("✅ Whisper model loaded")
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
            return False
        
        try:
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH, 
                n_ctx=2048, 
                temperature=0.7, 
                repeat_penalty=1.1, 
                n_gpu_layers=0, 
                verbose=False
            )
            print("✅ LLaMA model loaded")
        except Exception as e:
            print(f"❌ Failed to load LLaMA: {e}")
            return False
        
        return True
    
    def play_beep(self):
        """Play beep sound to acknowledge wake word"""
        try:
            subprocess.run(["aplay", BEEP_SOUND], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("❌ Could not play beep sound")
        except FileNotFoundError:
            print(f"❌ Beep file not found: {BEEP_SOUND}")
    
    def play_laughing(self):
        """Play laughing sound effect"""
        try:
            subprocess.run(["aplay", LAUGHING_SOUND], check=True, capture_output=True)
            print("😂 Played laughing sound")
        except subprocess.CalledProcessError:
            print("❌ Could not play laughing sound")
        except FileNotFoundError:
            print(f"❌ Laughing file not found: {LAUGHING_SOUND}")
    
    def speak_text(self, text):
        """Convert text to speech using Piper"""
        print(f"🔊 Speaking: {text}")
        try:
            command = [
                PIPER_EXECUTABLE,
                "--model", VOICE_PATH,
                "--config", CONFIG_PATH,
                "--output_file", RESPONSE_AUDIO
            ]
            subprocess.run(command, input=text.encode("utf-8"), check=True, capture_output=True)
            subprocess.run(["aplay", RESPONSE_AUDIO], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ TTS failed: {e}")
    
    def transcribe_audio(self, filename):
        """Transcribe audio using Whisper"""
        try:
            segments, _ = self.whisper_model.transcribe(filename)
            transcript = " ".join(segment.text for segment in segments).strip()
            return transcript
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""
    
    def detect_wake_word(self, text):
        """Check if text contains any wake word"""
        text_lower = text.lower().strip()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                print(f"🎯 Wake word detected: '{wake_word}' in '{text}'")
                return True
        return False
    
    def is_silence(self, audio_chunk):
        """Detect if audio chunk is silence"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms < SILENCE_THRESHOLD
    
    def record_with_silence_detection(self):
        """Record audio until silence is detected"""
        print("🎤 Recording... (speak now, I'll stop when you're quiet)")
        
        audio_data = []
        silence_duration = 0
        recording_duration = 0
        check_interval = 0.2  # Check every 200ms for better responsiveness
        samples_per_check = int(SAMPLE_RATE * check_interval)
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            audio_data.extend(indata[:, 0])  # Take first channel
        
        # Start recording
        with sd.InputStream(callback=audio_callback, 
                          samplerate=SAMPLE_RATE, 
                          channels=CHANNELS,
                          dtype='float32'):
            
            while recording_duration < MAX_RECORDING_DURATION:
                time.sleep(check_interval)
                recording_duration += check_interval
                
                # Check for silence in recent audio (last 200ms)
                if len(audio_data) >= samples_per_check:
                    recent_audio = np.array(audio_data[-samples_per_check:])
                    rms = np.sqrt(np.mean(recent_audio**2))
                    
                    # Debug output for silence detection
                    print(f"🔊 Audio level: {rms:.4f} (threshold: {SILENCE_THRESHOLD})")
                    
                    if rms < SILENCE_THRESHOLD:
                        silence_duration += check_interval
                        print(f"🔇 Silence: {silence_duration:.1f}s / {MIN_SILENCE_DURATION}s")
                        if silence_duration >= MIN_SILENCE_DURATION:
                            print("🔇 Silence detected, stopping recording")
                            break
                    else:
                        if silence_duration > 0:
                            print("🔊 Speech detected, resetting silence counter")
                        silence_duration = 0  # Reset silence counter
        
        # Save recorded audio
        if audio_data:
            audio_array = np.array(audio_data, dtype=np.float32)
            sf.write(WAV_FILENAME, audio_array, SAMPLE_RATE)
            print(f"✅ Recorded {len(audio_array)/SAMPLE_RATE:.1f} seconds of audio")
            return True
        
        return False
    
    def is_command(self, text):
        """Check if text is a command"""
        text_lower = text.lower().strip()
        for command in COMMANDS.keys():
            if command in text_lower:
                return command
        return None
    
    def execute_command(self, command):
        """Execute a system command"""
        command_func = COMMANDS.get(command)
        
        if command == "flush the toilet":
            response = "Oh Nick, you know I am a digital assistant. I cannot actually flush toilets! So why dont you haul your lazy arse up off the couch and flush the toilet yourself!"
        elif command == "turn on the lights":
            response = "I would turn on the lights if I were connected to a smart home system."
        elif command == "turn off the lights":
            response = "I would turn off the lights if I were connected to a smart home system."
        elif command == "play music":
            response = "I would start playing music if I had access to a music system."
        elif command == "stop music":
            response = "I would stop the music if any was playing."
        elif command == "who is sponsoring this video":
            # Play laughing sound first, then speak the rest
            self.play_laughing()
            response = "You are very funny Nick. You know you dont have any sponsors for your videos!"
        elif command == "how is the weather today":
            response = "O M G Nick! Surely you DO NOT want to waste my valuable resources by asking me what the weather is today. Cant you just look out the window or ask Siri. That is about all Siri is good for!"
        elif command == "what time is it":
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}"
        elif command == "shutdown system":
            response = "I would shutdown the system, but I will skip that for safety during testing."
        elif command == "reboot system":
            response = "I would reboot the system, but I will skip that for safety during testing."
        else:
            response = f"I understand you want me to {command}, but I don't have that capability yet."
        
        return response
    
    def query_llama(self, prompt):
        """Generate LLM response for questions"""
        print("🤖 Generating LLM response...")
        
        formatted_prompt = f"You are a friendly, helpful assistant. Give a brief, conversational answer.\nUser: {prompt}\nAssistant: "
        
        try:
            result = self.llama_model(formatted_prompt, max_tokens=100)
            if "choices" in result and result["choices"]:
                reply_text = result["choices"][0]["text"].strip()
                # Clean up the response
                reply_text = re.sub(r"\(.*?\)", "", reply_text)  # Remove roleplay
                reply_text = re.sub(r"(User:|Assistant:)", "", reply_text)  # Remove labels
                reply_text = reply_text.strip()
                
                # Ensure response isn't too long
                sentences = reply_text.split('.')
                if len(sentences) > 3:
                    reply_text = '. '.join(sentences[:3]) + '.'
                
                return reply_text
            else:
                return "I'm not sure how to answer that."
        except Exception as e:
            print(f"❌ LLM inference failed: {e}")
            return "Sorry, I had trouble processing that question."
    
    def process_user_input(self, text):
        """Process user input - determine if command or question"""
        print(f"📝 Processing: {text}")
        
        # Check if it's a command
        command = self.is_command(text)
        if command:
            print(f"⚙️ Executing command: {command}")
            response = self.execute_command(command)
        else:
            print("❓ Processing as question for LLM")
            response = self.query_llama(text)
        
        return response
    
    def listen_for_wake_word(self):
        """Continuously listen for wake words"""
        print("👂 Listening for wake words...")
        print(f"Wake words: {', '.join(WAKE_WORDS)}")
        
        while self.is_listening:
            try:
                # Record a short clip to check for wake word
                print("🔍 Checking for wake word...")
                
                # Record using sounddevice and save to file
                audio_data = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()  # Wait for recording to complete
                
                # Save the audio to file
                sf.write(WAKE_WORD_AUDIO, audio_data, SAMPLE_RATE)
                print(f"💾 Saved wake word check audio: {WAKE_WORD_AUDIO}")
                
                # Transcribe and check for wake word
                transcript = self.transcribe_audio(WAKE_WORD_AUDIO)
                print(f"🎧 DEBUG - I heard: '{transcript}'")  # Debug output
                
                if transcript and self.detect_wake_word(transcript):
                    # Wake word detected!
                    print("🎉 WAKE WORD ACTIVATED!")
                    self.play_beep()
                    
                    # Speak acknowledgment
                    response = random.choice(WAKE_RESPONSES)
                    self.speak_text(response)
                    
                    # Record user's full request
                    if self.record_with_silence_detection():
                        user_text = self.transcribe_audio(WAV_FILENAME)
                        if user_text:
                            print(f"👤 User said: {user_text}")
                            
                            # Process the input
                            ai_response = self.process_user_input(user_text)
                            self.speak_text(ai_response)
                        else:
                            self.speak_text("I didn't catch that. Could you try again?")
                    
                    print("👂 Back to listening for wake words...")
                else:
                    if transcript:
                        print(f"❌ No wake word in: '{transcript}'")
                    else:
                        print("❌ No speech detected")
                
                time.sleep(0.5)  # Brief pause between checks
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping wake word detection...")
                break
            except Exception as e:
                print(f"❌ Error in wake word detection: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def test_transcription(self):
        """Test function to check transcription accuracy"""
        print("🧪 TRANSCRIPTION TEST MODE")
        print("Speak phrases to test wake word detection accuracy")
        print("Press Ctrl+C to exit test mode")
        
        while True:
            try:
                input("Press Enter to record a test phrase...")
                
                # Record 3 seconds
                print("🎤 Recording test phrase...")
                audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                sf.write("test_audio.wav", audio, SAMPLE_RATE)
                
                # Transcribe
                transcript = self.transcribe_audio("test_audio.wav")
                print(f"📝 I heard: '{transcript}'")
                
                # Check wake word detection
                if self.detect_wake_word(transcript):
                    print("✅ WAKE WORD DETECTED!")
                else:
                    print("❌ No wake word detected")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n🛑 Exiting test mode")
                break
    
    def start_listening(self):
        """Start the wake word detection system"""
        if not self.whisper_model or not self.llama_model:
            print("❌ Models not loaded properly")
            return
        
        self.is_listening = True
        
        print("🚀 Chatty AI Wake Word Detection Started!")
        print("=" * 50)
        print("Say one of these wake words to activate:")
        for wake_word in WAKE_WORDS:
            print(f"  • {wake_word}")
        print("=" * 50)
        
        try:
            self.listen_for_wake_word()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Chatty AI...")
        finally:
            self.is_listening = False
    
    def run_single_interaction(self):
        """Run the original single-interaction mode"""
        print("🎤 Single interaction mode - Recording 5 seconds...")
        
        # Record audio
        audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        sf.write(WAV_FILENAME, audio, SAMPLE_RATE)
        
        # Transcribe
        user_text = self.transcribe_audio(WAV_FILENAME)
        if not user_text:
            print("❌ No voice input detected.")
            return
        
        print(f"👤 You said: {user_text}")
        
        # Process and respond
        response = self.process_user_input(user_text)
        self.speak_text(response)

def main():
    """Main function with mode selection"""
    chatty = ChattyAI()
    
    print("🤖 Chatty AI - Enhanced with Wake Word Detection")
    print("=" * 60)
    print("Choose mode:")
    print("1. Wake Word Detection (continuous listening)")
    print("2. Single Interaction (original 5-second recording)")
    print("3. Test Transcription (for fine-tuning wake words)")
    print("=" * 60)
    
    try:
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == "1":
            chatty.start_listening()
        elif choice == "2":
            chatty.run_single_interaction()
        elif choice == "3":
            chatty.test_transcription()
        else:
            print("❌ Invalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()