#!/usr/bin/env python3
"""
chatty_ai_preloader.py - Preload AI models and keep them in memory
This service runs at boot to preload all AI models for faster response times
"""

import os
import sys
import time
import logging
import signal
import mmap
import pickle
import threading
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama

# Configuration
WHISPER_MODEL_SIZE = "base"
LLAMA_MODEL_PATH = "/home/nickspi5/Chatty_AI/tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VOICE_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
CONFIG_PATH = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx.json"
PIPER_EXECUTABLE = "/home/nickspi5/Chatty_AI/piper/piper"
ENCODINGS_FILE = "/home/nickspi5/Chatty_AI/encodings.pickle"

# Shared memory paths for model state
SHARED_MEM_DIR = "/dev/shm/chatty_ai"
PRELOAD_STATUS_FILE = f"{SHARED_MEM_DIR}/preload_status"
MODEL_CACHE_DIR = f"{SHARED_MEM_DIR}/model_cache"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/nickspi5/Chatty_AI/logs/preloader.log')
    ]
)
logger = logging.getLogger(__name__)

class ChattyAIPreloader:
    def __init__(self):
        self.running = True
        self.whisper_model = None
        self.llama_model = None
        self.face_encodings = None
        self.voice_model_loaded = False
        
        # Create shared memory directories
        os.makedirs(SHARED_MEM_DIR, exist_ok=True)
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs('/home/nickspi5/Chatty_AI/logs', exist_ok=True)
        
        # Signal handling
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def update_status(self, status_dict):
        """Update preload status in shared memory"""
        try:
            with open(PRELOAD_STATUS_FILE, 'w') as f:
                f.write(str(status_dict))
            logger.info(f"Status updated: {status_dict}")
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            
    def preload_whisper(self):
        """Preload Whisper model into memory"""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
            start_time = time.time()
            
            self.whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, 
                device="cpu", 
                compute_type="int8",
                download_root="/home/nickspi5/Chatty_AI/.cache/whisper"
            )
            
            # Warm up the model with a dummy transcription
            logger.info("Warming up Whisper model...")
            dummy_audio = "/home/nickspi5/Chatty_AI/audio_files/beep.wav"
            if os.path.exists(dummy_audio):
                segments, _ = self.whisper_model.transcribe(dummy_audio)
                _ = list(segments)  # Force evaluation
            
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper model loaded and warmed up in {load_time:.2f} seconds")
            
            self.update_status({'whisper': 'loaded', 'whisper_time': load_time})
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            self.update_status({'whisper': 'failed', 'whisper_error': str(e)})
            return False
            
    def preload_llama(self):
        """Preload LLaMA model into memory"""
        try:
            logger.info(f"Loading LLaMA model: {LLAMA_MODEL_PATH}")
            start_time = time.time()
            
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=2048,
                n_batch=512,
                n_threads=4,  # Use all 4 cores on RPi5
                temperature=0.7,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                verbose=False,
                mlock=True,  # Lock model in RAM
                use_mmap=True  # Use memory mapping for faster loading
            )
            
            # Warm up the model with a dummy query
            logger.info("Warming up LLaMA model...")
            warmup_prompt = "Hello, this is a test. Please respond briefly."
            response = self.llama_model(warmup_prompt, max_tokens=10)
            
            load_time = time.time() - start_time
            logger.info(f"✅ LLaMA model loaded and warmed up in {load_time:.2f} seconds")
            
            self.update_status({'llama': 'loaded', 'llama_time': load_time})
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLaMA model: {e}")
            self.update_status({'llama': 'failed', 'llama_error': str(e)})
            return False
            
    def preload_piper(self):
        """Preload Piper TTS voice model"""
        try:
            logger.info(f"Preloading Piper voice model: {VOICE_PATH}")
            start_time = time.time()
            
            # Load the voice model file into memory
            if os.path.exists(VOICE_PATH):
                with open(VOICE_PATH, 'rb') as f:
                    voice_data = f.read()
                    
                # Cache the voice model in shared memory
                cache_path = f"{MODEL_CACHE_DIR}/voice_model.onnx"
                with open(cache_path, 'wb') as f:
                    f.write(voice_data)
                    
                # Test Piper with the cached model
                test_text = "System initialized"
                test_output = "/tmp/test_piper.wav"
                
                piper_command = [
                    PIPER_EXECUTABLE,
                    "--model", cache_path,
                    "--config", CONFIG_PATH,
                    "--output_file", test_output
                ]
                
                result = subprocess.run(
                    piper_command,
                    input=test_text,
                    text=True,
                    capture_output=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self.voice_model_loaded = True
                    load_time = time.time() - start_time
                    logger.info(f"✅ Piper voice model loaded and tested in {load_time:.2f} seconds")
                    
                    # Clean up test file
                    if os.path.exists(test_output):
                        os.remove(test_output)
                        
                    self.update_status({'piper': 'loaded', 'piper_time': load_time})
                    return True
                else:
                    raise Exception(f"Piper test failed: {result.stderr}")
                    
            else:
                raise FileNotFoundError(f"Voice model not found: {VOICE_PATH}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Piper voice model: {e}")
            self.update_status({'piper': 'failed', 'piper_error': str(e)})
            return False
            
    def preload_face_encodings(self):
        """Preload face recognition encodings"""
        try:
            logger.info(f"Loading face encodings from: {ENCODINGS_FILE}")
            start_time = time.time()
            
            if os.path.exists(ENCODINGS_FILE):
                with open(ENCODINGS_FILE, "rb") as f:
                    self.face_encodings = pickle.loads(f.read())
                    
                # Cache in shared memory for faster access
                cache_path = f"{MODEL_CACHE_DIR}/face_encodings.pkl"
                with open(cache_path, "wb") as f:
                    pickle.dump(self.face_encodings, f)
                    
                num_faces = len(self.face_encodings.get("encodings", []))
                load_time = time.time() - start_time
                logger.info(f"✅ Loaded {num_faces} face encodings in {load_time:.2f} seconds")
                
                self.update_status({'face_encodings': 'loaded', 'faces': num_faces, 'face_time': load_time})
                return True
            else:
                logger.warning("Face encodings file not found")
                self.update_status({'face_encodings': 'not_found'})
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load face encodings: {e}")
            self.update_status({'face_encodings': 'failed', 'face_error': str(e)})
            return False
            
    def preload_response_files(self):
        """Preload and cache response text files"""
        try:
            logger.info("Caching response files...")
            start_time = time.time()
            
            response_files = [
                "jokes.txt", "fun_facts.txt", "greeting_responses.txt",
                "listening_responses.txt", "waiting_responses.txt",
                "warning_responses.txt", "bored_responses.txt",
                "visitor_greeting_responses.txt", "bored_responses_generic.txt",
                "waiting_responses_generic.txt"
            ]
            
            cached_count = 0
            for filename in response_files:
                filepath = f"/home/nickspi5/Chatty_AI/{filename}"
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Cache in shared memory
                    cache_path = f"{MODEL_CACHE_DIR}/{filename}"
                    with open(cache_path, 'w') as f:
                        f.write(content)
                    cached_count += 1
                    
            load_time = time.time() - start_time
            logger.info(f"✅ Cached {cached_count} response files in {load_time:.2f} seconds")
            
            self.update_status({'response_files': 'cached', 'files': cached_count, 'cache_time': load_time})
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cache response files: {e}")
            self.update_status({'response_files': 'failed', 'cache_error': str(e)})
            return False
            
    def optimize_system(self):
        """Apply system optimizations for better performance"""
        try:
            logger.info("Applying system optimizations...")
            
            # Set CPU governor to performance mode
            try:
                subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'], 
                             capture_output=True, timeout=5)
                logger.info("✅ CPU governor set to performance mode")
            except:
                logger.warning("Could not set CPU governor (requires sudo)")
                
            # Increase file descriptor limits
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
                logger.info("✅ File descriptor limit increased")
            except:
                logger.warning("Could not increase file descriptor limit")
                
            # Clear system caches to make room for models
            try:
                subprocess.run(['sync'], capture_output=True, timeout=5)
                logger.info("✅ System caches synchronized")
            except:
                pass
                
            return True
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
            return False
            
    def keep_alive_loop(self):
        """Keep models in memory and prevent them from being paged out"""
        logger.info("Starting keep-alive loop...")
        
        while self.running:
            try:
                # Touch each model periodically to keep in RAM
                if self.whisper_model:
                    # Access model attributes to keep in memory
                    _ = self.whisper_model.model
                    
                if self.llama_model:
                    # Access model context to keep in memory
                    _ = self.llama_model.n_ctx
                    
                # Update heartbeat
                self.update_status({'heartbeat': time.time(), 'status': 'running'})
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
                time.sleep(30)
                
    def run(self):
        """Main preloader process"""
        logger.info("="*60)
        logger.info("🚀 Chatty AI Model Preloader Starting...")
        logger.info("="*60)
        
        # Apply system optimizations
        self.optimize_system()
        
        # Preload all models
        total_start = time.time()
        
        self.preload_whisper()
        self.preload_llama()
        self.preload_piper()
        self.preload_face_encodings()
        self.preload_response_files()
        
        total_time = time.time() - total_start
        
        logger.info("="*60)
        logger.info(f"✨ All models preloaded in {total_time:.2f} seconds")
        logger.info("Models will remain in memory for fast access")
        logger.info("="*60)
        
        # Update final status
        self.update_status({
            'status': 'ready',
            'total_load_time': total_time,
            'timestamp': time.time()
        })
        
        # Keep models in memory
        self.keep_alive_loop()
        
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up preloader resources...")
        self.running = False
        
        # Update status
        self.update_status({'status': 'stopped'})
        
        # Models will be garbage collected
        self.whisper_model = None
        self.llama_model = None
        self.face_encodings = None


def main():
    """Main entry point"""
    try:
        preloader = ChattyAIPreloader()
        preloader.run()
    except KeyboardInterrupt:
        logger.info("Preloader interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()