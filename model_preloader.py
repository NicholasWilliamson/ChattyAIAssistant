#!/usr/bin/env python3
"""
model_preloader.py - Preload and keep AI models warm in memory
This service loads all AI models at boot time and keeps them ready for the main application
"""

import os
import sys
import time
import logging
import threading
import signal
import pickle
import json
from datetime import datetime
from faster_whisper import WhisperModel
from llama_cpp import Llama
import face_recognition

# Configuration - same as your main app
WHISPER_MODEL_SIZE = "base"
LLAMA_MODEL_PATH = "tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
ENCODINGS_FILE = "encodings.pickle"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/nickspi5/Chatty_AI/model_preloader.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelPreloader:
    def __init__(self):
        self.whisper_model = None
        self.llama_model = None
        self.known_encodings = []
        self.known_names = []
        self.running = False
        self.warmup_thread = None
        
    def preload_whisper_model(self):
        """Preload Whisper model"""
        try:
            logger.info("🎤 Loading Whisper model...")
            start_time = time.time()
            self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper model loaded successfully in {load_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def preload_llama_model(self):
        """Preload LLaMA model"""
        try:
            logger.info("🤖 Loading LLaMA model...")
            start_time = time.time()
            self.llama_model = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=2048,
                temperature=0.7,
                repeat_penalty=1.1,
                n_gpu_layers=0,
                verbose=False
            )
            load_time = time.time() - start_time
            logger.info(f"✅ LLaMA model loaded successfully in {load_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load LLaMA model: {e}")
            return False
    
    def preload_face_encodings(self):
        """Preload facial recognition encodings"""
        try:
            logger.info("👤 Loading facial recognition encodings...")
            start_time = time.time()
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.loads(f.read())
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded {len(self.known_encodings)} face encodings in {load_time:.2f}s")
            return True
        except FileNotFoundError:
            logger.error(f"❌ Encodings file '{ENCODINGS_FILE}' not found!")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load encodings: {e}")
            return False
    
    def warmup_models(self):
        """Warm up models with test inputs"""
        logger.info("🔥 Warming up models...")
        
        # Warm up Whisper model
        if self.whisper_model:
            try:
                logger.info("🎤 Warming up Whisper model...")
                # Create a small test audio file
                import numpy as np
                import soundfile as sf
                test_audio = np.random.random(16000).astype(np.float32) * 0.01  # 1 second of quiet noise
                sf.write("test_warmup.wav", test_audio, 16000)
                
                segments, _ = self.whisper_model.transcribe("test_warmup.wav")
                list(segments)  # Force processing
                
                os.remove("test_warmup.wav")
                logger.info("✅ Whisper model warmed up")
            except Exception as e:
                logger.warning(f"⚠️ Whisper warmup failed: {e}")
        
        # Warm up LLaMA model
        if self.llama_model:
            try:
                logger.info("🤖 Warming up LLaMA model...")
                test_prompt = "Test prompt for warmup.\nUser: Hello\nAssistant: "
                result = self.llama_model(test_prompt, max_tokens=10)
                logger.info("✅ LLaMA model warmed up")
            except Exception as e:
                logger.warning(f"⚠️ LLaMA warmup failed: {e}")
    
    def continuous_warmup_loop(self):
        """Keep models warm with periodic test calls"""
        while self.running:
            try:
                # Every 5 minutes, do a small test to keep models warm
                time.sleep(300)  # 5 minutes
                
                if not self.running:
                    break
                    
                logger.info("🔥 Periodic model warmup...")
                
                # Quick Whisper test
                if self.whisper_model:
                    try:
                        import numpy as np
                        import soundfile as sf
                        test_audio = np.random.random(8000).astype(np.float32) * 0.005  # Very quiet
                        sf.write("periodic_test.wav", test_audio, 16000)
                        segments, _ = self.whisper_model.transcribe("periodic_test.wav")
                        list(segments)
                        os.remove("periodic_test.wav")
                    except:
                        pass
                
                # Quick LLaMA test
                if self.llama_model:
                    try:
                        self.llama_model("Test", max_tokens=1)
                    except:
                        pass
                        
                logger.info("🔥 Periodic warmup completed")
                
            except Exception as e:
                logger.error(f"❌ Warmup loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start_preloader(self):
        """Start the model preloader service"""
        logger.info("🚀 Starting Model Preloader Service")
        logger.info("=" * 50)
        
        total_start = time.time()
        
        # Load all models
        success_count = 0
        
        if self.preload_whisper_model():
            success_count += 1
        
        if self.preload_llama_model():
            success_count += 1
            
        if self.preload_face_encodings():
            success_count += 1
        
        total_load_time = time.time() - total_start
        
        if success_count == 3:
            logger.info(f"🎉 All models loaded successfully in {total_load_time:.2f}s")
            
            # Warm up models
            self.warmup_models()
            
            # Start continuous warmup
            self.running = True
            self.warmup_thread = threading.Thread(target=self.continuous_warmup_loop, daemon=True)
            self.warmup_thread.start()
            
            logger.info("✅ Model Preloader Service is ready!")
            logger.info("🔥 Models are warm and ready for use")
            logger.info("=" * 50)
            
            # Write status file to indicate models are ready
            with open("/tmp/chatty_ai_models_ready", "w") as f:
                f.write(f"Models loaded at: {datetime.now().isoformat()}\n")
                f.write(f"Load time: {total_load_time:.2f}s\n")
                f.write(f"Whisper: {'✅' if self.whisper_model else '❌'}\n")
                f.write(f"LLaMA: {'✅' if self.llama_model else '❌'}\n")
                f.write(f"Face encodings: {'✅' if self.known_encodings else '❌'}\n")
            
            return True
        else:
            logger.error(f"❌ Failed to load some models ({success_count}/3 successful)")
            return False
    
    def stop_preloader(self):
        """Stop the preloader service"""
        logger.info("🛑 Stopping Model Preloader Service")
        self.running = False
        
        # Remove status file
        try:
            os.remove("/tmp/chatty_ai_models_ready")
        except:
            pass
    
    def get_models(self):
        """Return the loaded models for use by main application"""
        return {
            'whisper_model': self.whisper_model,
            'llama_model': self.llama_model,
            'known_encodings': self.known_encodings,
            'known_names': self.known_names
        }

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}")
    if 'preloader' in globals():
        preloader.stop_preloader()
    sys.exit(0)

def main():
    """Main function"""
    global preloader
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        preloader = ModelPreloader()
        
        if preloader.start_preloader():
            # Keep the service running
            while preloader.running:
                time.sleep(1)
        else:
            logger.error("❌ Failed to start preloader service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
        if 'preloader' in locals():
            preloader.stop_preloader()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()