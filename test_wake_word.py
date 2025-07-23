#!/usr/bin/env python3
"""
test_wake_word.py
Simple test script for wake word detection functionality
"""

import threading
import time
import numpy as np
import sounddevice as sd
import pvporcupine

class WakeWordTester:
    def __init__(self):
        self.porcupine = None
        self.running = False
        self.init_porcupine()
    
    def init_porcupine(self):
        """Initialize Porcupine wake word detector"""
        try:
            # Using built-in "Hey Google" wake word
            self.porcupine = pvporcupine.create(
                keywords=["hey google"],
                sensitivities=[0.5]
            )
            print("Wake word detection initialized successfully!")
            print(f"Frame length: {self.porcupine.frame_length}")
            print(f"Sample rate: {self.porcupine.sample_rate}")
            print(f"Version: {self.porcupine.version}")
        except Exception as e:
            print(f"Wake word detection initialization failed: {e}")
            return False
        return True
    
    def start_test(self):
        """Start wake word detection test"""
        if not self.porcupine:
            print("Porcupine not initialized")
            return
        
        print("\nStarting wake word detection test...")
        print("Say 'Hey Google' to test wake word detection")
        print("Press Ctrl+C to stop")
        
        self.running = True
        
        try:
            def audio_callback(indata, frames, time, status):
                if self.running and self.porcupine:
                    try:
                        pcm = indata[:, 0].astype(np.int16)
                        keyword_index = self.porcupine.process(pcm)
                        if keyword_index >= 0:
                            print(f"\n🎉 Wake word detected! Keyword index: {keyword_index}")
                            print("Wake word detection is working correctly!")
                    except Exception as e:
                        print(f"Processing error: {e}")
            
            with sd.InputStream(
                channels=1,
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                callback=audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        except Exception as e:
            print(f"Audio stream error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.porcupine:
            self.porcupine.delete()
            print("Wake word detector cleaned up")

if __name__ == "__main__":
    tester = WakeWordTester()
    if tester.porcupine:
        tester.start_test()
    else:
        print("Cannot run test - wake word detection failed to initialize")