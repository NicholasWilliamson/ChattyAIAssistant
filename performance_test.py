#!/usr/bin/env python3
"""
Chatty AI Performance Benchmark
Tests wake word detection, STT, TTS, and LLM response times
"""

import time
import sys
import os

# Ensure UTF-8 output
os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, '/home/nickspi5/Chatty_AI')

def benchmark(name, func, *args, iterations=3):
    """Run a function multiple times and report timing"""
    times = []
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    for i in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
    
    avg = sum(times) / len(times)
    print(f"  Average: {avg:.3f}s")
    print(f"  Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    return avg

def test_whisper_loading():
    """Test Whisper model loading time"""
    import whisper
    model = whisper.load_model("base")
    return model

def test_whisper_transcription(model, audio_file):
    """Test speech-to-text transcription"""
    result = model.transcribe(audio_file)
    return result["text"]

def test_llama_loading():
    """Test LLaMA model loading time"""
    from llama_cpp import Llama
    model_path = "/home/nickspi5/Chatty_AI/tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
    return llm

def test_llama_inference(llm, prompt):
    """Test LLM response generation"""
    response = llm(prompt, max_tokens=50, stop=["\n"])
    return response

def test_piper_tts(text):
    """Test text-to-speech generation"""
    import subprocess
    voice_model = "/home/nickspi5/Chatty_AI/voices/en_US-amy-low/en_US-amy-low.onnx"
    output_file = "/tmp/tts_test.wav"
    
    cmd = f'echo "{text}" | /home/nickspi5/Chatty_AI/piper/piper --model {voice_model} --output_file {output_file}'
    subprocess.run(cmd, shell=True, capture_output=True)
    return output_file

def main():
    print("\n" + "="*60)
    print("  CHATTY AI PERFORMANCE BENCHMARK")
    print("="*60)
    
    # System info
    import platform
    import psutil
    print(f"\nSystem: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    
    # Check Hailo status
    print("\nHailo AI HAT Status:")
    import subprocess
    result = subprocess.run(['hailortcli', 'fw-control', 'identify'], capture_output=True, text=True)
    if result.returncode == 0:
        print("  [OK] Hailo AI HAT is working")
        print(f"  {result.stdout.strip()}")
    else:
        print("  [NOT WORKING] Hailo AI HAT firmware not loaded")
        print("  Run: sudo apt install hailo-all && sudo reboot")
    
    results = {}
    
    # Test 1: Whisper Loading
    print("\n[1/5] Testing Whisper Model Loading...")
    try:
        whisper_load_time = benchmark("Whisper Load", test_whisper_loading, iterations=1)
        results['whisper_load'] = whisper_load_time
        
        # Keep model for transcription test
        import whisper
        whisper_model = whisper.load_model("base")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['whisper_load'] = None
        whisper_model = None
    
    # Test 2: Speech-to-Text
    print("\n[2/5] Testing Speech-to-Text (Whisper)...")
    test_audio = "/home/nickspi5/Chatty_AI/test_audio.wav"
    try:
        if os.path.exists(test_audio) and whisper_model:
            stt_time = benchmark("STT Transcription", test_whisper_transcription, whisper_model, test_audio, iterations=3)
            results['stt'] = stt_time
        else:
            # Try alternative audio files
            alt_files = [
                "/home/nickspi5/Chatty_AI/user_audio.wav",
                "/home/nickspi5/Chatty_AI/wake_word_audio.wav",
                "/home/nickspi5/Chatty_AI/record.wav"
            ]
            found_audio = None
            for f in alt_files:
                if os.path.exists(f):
                    found_audio = f
                    break
            
            if found_audio and whisper_model:
                print(f"  Using: {found_audio}")
                stt_time = benchmark("STT Transcription", test_whisper_transcription, whisper_model, found_audio, iterations=3)
                results['stt'] = stt_time
            else:
                print(f"  SKIPPED: No test audio found")
                results['stt'] = None
    except Exception as e:
        print(f"  ERROR: {e}")
        results['stt'] = None
    
    # Test 3: LLaMA Loading
    print("\n[3/5] Testing LLaMA Model Loading...")
    try:
        llama_load_time = benchmark("LLaMA Load", test_llama_loading, iterations=1)
        results['llama_load'] = llama_load_time
        
        # Keep model for inference test
        from llama_cpp import Llama
        model_path = "/home/nickspi5/Chatty_AI/tinyllama-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        results['llama_load'] = None
        llm = None
    
    # Test 4: LLM Inference
    print("\n[4/5] Testing LLM Inference...")
    try:
        if llm:
            test_prompt = "Hello, how are you today?"
            llm_time = benchmark("LLM Inference", test_llama_inference, llm, test_prompt, iterations=3)
            results['llm'] = llm_time
        else:
            print("  SKIPPED: LLaMA model not loaded")
            results['llm'] = None
    except Exception as e:
        print(f"  ERROR: {e}")
        results['llm'] = None
    
    # Test 5: Text-to-Speech
    print("\n[5/5] Testing Text-to-Speech (Piper)...")
    try:
        tts_time = benchmark("TTS Generation", test_piper_tts, "Hello, this is a test of the text to speech system.", iterations=3)
        results['tts'] = tts_time
    except Exception as e:
        print(f"  ERROR: {e}")
        results['tts'] = None
    
    # Summary
    print("\n" + "="*60)
    print("  PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\n{'Component':<25} {'Time (seconds)':<15} {'Status'}")
    print("-"*50)
    
    for name, time_val in results.items():
        if time_val is not None:
            status = "[OK]" if time_val < 5 else "[SLOW]"
            print(f"{name:<25} {time_val:<15.3f} {status}")
        else:
            print(f"{name:<25} {'N/A':<15} [FAILED]")
    
    print("\n" + "="*60)
    print("  RECOMMENDATIONS")
    print("="*60)
    
    if results.get('llm') and results['llm'] > 2:
        print("- LLM inference is slow - consider using more CPU threads")
        print("  or enabling Hailo AI acceleration")
    
    if results.get('stt') and results['stt'] > 3:
        print("- STT is slow - consider using whisper 'tiny' model instead of 'base'")
    
    if results.get('tts') and results['tts'] > 1:
        print("- TTS is slow - check Piper voice model")
    
    if results.get('whisper_load') is None:
        print("- Whisper not installed. Run: pip install openai-whisper")
    
    print("\nDone!")

if __name__ == "__main__":
    main()