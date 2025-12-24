#!/usr/bin/env python3
"""
Chatty AI Performance Benchmark
Tests wake word detection, STT, TTS, and LLM response times
"""

import time
import sys
import os

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

def test_faster_whisper_loading():
    """Test faster-whisper model loading time"""
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    return model

def test_faster_whisper_transcription(model, audio_file):
    """Test speech-to-text transcription with faster-whisper"""
    segments, info = model.transcribe(audio_file, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text

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
        # Print just the key info
        for line in result.stdout.strip().split('\n'):
            if any(x in line for x in ['Firmware Version', 'Device Architecture', 'Product Name']):
                print(f"    {line.strip()}")
    else:
        print("  [NOT WORKING] Hailo AI HAT firmware not loaded")
    
    results = {}
    
    # Test 1: Faster-Whisper Loading
    print("\n[1/5] Testing Faster-Whisper Model Loading...")
    whisper_model = None
    try:
        whisper_load_time = benchmark("Faster-Whisper Load", test_faster_whisper_loading, iterations=1)
        results['whisper_load'] = whisper_load_time
        
        # Keep model for transcription test
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['whisper_load'] = None
    
    # Test 2: Speech-to-Text
    print("\n[2/5] Testing Speech-to-Text (Faster-Whisper)...")
    try:
        # Find an audio file to test with
        audio_files = [
            "/home/nickspi5/Chatty_AI/test_audio.wav",
            "/home/nickspi5/Chatty_AI/user_audio.wav",
            "/home/nickspi5/Chatty_AI/wake_word_audio.wav",
            "/home/nickspi5/Chatty_AI/record.wav",
            "/home/nickspi5/Chatty_AI/user_input.wav"
        ]
        
        found_audio = None
        for f in audio_files:
            if os.path.exists(f) and os.path.getsize(f) > 1000:
                found_audio = f
                break
        
        if found_audio and whisper_model:
            print(f"  Using: {found_audio}")
            stt_time = benchmark("STT Transcription", test_faster_whisper_transcription, whisper_model, found_audio, iterations=3)
            results['stt'] = stt_time
        else:
            print(f"  SKIPPED: No suitable test audio found or model not loaded")
            results['stt'] = None
    except Exception as e:
        print(f"  ERROR: {e}")
        results['stt'] = None
    
    # Test 3: LLaMA Loading
    print("\n[3/5] Testing LLaMA Model Loading...")
    llm = None
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
    print("-"*55)
    
    targets = {
        'whisper_load': 5.0,
        'stt': 3.0,
        'llama_load': 1.0,
        'llm': 2.0,
        'tts': 1.0
    }
    
    for name, time_val in results.items():
        if time_val is not None:
            target = targets.get(name, 5.0)
            status = "[OK]" if time_val < target else "[SLOW]"
            print(f"{name:<25} {time_val:<15.3f} {status}")
        else:
            print(f"{name:<25} {'N/A':<15} [FAILED]")
    
    # Total end-to-end estimate
    print("\n" + "-"*55)
    valid_times = [v for v in results.values() if v is not None]
    if valid_times:
        # Estimate total response time (STT + LLM + TTS)
        stt = results.get('stt', 0) or 0
        llm = results.get('llm', 0) or 0
        tts = results.get('tts', 0) or 0
        total = stt + llm + tts
        print(f"{'Estimated Response Time':<25} {total:<15.3f}")
    
    print("\n" + "="*60)
    print("  RECOMMENDATIONS")
    print("="*60)
    
    if results.get('llm') and results['llm'] > 2:
        print("- LLM inference is slow on first run (cache warming)")
        print("  Consider keeping the model warm in memory")
    
    if results.get('stt') and results['stt'] > 3:
        print("- STT is slow - consider using 'tiny' model instead of 'base'")
    
    if results.get('tts') and results['tts'] > 1:
        print("- TTS could be faster with a smaller voice model")
    
    print("\n- To use Hailo AI HAT acceleration, models need to be")
    print("  converted to HEF format (Hailo Executable Format)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
