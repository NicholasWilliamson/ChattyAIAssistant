#!/usr/bin/env python3
"""
run_tinyllama_tts.py

Interact with TinyLLaMA and speak the response using eSpeak.

Usage:
    python run_tinyllama_tts.py \
        --model-path models/tinyllama-1.1b-chat.Q4_K_M.gguf \
        --prompt "What is the capital of Japan?" \
        [--max-tokens 50] \
        [--temperature 0.7] \
        [--repeat-penalty 1.1] \
        [--context-length 2048]
"""

import argparse
import sys
import subprocess
from llama_cpp import Llama

def parse_args():
    parser = argparse.ArgumentParser(description="Interact with TinyLLaMA and speak response via eSpeak.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to TinyLLaMA GGUF model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (creativity)")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length in tokens")
    return parser.parse_args()

def speak_text(text):
    """Speak the response using eSpeak"""
    try:
        subprocess.run(["espeak", "-ven+f3", "-s", "150", text], check=True)
    except FileNotFoundError:
        print("❌ eSpeak not found. Please install it using: sudo apt install espeak", file=sys.stderr)
    except Exception as e:
        print(f"Error using TTS: {e}", file=sys.stderr)

def main():
    args = parse_args()

    print(f"🔍 Loading TinyLLaMA model from: {args.model_path}")
    try:
        llm = Llama(
            model_path=args.model_path,
            n_ctx=args.context_length,
            temperature=args.temperature,
            repeat_penalty=args.repeat_penalty
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Prompt formatting
    prompt = (
        "[INST] <<SYS>>"
        "You are a helpful assistant that answers conversationally but precisely."
        "<</SYS>>\n\n"
        + args.prompt +
        " [/INST]\nA:"
    )

    print(f"🧠 Prompt: {args.prompt}")
    print(f"🕐 Generating response with max {args.max_tokens} tokens...\n")

    try:
        response = llm(prompt, max_tokens=args.max_tokens)
        text = response["choices"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Error during inference: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"🤖 TinyLLaMA says:\n{text}\n")

    # Speak response
    speak_text(text)

if __name__ == "__main__":
    main()