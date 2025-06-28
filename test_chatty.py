import requests

try:
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "Hermes-3-Llama-3.2-3B-Q4_K_M.gguf",
            "max_tokens": 128,
            "messages": [
                {"role": "system", "content": "You're a helpful local assistant."},
                {"role": "user", "content": "Hi Chatty, what can you do?"}
            ]
        },
        timeout=60  # timeout in seconds
    )
    print(response.json()["choices"][0]["message"]["content"])
except requests.exceptions.Timeout:
    print("Request timed out. Try a smaller model or fewer tokens.")