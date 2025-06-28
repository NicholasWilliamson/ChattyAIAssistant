import requests

response = requests.post("http://localhost:8080/v1/chat/completions", json={
    "model": "tinyllama",
    "messages": [
        {"role": "system", "content": "You are a friendly offline AI assistant."},
        {"role": "user", "content": "Hey Chatty. How are you today?"}
    ]
})

print(response.json()["choices"][0]["message"]["content"])