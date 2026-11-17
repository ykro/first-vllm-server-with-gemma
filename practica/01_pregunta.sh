# Una pregunta a vLLM con curl: el mismo JSON que le mandas a la API de OpenAI.
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [
      {"role": "user",
       "content": "En una frase, ¿qué hace un servidor de inferencia?"}
    ],
    "max_tokens": 80,
    "temperature": 0
  }' | python3 -m json.tool --no-ensure-ascii
