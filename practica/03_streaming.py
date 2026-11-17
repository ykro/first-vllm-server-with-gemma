"""Lo que siente tu usuario: la primera palabra (TTFT) y los tokens por segundo."""
import time

from openai import OpenAI

cliente = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
inicio = time.perf_counter()
stream = cliente.chat.completions.create(
    model="gemma", max_tokens=100, stream=True, extra_body={"ignore_eos": True},
    messages=[{"role": "user", "content": "Explica qué es un token."}])
llegadas = []
for pedazo in stream:
    if pedazo.choices and pedazo.choices[0].delta.content:
        llegadas.append(time.perf_counter() - inicio)
ttft = llegadas[0]
por_segundo = (len(llegadas) - 1) / (llegadas[-1] - ttft)
print(f"primer token a los {ttft:.3f} s")
print(f"{len(llegadas)} tokens, {por_segundo:.1f} tokens por segundo")
print(f"respuesta completa a los {llegadas[-1]:.2f} s")
