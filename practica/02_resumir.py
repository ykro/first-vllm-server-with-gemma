"""Una función de tu app con el paquete de OpenAI, apuntada a tu servidor."""
from openai import OpenAI

cliente = OpenAI(base_url="http://localhost:8000/v1", api_key="local")


def resumir(ticket):
    r = cliente.chat.completions.create(
        model="gemma",
        messages=[
            {"role": "system", "content": "Resume el ticket en una línea."},
            {"role": "user", "content": ticket}],
        max_tokens=60, temperature=0)
    print("tokens de entrada:", r.usage.prompt_tokens)
    print("tokens de salida:", r.usage.completion_tokens)
    return r.choices[0].message.content


ticket = "Mis audífonos llegaron con la caja abierta y el lado izquierdo no suena."
print(resumir(ticket))
