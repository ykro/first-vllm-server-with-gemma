"""Cuánto cuesta un millón de tokens en tu L4, con la corrida real guardada."""
import json
from pathlib import Path

precio = json.loads(Path("benchmarks/pricing.json").read_text())
corrida = json.loads(Path("benchmarks/latest_results.json").read_text())
por_hora = precio["usd_per_hour"]
print(f"la L4 cuesta {por_hora} dólares la hora")

for fila in corrida["experiments"]["05_cost_per_million_tokens"]["rows"]:
    tokens_por_hora = fila["output_tokens_per_s"] * 3600
    dolares = por_hora / tokens_por_hora * 1_000_000
    print(f"{fila['users']} a la vez: {fila['output_tokens_per_s']} tokens/s, "
          f"{tokens_por_hora:,.0f} por hora, ${dolares:.2f} por millón")
