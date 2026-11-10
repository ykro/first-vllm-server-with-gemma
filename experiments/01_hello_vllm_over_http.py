"""01. Your model is just an HTTP endpoint.

Talks to vLLM with nothing but the standard library: ask which model is loaded, send one
question, read the answer and the token counts. No SDK, no framework.

    python experiments/01_hello_vllm_over_http.py          # live server (tunnel open)
    python experiments/01_hello_vllm_over_http.py --mock   # no GPU
"""

from __future__ import annotations

import json
import time
import urllib.request

from vllm_lab import cli

QUESTION = "En una frase, ¿qué hace un servidor de inferencia?"


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def run(base_url: str) -> dict:
    with urllib.request.urlopen(base_url + "/models", timeout=60) as resp:
        models = json.loads(resp.read())
    model = models["data"][0]
    body = {
        "model": model["id"],
        "messages": [{"role": "user", "content": QUESTION}],
        "max_tokens": 80,
        "temperature": 0,
        "seed": 7,
    }
    t0 = time.perf_counter()
    answer = post_json(base_url + "/chat/completions", body)
    wall = time.perf_counter() - t0
    return {
        "served_model_name": model["id"],
        "weights": model.get("root"),
        "max_model_len": model.get("max_model_len"),
        "request": body,
        "answer": answer["choices"][0]["message"]["content"].strip(),
        "finish_reason": answer["choices"][0]["finish_reason"],
        "usage": answer["usage"],
        "wall_s": round(wall, 3),
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    cli.show("GET /v1/models", {k: result[k] for k in ("served_model_name", "weights",
                                                        "max_model_len")})
    cli.show("POST /v1/chat/completions", result["request"])
    cli.show("Answer", {k: result[k] for k in ("answer", "finish_reason", "usage", "wall_s")})


if __name__ == "__main__":
    main()
