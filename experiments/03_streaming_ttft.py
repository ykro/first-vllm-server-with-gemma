"""03. What the user feels: time to first token (TTFT) and tokens per second.

Asks the same question twice: once waiting for the whole answer, once with streaming.
The total time is about the same, but with streaming the first words show up after the
TTFT instead of after the whole answer.

    python experiments/03_streaming_ttft.py [--mock]
"""

from __future__ import annotations

from vllm_lab import chat_once, cli, fresh, make_client, stream_chat

PROMPT = (
    "En un párrafo, para alguien que programa backend, explica qué es un token en un "
    "modelo de lenguaje y por qué el modelo escribe un token a la vez."
)


def run(base_url: str) -> dict:
    client = make_client(base_url)
    stream_chat(client, "Hola", max_tokens=8)  # warmup: first request after a cold start
    blocking = chat_once(client, fresh(PROMPT), max_tokens=200)
    streamed = stream_chat(client, fresh(PROMPT), max_tokens=200)
    return {
        "blocking_wait_s": round(blocking["wall_s"], 3),
        "streaming_ttft_s": round(streamed.ttft_s, 3),
        "streaming_total_s": round(streamed.total_s, 3),
        "completion_tokens": streamed.completion_tokens,
        "stream_chunks": streamed.chunks,
        "decode_tokens_per_s": round(streamed.decode_tokens_per_s, 1),
        "first_words": streamed.text[:120],
        "first_word_sooner_x": round(blocking["wall_s"] / max(streamed.ttft_s, 1e-9), 1),
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("Blocking vs streaming", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
