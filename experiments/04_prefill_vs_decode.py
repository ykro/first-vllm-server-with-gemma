"""04. Reading is cheap, writing is expensive: prefill vs decode.

Two sweeps. (a) Longer prompts, one-token answers: TTFT grows with the prompt, but the GPU
reads thousands of prompt tokens per second because it processes them all at once.
(b) Short prompt, longer answers: total time grows almost linearly, because every new token
needs its own step.

    python experiments/04_prefill_vs_decode.py [--mock]
"""

from __future__ import annotations

from vllm_lab import cli, fresh, make_client, stream_chat

PARAGRAPH = (
    "El equipo de logística registra cada envío con su origen, destino, peso, fecha de "
    "salida y fecha de entrega. Algunos envíos llegan tarde por lluvias o por aduanas. "
)
PROMPT_REPEATS = (2, 20, 80, 200)  # the last one is about 7,200 tokens, under 8,192
OUTPUT_LENGTHS = (16, 64, 256)


def run(base_url: str) -> dict:
    client = make_client(base_url)
    stream_chat(client, "Hola", max_tokens=8)  # warmup
    prefill = []
    for reps in PROMPT_REPEATS:
        prompt = fresh(PARAGRAPH * reps + "\nResponde solo: listo.")
        t = stream_chat(client, prompt, max_tokens=1)
        prefill.append({"prompt_tokens": t.prompt_tokens, "ttft_s": round(t.ttft_s, 4),
                        "prefill_tokens_per_s": round(t.prefill_tokens_per_s)})
    decode = []
    for n in OUTPUT_LENGTHS:
        t = stream_chat(client, fresh("Cuenta una historia larga sobre un envío."), max_tokens=n,
                        ignore_eos=True)
        decode.append({"completion_tokens": t.completion_tokens, "total_s": round(t.total_s, 3),
                       "decode_tokens_per_s": round(t.decode_tokens_per_s, 1)})
    # The shortest prompt's TTFT is mostly network round trip and scheduling. The slope
    # between the shortest and the longest prompt is the prefill speed without that floor.
    first, last = prefill[0], prefill[-1]
    marginal = (last["prompt_tokens"] - first["prompt_tokens"]) / max(
        last["ttft_s"] - first["ttft_s"], 1e-9)
    rate = decode[-1]["decode_tokens_per_s"]
    return {"prefill_sweep": prefill, "decode_sweep": decode,
            "ttft_floor_s": first["ttft_s"],
            "prefill_marginal_tokens_per_s": round(marginal),
            "prefill_vs_decode_x": round(marginal / max(rate, 1e-9), 1)}


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("Prefill vs decode", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
