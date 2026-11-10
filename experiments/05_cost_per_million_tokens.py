"""05. What a million tokens costs on your own GPU, alone and with company.

Measures generated tokens per second with 1 user and with 8 users at the same time, and
turns that into dollars per million output tokens using the hourly price of the Cloud Run
instance (benchmarks/pricing.json, a list price you should re-check). One user leaves the
GPU mostly idle; with eight, the same instance writes several times more tokens per hour.
That gap is the topic of C3.

    python experiments/05_cost_per_million_tokens.py [--mock]
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vllm_lab import cli, fresh, make_client, stream_chat

PRICING = Path(__file__).resolve().parent.parent / "benchmarks" / "pricing.json"
OUT_TOKENS = 256


def throughput(base_url: str, users: int) -> float:
    client = make_client(base_url)
    prompts = [fresh(f"Usuario {i}: cuenta una historia sobre un envío número {i}.") for i in
               range(users)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as pool:
        results = list(pool.map(lambda p: stream_chat(client, p, max_tokens=OUT_TOKENS,
                                                      ignore_eos=True), prompts))
    wall = time.perf_counter() - t0
    return sum(r.completion_tokens for r in results) / wall


def run(base_url: str) -> dict:
    price = json.loads(PRICING.read_text())
    per_hour = price["usd_per_hour"]
    stream_chat(make_client(base_url), "Hola", max_tokens=8)  # warmup
    rows = []
    for users in (1, 8):
        tps = throughput(base_url, users)
        rows.append({"users": users, "output_tokens_per_s": round(tps, 1),
                     "usd_per_million_output_tokens": round(per_hour / (tps * 3600) * 1e6, 2)})
    return {"instance": price["instance"], "usd_per_hour": per_hour,
            "price_checked": price["checked"], "rows": rows,
            "eight_users_gain_x": round(rows[1]["output_tokens_per_s"] /
                                        rows[0]["output_tokens_per_s"], 1)}


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    cli.show("Cost per million output tokens", run(cli.resolve(args)))


if __name__ == "__main__":
    main()
