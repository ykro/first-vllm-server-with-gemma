"""The Outside Judge for your first vLLM server: runs the five experiments once and checks what must hold.

    python verify_first_server.py          # live vLLM through the tunnel (real numbers)
    python verify_first_server.py --mock   # fake server, checks the logic without a GPU

The checks are about behavior (the answer exists, streaming shows words before the end,
long prompts take longer to start, eight users get more tokens per second than one), not
about exact numbers. Times vary from run to run; the JSON keeps the raw values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vllm_lab.client import BASE_URL

EXPERIMENTS = [
    "01_hello_vllm_over_http",
    "02_same_code_any_provider",
    "03_streaming_ttft",
    "04_prefill_vs_decode",
    "05_cost_per_million_tokens",
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_version(base_url: str) -> str:
    try:
        root = base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + "/version", timeout=30) as r:
            return json.loads(r.read()).get("version", "?")
    except (OSError, ValueError):
        return "mock"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--verbose", action="store_true", help="print each experiment's raw result")
    args = p.parse_args()

    from vllm_lab import cli

    base_url = cli.resolve(args)
    res = {name: load(name).run(base_url) for name in EXPERIMENTS}
    r1, r2, r3, r4, r5 = (res[n] for n in EXPERIMENTS)
    if args.verbose:
        for name, value in res.items():
            cli.show(name, value)

    pf, dc = r4["prefill_sweep"], r4["decode_sweep"]
    one, eight = r5["rows"]
    rows = [
        ("01. Model over plain HTTP", f"{r1['served_model_name']} ({r1['usage']['total_tokens']} tok)",
         f"{r1['wall_s']:.2f} s", bool(r1["answer"]) and r1["usage"]["completion_tokens"] > 0),
        ("02. Same code, OpenAI SDK", "summary at temperature 0",
         "repeatable" if r2["repeatable"] else "differs", bool(r2["summary"]) and r2["repeatable"]),
        ("03. Streaming vs blocking", f"TTFT {r3['streaming_ttft_s']:.3f} s",
         f"{r3['decode_tokens_per_s']:.1f} tok/s",
         r3["streaming_ttft_s"] < r3["streaming_total_s"] and r3["first_word_sooner_x"] > 1),
        ("04. Prefill vs decode",
         (f"{pf[0]['prompt_tokens']}->{pf[-1]['prompt_tokens']} tok: "
          f"{pf[0]['ttft_s']:.3f}->{pf[-1]['ttft_s']:.3f} s"),
         f"read {r4['prefill_vs_decode_x']:.0f}x faster",
         pf[-1]["ttft_s"] > pf[0]["ttft_s"] and dc[-1]["total_s"] > dc[0]["total_s"]
         and r4["prefill_vs_decode_x"] > 1),
        ("05. Cost per 1M output tokens",
         f"1 user ${one['usd_per_million_output_tokens']:.2f}",
         f"8 users ${eight['usd_per_million_output_tokens']:.2f}",
         eight["output_tokens_per_s"] > one["output_tokens_per_s"]),
    ]

    mode = "mock (fake server, numbers are not real)" if args.mock else "live"
    version = server_version(base_url)
    lines = [
        "=" * 100,
        "THE OUTSIDE JUDGE — Your First vLLM Server with Gemma",
        "=" * 100,
        f"Mode    : {mode}",
        (f"Server  : vLLM {version} | model {r1['weights']} served as "
         f"'{r1['served_model_name']}' | max_model_len {r1['max_model_len']}"),
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 100,
        f"{'Experiment':32} | {'Measured':34} | {'Result':20} | Check",
        "-" * 100,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:32} | {measured:34} | {result:20} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 100,
        "Times include the network round trip through the tunnel and vary run to run.",
        (f"Prices: {r5['instance']}, ${r5['usd_per_hour']}/h list price checked "
         f"{r5['price_checked']}."),
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "vllm_version": version,
            "experiments": res,
            "checks": {name: ok for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
