"""Shared command line for the experiments: live server by default, --mock without a GPU."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from vllm_lab.client import BASE_URL


def parse(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mock", action="store_true", help="use the built-in fake server (no GPU)")
    p.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible /v1 URL")
    return p.parse_args()


def resolve(args: argparse.Namespace) -> str:
    if args.mock:
        from vllm_lab import mock_server

        _, url = mock_server.start()
        return url
    root = args.base_url.rsplit("/v1", 1)[0]
    try:
        urllib.request.urlopen(root + "/health", timeout=600).read()
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"No vLLM server at {root} ({exc}).\n"
            "Open the tunnel first:  gcloud run services proxy vllm-gemma "
            "--region us-east4 --port 8000\nor run with --mock."
        )
    return args.base_url


def show(title: str, payload: dict) -> None:
    print(f"\n== {title}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
