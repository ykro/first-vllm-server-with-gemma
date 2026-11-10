"""OpenAI-compatible client pointed at your own vLLM server.

The only thing that changes from a hosted API is BASE_URL. With the Cloud Run tunnel
(`gcloud run services proxy vllm-gemma --region us-east4 --port 8000`) the server is
at http://localhost:8000/v1 and no API key is needed: the tunnel signs each request with
your gcloud identity.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field

from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL_NAME", "gemma")


def fresh(prompt: str) -> str:
    """Gives a prompt a unique first line.

    vLLM keeps the KV-cache of prompts it already read (prefix caching, unit C3). A second
    request that starts with the same text skips most of the prefill and looks faster than it
    is. Timing experiments start every prompt with a random id so each one is read from zero.
    """
    return f"[{uuid.uuid4().hex[:12]}]\n{prompt}"


def make_client(base_url: str | None = None) -> OpenAI:
    # vLLM ignores the key unless it was started with --api-key; the SDK just needs a string.
    return OpenAI(base_url=base_url or BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "local"))


def chat_once(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int | None = 7,
) -> dict:
    """One non-streaming request. Returns the text, token usage and wall time."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    wall = time.perf_counter() - t0
    return {
        "text": resp.choices[0].message.content or "",
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "finish_reason": resp.choices[0].finish_reason,
        "wall_s": wall,
    }


@dataclass
class StreamTiming:
    """What a user feels: when the first token shows up, and how fast the rest arrives."""

    ttft_s: float = 0.0
    total_s: float = 0.0
    chunks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    text: str = ""
    arrivals: list[float] = field(default_factory=list)

    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-9)

    @property
    def decode_tokens_per_s(self) -> float:
        # The first token closes the prefill; the remaining ones are pure decode.
        return max(self.completion_tokens - 1, 0) / self.decode_s

    @property
    def prefill_tokens_per_s(self) -> float:
        return self.prompt_tokens / max(self.ttft_s, 1e-9)


def stream_chat(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int | None = 7,
    ignore_eos: bool = False,
) -> StreamTiming:
    """Streams one answer and records when each chunk arrives."""
    extra = {"ignore_eos": True} if ignore_eos else None
    timing = StreamTiming()
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra,
    )
    parts: list[str] = []
    for chunk in stream:
        now = time.perf_counter() - t0
        if chunk.usage is not None:
            timing.prompt_tokens = chunk.usage.prompt_tokens
            timing.completion_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if timing.chunks == 0:
                timing.ttft_s = now
            timing.chunks += 1
            timing.arrivals.append(now)
            parts.append(chunk.choices[0].delta.content)
    timing.total_s = time.perf_counter() - t0
    timing.text = "".join(parts)
    return timing
