"""Runs every experiment against the built-in mock server (no GPU needed)."""

import importlib.util
from pathlib import Path

import pytest

from vllm_lab import chat_once, make_client, mock_server, stream_chat

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def base_url():
    server, url = mock_server.start()
    yield url
    server.shutdown()


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chat_once_returns_usage(base_url):
    r = chat_once(make_client(base_url), "hola", max_tokens=5)
    assert r["text"] and r["completion_tokens"] == 5 and r["finish_reason"] == "length"


def test_stream_counts_every_token(base_url):
    t = stream_chat(make_client(base_url), "hola", max_tokens=12, ignore_eos=True)
    assert t.chunks == t.completion_tokens == 12
    assert 0 < t.ttft_s < t.total_s


def test_01_reads_model_and_answer(base_url):
    r = load("01_hello_vllm_over_http").run(base_url)
    assert r["served_model_name"] == "gemma" and r["usage"]["total_tokens"] > 0


def test_02_same_code_is_repeatable(base_url):
    assert load("02_same_code_any_provider").run(base_url)["repeatable"]


def test_03_streaming_shows_words_earlier(base_url):
    r = load("03_streaming_ttft").run(base_url)
    assert r["streaming_ttft_s"] < r["blocking_wait_s"]


def test_04_long_prompts_start_later(base_url):
    r = load("04_prefill_vs_decode").run(base_url)
    ttft = [row["ttft_s"] for row in r["prefill_sweep"]]
    assert ttft[-1] > ttft[0] and r["prefill_vs_decode_x"] > 1


def test_05_more_users_more_tokens(base_url):
    rows = load("05_cost_per_million_tokens").run(base_url)["rows"]
    assert rows[1]["output_tokens_per_s"] > rows[0]["output_tokens_per_s"]
