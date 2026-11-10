"""Small helpers shared by the experiments: where the server lives, timing, the mock."""

from vllm_lab.client import (
    BASE_URL,
    MODEL,
    StreamTiming,
    chat_once,
    fresh,
    make_client,
    stream_chat,
)

__all__ = ["BASE_URL", "MODEL", "StreamTiming", "chat_once", "fresh", "make_client", "stream_chat"]
