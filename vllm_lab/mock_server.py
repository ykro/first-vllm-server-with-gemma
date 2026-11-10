"""A tiny fake of vLLM's OpenAI-compatible server, for running everything without a GPU.

It speaks the same HTTP as `vllm serve` (/health, /v1/models, /v1/chat/completions with
and without streaming) and simulates the two phases of generation: a prefill that costs a
little per prompt token and a decode that costs a lot more per generated token. Its numbers
are made up on purpose; only the live judge (`make verify-live`) measures a real GPU.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_FIXED_S = 0.010
PREFILL_PER_TOKEN_S = 0.00004
DECODE_PER_TOKEN_S = 0.012
BATCH_SLOWDOWN = 0.05  # each extra concurrent request slows a decode step by 5 %
NATURAL_LENGTH = 40

_SENTENCE = (
    "Un servidor de inferencia carga el modelo una sola vez en la GPU y atiende muchas "
    "peticiones por HTTP. Primero lee todo el prompt de golpe, y después escribe la "
    "respuesta token por token."
)
WORDS = _SENTENCE.split(" ")

_active = 0
_lock = threading.Lock()


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def fake_tokens(prompt: str, n: int) -> list[str]:
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(WORDS)
    return [WORDS[(start + i) % len(WORDS)] + " " for i in range(n)]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the terminal clean
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "gemma", "object": "model", "owned_by": "vllm", "max_model_len": 8192,
                 "root": "google/gemma-4-E4B-it"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        global _active
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = "".join(m["content"] for m in body["messages"])
        max_tokens = int(body.get("max_tokens") or 256)
        n = max_tokens if body.get("ignore_eos") else min(max_tokens, NATURAL_LENGTH)
        finish = "length" if n == max_tokens else "stop"
        ptoks = count_tokens(prompt)
        tokens = fake_tokens(prompt, n)
        usage = {"prompt_tokens": ptoks, "completion_tokens": n, "total_tokens": ptoks + n}
        with _lock:
            _active += 1
        try:
            time.sleep(PREFILL_FIXED_S + PREFILL_PER_TOKEN_S * ptoks)
            if not body.get("stream"):
                for _ in tokens[1:]:
                    time.sleep(self._step())
                self._json(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "model": "gemma",
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": "".join(tokens)}}],
                    "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i, tok in enumerate(tokens):
                if i:
                    time.sleep(self._step())
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": "gemma", "choices": [{"index": 0, "delta": {"content": tok},
                                                            "finish_reason": None}]})
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": "gemma",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                             "model": "gemma", "choices": [], "usage": usage})
            self._raw(b"data: [DONE]\n\n")
            self._raw(b"")
        finally:
            with _lock:
                _active -= 1

    @staticmethod
    def _step() -> float:
        return DECODE_PER_TOKEN_S * (1 + BATCH_SLOWDOWN * max(_active - 1, 0))

    def _chunk(self, obj: dict) -> None:
        self._raw(f"data: {json.dumps(obj)}\n\n".encode())

    def _raw(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # clients closing keep-alive sockets
        pass


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Starts the mock in a background thread; returns the server and its /v1 base URL."""
    server = _QuietServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(8000)
    print(f"Mock vLLM server at {url} (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
