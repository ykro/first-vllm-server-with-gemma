"""Each snippet in practica/ runs against the fake server and prints what the video shows."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_lab import mock_server

ROOT = Path(__file__).resolve().parent.parent
PRACTICA = ROOT / "practica"


@pytest.fixture(scope="module")
def port():
    server, _ = mock_server.start()
    yield server.server_address[1]
    server.shutdown()


def run_against(port: int, name: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """A snippet exactly as written, pointed at the fake server's port instead of 8000."""
    src = (PRACTICA / name).read_text()
    assert "localhost:8000" in src
    copy = tmp_path / name
    copy.write_text(src.replace("localhost:8000", f"localhost:{port}"))
    cmd = ["sh", str(copy)] if name.endswith(".sh") else [sys.executable, str(copy)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_curl_gets_openai_shaped_json(port, tmp_path):
    out = run_against(port, "01_pregunta.sh", tmp_path).stdout
    assert '"object": "chat.completion"' in out
    assert '"finish_reason": "stop"' in out
    assert '"prompt_tokens": 12' in out and '"completion_tokens": 40' in out


def test_openai_package_needs_v1_in_base_url(port, tmp_path):
    r = run_against(port, "02_resumir_roto.py", tmp_path)
    assert r.returncode != 0
    assert "openai.NotFoundError: Error code: 404" in r.stderr


def test_openai_package_with_v1_answers(port, tmp_path):
    r = run_against(port, "02_resumir.py", tmp_path)
    assert r.returncode == 0, r.stderr
    assert "tokens de entrada: 25" in r.stdout
    assert "tokens de salida: 40" in r.stdout


def test_streaming_reports_ttft_before_the_end(port, tmp_path):
    r = run_against(port, "03_streaming.py", tmp_path)
    assert r.returncode == 0, r.stderr
    ttft = float(re.search(r"primer token a los ([\d.]+) s", r.stdout).group(1))
    total = float(re.search(r"respuesta completa a los ([\d.]+) s", r.stdout).group(1))
    assert "100 tokens," in r.stdout
    assert 0 < ttft < total


def test_cost_per_million_from_the_saved_run():
    r = subprocess.run([sys.executable, str(PRACTICA / "04_costo.py")], cwd=ROOT,
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.stdout.splitlines() == [
        "la L4 cuesta 1.4209 dólares la hora",
        "1 a la vez: 25.3 tokens/s, 91,080 por hora, $15.60 por millón",
        "8 a la vez: 192.0 tokens/s, 691,200 por hora, $2.06 por millón",
    ]
