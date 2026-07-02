"""vLLM server lifecycle: launch `vllm serve`, poll /health, warm up, tear down.

Runs INSIDE a SLURM GPU allocation (Phase 1). Builds the `vllm serve` command
from a RunConfig, waits for readiness, and guarantees teardown even on failure.

The exact vLLM CLI flags are pinned in Phase 0 against the installed vLLM version
and recorded in scratchpad.md; the mapping below reflects vLLM V1 conventions and
should be re-verified there (search: "Phase 0 verify flags").
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


# Context cap (see build_serve_command). Must exceed max workload input+output.
MAX_MODEL_LEN = 8192


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_serve_command(cfg, *, port: int) -> list[str]:
    """Translate a RunConfig into a `vllm serve` argv.

    Phase 0 verify flags: confirm each of these against `vllm serve --help` on the
    installed version and note deviations in scratchpad.md.
    """
    if not cfg.model_id:
        raise ValueError(f"cfg has no resolved model_id (precision={cfg.precision})")

    cmd = ["vllm", "serve", cfg.model_id, "--port", str(port)]

    # Cap context to fit our workloads (max is long_prompt 4096-in + long_decode
    # 2048-out ~= 6k tokens). The model's native 131072 context needs ~16 GiB KV for a
    # single full-length request, which does NOT fit alongside weights on a 48 GB L40S
    # when chunked prefill is off or spec-decode reserves draft slots -> vLLM refuses to
    # start ("KV cache needed > available"). 8192 leaves generous headroom and, because
    # KV is block-allocated per actual sequence length, does not change short-seq results.
    cmd += ["--max-model-len", str(MAX_MODEL_LEN)]

    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", cfg.kv_cache_dtype]

    cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]

    # Chunked prefill: vLLM V1 enables it by default; disable explicitly when off.
    if cfg.chunked_prefill is False:
        cmd += ["--no-enable-chunked-prefill"]

    # Speculative decoding (V1 --speculative-config JSON).
    if cfg.speculative == "eagle3":
        spec = f'{{"method":"eagle3","model":"{cfg.spec_model}","num_speculative_tokens":3}}'
        cmd += ["--speculative-config", spec]
    elif cfg.speculative == "ngram":
        spec = '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}'
        cmd += ["--speculative-config", spec]

    return cmd


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int
    base_url: str
    log_path: Optional[str] = None
    cmd: list[str] = field(default_factory=list)


def _health_ok(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def launch(cfg, *, log_path: Optional[str] = None, ready_timeout_s: int = 1800) -> ServerHandle:
    """Start `vllm serve` and block until /health returns 200 or timeout.

    Raises TimeoutError (and tears the process down) if the server never becomes
    ready — a model that OOMs or an invalid combo surfaces here, not silently.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = build_serve_command(cfg, port=port)

    logf = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT,
        # New process group so we can kill the whole server tree on teardown.
        preexec_fn=os.setsid,
        env=os.environ.copy(),
    )
    handle = ServerHandle(proc=proc, port=port, base_url=base_url, log_path=log_path, cmd=cmd)

    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            teardown(handle)
            raise RuntimeError(f"vllm serve exited early (code {proc.returncode}); see {log_path}")
        if _health_ok(base_url):
            return handle
        time.sleep(2.0)

    teardown(handle)
    raise TimeoutError(f"vllm serve not ready within {ready_timeout_s}s; see {log_path}")


def warm_up(handle: ServerHandle, *, model_id: str, n: int = 2) -> None:
    """Fire a couple of tiny completions so the first benchmarked request isn't cold
    (CUDA graphs / caches primed). Failures here are non-fatal warnings."""
    body = {"model": model_id, "prompt": "Warm up.", "max_tokens": 8, "temperature": 0.0}
    data = json.dumps(body).encode()
    for _ in range(n):
        try:
            req = urllib.request.Request(
                f"{handle.base_url}/v1/completions", data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60):
                pass
        except Exception:
            break


def teardown(handle: ServerHandle) -> None:
    """Kill the server process group; never raises."""
    proc = handle.proc
    if proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except Exception:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


@contextlib.contextmanager
def serving(cfg, *, log_path: Optional[str] = None, ready_timeout_s: int = 1800):
    """Context manager: launch + warm up on enter, guaranteed teardown on exit."""
    handle = launch(cfg, log_path=log_path, ready_timeout_s=ready_timeout_s)
    try:
        warm_up(handle, model_id=cfg.model_id)
        yield handle
    finally:
        teardown(handle)
