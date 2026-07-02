"""Benchmark runner: invoke `vllm bench serve`, capture + parse its JSON result.

Runs INSIDE a SLURM GPU allocation (Phase 1). Given a ready ServerHandle and a
RunConfig (+ its workload shape), it runs the benchmark against the local server
and returns a tidy metrics dict (TTFT / TPOT / throughput / completed requests).

`vllm bench serve --save-result` writes a JSON file; we parse that rather than
scraping stdout so the numbers match a hand-run benchmark exactly (Phase 1 exit).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


# Per-concurrent-stream prompt budget. Total prompts scale with concurrency (floored)
# so serialized/low-concurrency cells finish in bounded time. Throughput/latency/energy
# are RATE metrics, so a smaller count at low concurrency doesn't bias them — it only
# bounds duration and slightly widens stats noise (fine for these operating points).
PROMPTS_PER_STREAM = 16
MIN_PROMPTS = 16


def effective_num_prompts(configured: int, concurrency: int, max_num_seqs: int = 10**9) -> int:
    # Effective parallelism is bounded by BOTH the client concurrency and the server's
    # max_num_seqs (max_num_seqs=1 serializes even with many clients), so scale by the min.
    parallelism = min(concurrency, max_num_seqs)
    return int(min(configured, max(MIN_PROMPTS, parallelism * PROMPTS_PER_STREAM)))


def build_bench_command(cfg, shape: dict, *, base_url: str, result_path: str) -> list[str]:
    """Build the `vllm bench serve` argv from a RunConfig + workload shape.

    Phase 0 verify flags: confirm names against `vllm bench serve --help`.
    """
    host_port = base_url.replace("http://", "").split(":")
    host, port = host_port[0], host_port[1]

    n_prompts = effective_num_prompts(shape["num_prompts"], cfg.concurrency, cfg.max_num_seqs)
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", cfg.model_id,
        "--host", host,
        "--port", str(port),
        "--dataset-name", shape.get("dataset", "random"),
        "--random-input-len", str(shape["input_len"]),
        "--random-output-len", str(shape["output_len"]),
        "--num-prompts", str(n_prompts),
        "--max-concurrency", str(cfg.concurrency),
        "--seed", str(shape.get("seed", 1234)),
        "--save-result",
        "--result-filename", result_path,
    ]
    if shape.get("ignore_eos", False):
        cmd += ["--ignore-eos"]
    return cmd


# Map vLLM's result JSON keys -> our tidy row. vLLM has used both snake_case and
# a couple of variants across versions; we look up several candidates per metric.
_METRIC_ALIASES = {
    "ttft_ms_mean": ["mean_ttft_ms"],
    "ttft_ms_median": ["median_ttft_ms"],
    "ttft_ms_p99": ["p99_ttft_ms"],
    "tpot_ms_mean": ["mean_tpot_ms"],
    "tpot_ms_median": ["median_tpot_ms"],
    "tpot_ms_p99": ["p99_tpot_ms"],
    "itl_ms_mean": ["mean_itl_ms"],
    "output_throughput_tok_s": ["output_throughput"],
    "total_token_throughput_tok_s": ["total_token_throughput"],
    "request_throughput_req_s": ["request_throughput"],
    "completed": ["completed", "num_prompts"],
    "total_output_tokens": ["total_output_tokens"],
    "duration_s": ["duration"],
}


def parse_bench_json(raw: dict) -> dict[str, Any]:
    """Reduce the raw vLLM bench JSON to our tidy metric row (aliases tolerate
    key-name drift across vLLM versions)."""
    out: dict[str, Any] = {}
    for tidy_key, candidates in _METRIC_ALIASES.items():
        for c in candidates:
            if c in raw and raw[c] is not None:
                out[tidy_key] = raw[c]
                break
        else:
            out[tidy_key] = None
    return out


def run_benchmark(cfg, shape: dict, *, base_url: str, result_path: str,
                  wallclock_cap_s: Optional[int] = None) -> dict[str, Any]:
    """Run `vllm bench serve` and return {metrics..., _bench_raw, _elapsed_s, _timed_out}.

    Enforces the per-run wall-clock cap (budget guard): a run that overshoots is
    killed and flagged rather than allowed to spiral.
    """
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = build_bench_command(cfg, shape, base_url=base_url, result_path=result_path)

    t0 = time.monotonic()
    timed_out = False
    try:
        subprocess.run(cmd, check=True, timeout=wallclock_cap_s,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.monotonic() - t0

    metrics: dict[str, Any] = {}
    raw: dict[str, Any] = {}
    if not timed_out and Path(result_path).exists():
        with open(result_path) as f:
            raw = json.load(f)
        metrics = parse_bench_json(raw)

    metrics["_bench_raw"] = raw
    metrics["_elapsed_s"] = round(elapsed, 2)
    metrics["_timed_out"] = timed_out
    return metrics
