"""Merge bench metrics + DCGM telemetry into one record; energy validity gate.

Pure-CPU given a DCGM CSV + bench dict, so the reduction/gate logic is
unit-testable without a GPU (Phase 2). The gate marks a tokens-per-joule run
`ok` or `contaminated`; contaminated runs are re-run, never averaged in.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Optional

from .telemetry import DCGM_FIELDS

# Validity-gate thresholds (see scratchpad.md for how the SM-active threshold was
# chosen empirically in Phase 2). Overridable via reduce_telemetry(..., gate=...).
# Contamination gate: what makes a tokens/joule run untrustworthy on the shared node.
# We flag ONLY on a directly-detectable hardware condition (thermal/HW clock throttle),
# NOT on an absolute SM-active bar. Rationale: LLM decode is memory-bandwidth-bound, so
# SM-active is legitimately 0.4-0.8 even when fully GPU-bound (fast fp8 sits lower still,
# and low-concurrency runs lower by design). An absolute SM-active threshold conflates
# "memory-bound" with "contaminated" and false-flags valid operating points. Residual
# host-contention noise is surfaced by the energy repeats (median + IQR), per the plan.
DEFAULT_GATE = {
    "energy_gpu_bound_min": 0.30, # INFORMATIONAL only: sets energy_gpu_bound flag, not status
    "max_neighbor_power_w": None, # optional hard cap on neighbor draw (off by default)
    "allow_throttle": False,      # any thermal/HW throttle flag => contaminated
}


def _reduce(values: list[float]) -> dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    s = sorted(vals)
    return {
        "mean": round(statistics.fmean(vals), 4),
        "p50": round(_pct(s, 50), 4),
        "p95": round(_pct(s, 95), 4),
        "max": round(max(vals), 4),
    }


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def parse_dcgm_csv(csv_path: str | Path) -> dict[str, list[float]]:
    """Parse a `dcgmi dmon` CSV/whitespace-table into {tidy_column: [samples]}.

    dcgmi dmon output is a whitespace-aligned table with a header line beginning
    with '#'. We map columns positionally onto DCGM_FIELDS order. Rows that don't
    parse to floats (e.g. 'N/A') become None for that cell.
    """
    cols = {name: [] for _, name in DCGM_FIELDS}
    p = Path(csv_path)
    if not p.exists():
        return cols

    names = [name for _, name in DCGM_FIELDS]
    for line in p.read_text().splitlines():
        line = line.strip()
        # Skip blanks and comment/header lines. Data rows carry an entity tag like
        # "GPU 0 <values...>"; the header/units lines dcgmi emits start with '#'.
        if not line or line.startswith("#"):
            continue
        # Extract the contiguous run of numeric tokens at the tail (drops the
        # leading "GPU"/entity-id tokens). Only accept rows with a full field set.
        nums = []
        for tok in line.split():
            try:
                nums.append(float(tok))
            except ValueError:
                nums = []  # reset — the numeric run must be contiguous at the tail
        tail = nums[-len(names):] if len(nums) >= len(names) else None
        if tail is None:
            continue  # header/units/malformed line -> not a data row
        for i, name in enumerate(names):
            cols[name].append(tail[i])
    return cols


def _steady_window(samples: list[float], skip_frac: float = 0.2) -> list[float]:
    """Drop the first `skip_frac` of samples (warm-up / prefill ramp) to isolate
    the steady decode window used by the validity gate."""
    vals = [v for v in samples if v is not None]
    if not vals:
        return []
    k = int(len(vals) * skip_frac)
    return vals[k:] or vals


def apply_validity_gate(reduced: dict, extras: dict, gate: dict) -> tuple[str, list[str]]:
    """Return (status, reasons). status is 'ok' or 'contaminated'.

    Contamination = a directly-detectable hardware condition only (thermal/HW throttle,
    or an optional neighbor-power cap). SM-active is NOT a contamination criterion (see
    DEFAULT_GATE); it feeds the informational energy_gpu_bound flag instead."""
    reasons: list[str] = []

    tr = extras.get("throttle_reasons")
    # Bit 0 (GpuIdle) is benign; any hardware/thermal throttle bit is not.
    THERMAL_MASK = 0xFFFFFFFF & ~0x1
    if not gate["allow_throttle"] and tr not in (None, 0) and (int(tr) & THERMAL_MASK):
        reasons.append(f"clock throttle flags set (0x{int(tr):x})")

    cap = gate.get("max_neighbor_power_w")
    npw = extras.get("neighbor_power_w")
    if cap is not None and npw is not None and npw > cap:
        reasons.append(f"neighbor power {npw}W > cap {cap}W")

    return ("contaminated" if reasons else "ok"), reasons


def reduce_telemetry(csv_path: str | Path, extras: Optional[dict] = None,
                     gate: Optional[dict] = None) -> dict[str, Any]:
    """Reduce a DCGM CSV to summary stats + steady-window signals + gate status."""
    extras = extras or {}
    gate = {**DEFAULT_GATE, **(gate or {})}
    cols = parse_dcgm_csv(csv_path)

    out: dict[str, Any] = {}
    for _, name in DCGM_FIELDS:
        r = _reduce(cols.get(name, []))
        out[f"{name}_mean"] = r["mean"]
        out[f"{name}_p50"] = r["p50"]
        out[f"{name}_p95"] = r["p95"]
        out[f"{name}_max"] = r["max"]

    # Steady-decode signals for the gate.
    sm_steady = _steady_window(cols.get("sm_active", []))
    out["sm_active_steady_mean"] = round(statistics.fmean(sm_steady), 4) if sm_steady else None
    power_steady = _steady_window(cols.get("power_w", []))
    out["power_w_steady_mean"] = round(statistics.fmean(power_steady), 2) if power_steady else None

    out.update({f"nvml_{k}": v for k, v in extras.items()})
    status, reasons = apply_validity_gate(out, extras, gate)
    out["status"] = status
    out["gate_reasons"] = "; ".join(reasons) if reasons else ""
    # Informational: was the GPU meaningfully busy in steady decode? Used to annotate
    # the operating point (low at low concurrency / fast fp8), NOT to gate validity.
    sm = out.get("sm_active_steady_mean")
    out["energy_gpu_bound"] = bool(sm is not None and sm >= gate["energy_gpu_bound_min"])
    return out


def tokens_per_joule(output_throughput_tok_s: Optional[float],
                     mean_power_w: Optional[float]) -> Optional[float]:
    """tokens/joule = (tokens/s) / (joules/s = watts). Self-normalizing per-GPU."""
    if not output_throughput_tok_s or not mean_power_w:
        return None
    return round(output_throughput_tok_s / mean_power_w, 4)


def build_record(cfg, bench: dict, telemetry: dict, *, pins: dict) -> dict[str, Any]:
    """Assemble the final tidy row: knobs + bench metrics + telemetry + energy + pins.

    `pins` carries reproducibility metadata (vllm/torch/cuda/driver/gpu_uuid/node).
    """
    power_for_energy = telemetry.get("power_w_steady_mean") or telemetry.get("power_w_mean")
    tpj = tokens_per_joule(bench.get("output_throughput_tok_s"), power_for_energy)

    record: dict[str, Any] = {
        # identity / provenance
        "cell_id": cfg.cell_id,
        "cell_name": cfg.cell_name,
        "source": cfg.source,
        "repeat_idx": cfg.repeat_idx,
        "workload": cfg.workload,
        # knobs
        "precision": cfg.precision,
        "speculative": cfg.speculative,
        "chunked_prefill": cfg.chunked_prefill,
        "max_num_seqs": cfg.max_num_seqs,
        "concurrency": cfg.concurrency,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "model_id": cfg.model_id,
        "spec_model": cfg.spec_model,
        # energy headline
        "tokens_per_joule": tpj,
    }
    record.update({k: v for k, v in bench.items() if not k.startswith("_")})
    record["bench_timed_out"] = bench.get("_timed_out", False)
    record["bench_elapsed_s"] = bench.get("_elapsed_s")
    record.update(telemetry)
    record.update(pins)
    # A timed-out bench has truncated/no data -> treat as a re-runnable failure, not an
    # energy "contamination" (which is reserved for hardware throttle on a valid run).
    if record.get("bench_timed_out"):
        record["status"] = "failed"
        record["gate_reasons"] = (record.get("gate_reasons", "") + "; bench wall-clock cap exceeded").strip("; ")
    # Defensive: a run where 0 requests completed is a FAILURE even though the server
    # stayed up and `vllm bench serve` returned a JSON (e.g. every request 500'd). Never
    # let a completed=0 / 0-throughput row masquerade as a valid 'ok' data point.
    completed = record.get("completed")
    if completed is not None and completed == 0:
        record["status"] = "failed"
        record["gate_reasons"] = (record.get("gate_reasons", "") + "; 0 requests completed (all errored)").strip("; ")
    return record
