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
DEFAULT_GATE = {
    "sm_active_min": 0.80,        # steady decode should keep SMs busy
    "max_neighbor_power_w": None, # optional hard cap on neighbor draw
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
    """Return (status, reasons). status is 'ok' or 'contaminated'."""
    reasons: list[str] = []

    sm = reduced.get("sm_active_steady_mean")
    if sm is not None and sm < gate["sm_active_min"]:
        reasons.append(f"sm_active {sm:.2f} < {gate['sm_active_min']} (GPU idle-waiting / host-bound)")

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
    # If the benchmark timed out (budget guard) it can't be a valid energy point.
    if record.get("bench_timed_out"):
        record["status"] = "contaminated"
        record["gate_reasons"] = (record.get("gate_reasons", "") + "; bench wall-clock cap exceeded").strip("; ")
    return record
