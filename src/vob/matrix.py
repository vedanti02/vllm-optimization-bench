"""Matrix engine: expand OFAT axes + interaction cells into concrete RunConfigs.

Pure-CPU, unit-testable (Phase 3). No GPU, no vLLM import.

The public entry point is :func:`expand_matrix`, which reads the parsed
matrix.yaml + workloads.yaml dicts and returns a de-duplicated, invalid-pruned
list of :class:`RunConfig`. Each config carries a stable ``cell_id`` used by
``store.py`` for resumability (skip completed cells across separate SLURM jobs).
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Optional

from pydantic import BaseModel, Field

# Knobs that define a unique experimental cell (repeat_idx + workload added on top).
# Order matters: it fixes the canonical key ordering for cell_id hashing.
_KNOBS = ("precision", "speculative", "chunked_prefill", "max_num_seqs", "concurrency", "kv_cache_dtype")


class RunConfig(BaseModel):
    """One concrete benchmark cell: a knob setting × workload × repeat."""

    # Experimental knobs
    precision: str = "bf16"
    speculative: str = "none"
    chunked_prefill: bool = True
    max_num_seqs: int = 256
    concurrency: int = 16
    kv_cache_dtype: str = "auto"

    # Workload + repeat
    workload: str = "chat"
    repeat_idx: int = 0

    # Resolved model metadata (filled by resolve_models)
    model_id: Optional[str] = None
    quantization: Optional[str] = None
    spec_model: Optional[str] = None

    # Provenance
    source: str = "ofat"            # baseline | ofat | interaction
    cell_name: str = "baseline"     # axis name or interaction name

    cell_id: str = Field(default="", description="stable hash for resumability")

    def knob_key(self) -> tuple:
        return tuple(getattr(self, k) for k in _KNOBS) + (self.workload, self.repeat_idx)

    def compute_cell_id(self) -> str:
        payload = "|".join(f"{k}={getattr(self, k)}" for k in _KNOBS)
        # cell_name is part of the identity: an energy repeat is a distinct
        # experimental unit from a baseline/OFAT cell even when the knobs coincide,
        # so it must not dedup against one (otherwise energy cells silently lose a
        # repeat to a differently-labeled cell and median/IQR aggregates over N-1).
        payload += f"|cell={self.cell_name}|workload={self.workload}|repeat={self.repeat_idx}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def finalize(self) -> "RunConfig":
        self.cell_id = self.compute_cell_id()
        return self


def _is_invalid(cfg_knobs: dict[str, Any], invalid_combos: list[dict]) -> Optional[str]:
    """Return the prune reason if cfg matches an invalid combo, else None.

    A combo is invalid iff every key/value in an entry's ``when`` block matches
    the config. Empty ``when`` blocks never match (avoids pruning everything)."""
    for entry in invalid_combos:
        when = entry.get("when", {})
        if when and all(cfg_knobs.get(k) == v for k, v in when.items()):
            return entry.get("reason", "invalid combo")
    return None


def normalize_dependencies(knobs: dict[str, Any]) -> list[str]:
    """Apply structural knob dependencies in place; return notes describing edits.

    EAGLE-3 speculative decoding is incompatible with chunked prefill on vLLM V1,
    so chunked prefill is a *dependent* knob here, not a free axis: selecting
    EAGLE-3 forces it off. We normalize rather than prune, otherwise the entire
    EAGLE-3 story (whose baseline has chunked prefill on) would be dropped.

    NOTE: re-verify this constraint against the installed vLLM version in Phase 5;
    newer vLLM builds may support EAGLE-3 + chunked prefill, in which case drop
    this normalization so the OFAT comparison stops confounding the two knobs.
    """
    notes: list[str] = []
    if knobs.get("speculative") == "eagle3" and knobs.get("chunked_prefill") is True:
        knobs["chunked_prefill"] = False
        notes.append("eagle3 forces chunked_prefill=False (vLLM V1 dependency)")
    return notes


def resolve_models(cfg: RunConfig, matrix: dict) -> RunConfig:
    """Fill model_id / quantization / spec_model from the matrix `models` block.

    Handles the special ``fp8-kv`` precision: it reuses the fp8-static weights
    but forces ``kv_cache_dtype=fp8``.
    """
    models = matrix.get("models", {})
    spec_models = matrix.get("speculative_models", {})

    prec = cfg.precision
    if prec == "fp8-kv":
        entry = models.get("fp8-static", {})
        cfg.kv_cache_dtype = "fp8"
    else:
        entry = models.get(prec, {})

    cfg.model_id = entry.get("id")
    cfg.quantization = entry.get("quantization")

    if cfg.speculative and cfg.speculative != "none":
        cfg.spec_model = spec_models.get(cfg.speculative)  # may be None (ngram)
    return cfg


def _mk(base: dict, overrides: dict, *, source: str, cell_name: str,
        workload: str, repeat_idx: int, matrix: dict) -> RunConfig:
    knobs = {**base, **overrides}
    normalize_dependencies(knobs)   # apply structural dependencies before hashing cell_id
    cfg = RunConfig(
        source=source, cell_name=cell_name, workload=workload, repeat_idx=repeat_idx,
        **{k: knobs[k] for k in _KNOBS if k in knobs},
    )
    cfg = resolve_models(cfg, matrix)
    return cfg.finalize()


def expand_matrix(matrix: dict, workloads: dict) -> tuple[list[RunConfig], list[dict]]:
    """Expand the matrix into concrete RunConfigs.

    Returns ``(configs, pruned)`` where ``pruned`` is a list of
    ``{"knobs": ..., "reason": ...}`` for every dropped invalid combo (for logging).
    Duplicate cells (same cell_id) are collapsed to one.
    """
    baseline = dict(matrix["baseline"])
    invalid = matrix.get("invalid_combos", [])
    ofat_workloads = matrix.get("ofat_workloads", list(workloads.get("workloads", {}).keys()))

    seen: dict[str, RunConfig] = {}
    pruned: list[dict] = []

    def add(cfg: RunConfig) -> None:
        knobs = {k: getattr(cfg, k) for k in _KNOBS}
        # invalid_combos may also key on `workload` (e.g. eagle3 x long_prompt), so
        # include it in the match dict.
        reason = _is_invalid({**knobs, "workload": cfg.workload}, invalid)
        if reason:
            pruned.append({"knobs": knobs, "workload": cfg.workload, "reason": reason})
            return
        if cfg.cell_id not in seen:
            seen[cfg.cell_id] = cfg

    # 1) Baseline, once per ofat workload.
    for wl in ofat_workloads:
        add(_mk(baseline, {}, source="baseline", cell_name="baseline",
                workload=wl, repeat_idx=0, matrix=matrix))

    # 2) OFAT axes: vary one knob, skip the level that equals baseline (already emitted).
    for axis, spec in matrix.get("ofat", {}).items():
        levels = spec["levels"]
        axis_workloads = spec.get("workloads", ofat_workloads)
        for level in levels:
            # `fp8-kv` is a precision *level* even though baseline precision is bf16.
            if axis == "precision" and level == "fp8-kv":
                override = {"precision": "fp8-kv"}
            else:
                if level == baseline.get(axis):
                    continue
                override = {axis: level}
            for wl in axis_workloads:
                add(_mk(baseline, override, source="ofat", cell_name=axis,
                        workload=wl, repeat_idx=0, matrix=matrix))

    # 3) Interaction cells: cartesian product of `fixed` × `sweep`, with repeats.
    for cell in matrix.get("interactions", []):
        name = cell["name"]
        fixed = cell.get("fixed", {})
        sweep = cell.get("sweep", {})
        cell_workloads = cell.get("workloads", ofat_workloads)
        repeats = int(cell.get("repeats", 1))

        sweep_keys = list(sweep.keys())
        sweep_values = [sweep[k] for k in sweep_keys]
        combos = list(itertools.product(*sweep_values)) if sweep_values else [()]

        for combo in combos:
            override = {**fixed, **dict(zip(sweep_keys, combo))}
            for wl in cell_workloads:
                for r in range(repeats):
                    add(_mk(baseline, override, source="interaction", cell_name=name,
                            workload=wl, repeat_idx=r, matrix=matrix))

    configs = list(seen.values())
    return configs, pruned


def attach_workload_shape(cfg: RunConfig, workloads: dict) -> dict:
    """Return the workload shape dict (input_len/output_len/num_prompts/...) for a config,
    merged over the workloads `defaults` block. Used by the runner to build the bench command."""
    wl = workloads["workloads"][cfg.workload]
    merged = {**workloads.get("defaults", {}), **wl}
    return merged
