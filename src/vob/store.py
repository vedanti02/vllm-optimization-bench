"""Resumable result store: append tidy rows to parquet, skip completed cells.

Pure-CPU, unit-testable (Phase 3). The store is keyed by ``cell_id`` (from
matrix.py). ``completed_cell_ids`` lets the runner skip cells already benchmarked
in a previous SLURM job, so a killed sweep restarts cleanly.

Raw per-run artifacts (the full `vllm bench serve` JSON + the DCGM CSV) are
written under ``<results_dir>/raw/<cell_id>/`` for provenance; the parquet holds
the reduced one-row-per-run record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

_PARQUET_NAME = "runs.parquet"


class ResultStore:
    def __init__(self, results_dir: str | Path):
        self.dir = Path(results_dir)
        self.raw_dir = self.dir / "raw"
        self.parquet_path = self.dir / _PARQUET_NAME

    def ensure_dirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if self.parquet_path.exists():
            return pd.read_parquet(self.parquet_path)
        return pd.DataFrame()

    def completed_cell_ids(self, *, include_contaminated: bool = False) -> set[str]:
        """cell_ids already recorded. By default a `contaminated` energy run is NOT
        considered complete (so it gets re-run); pass include_contaminated=True to
        treat any recorded row as done."""
        df = self.load()
        if df.empty or "cell_id" not in df.columns:
            return set()
        if not include_contaminated and "status" in df.columns:
            df = df[df["status"] != "contaminated"]
        return set(df["cell_id"].astype(str).tolist())

    def is_complete(self, cell_id: str, **kw) -> bool:
        return cell_id in self.completed_cell_ids(**kw)

    def append_row(self, row: dict[str, Any]) -> None:
        """Append one record. Upserts on (cell_id) — a re-run of a contaminated cell
        replaces the prior row for that cell_id rather than duplicating it, EXCEPT
        energy repeats which carry distinct repeat_idx -> distinct cell_id already."""
        self.ensure_dirs()
        df = self.load()
        cid = row.get("cell_id")
        if not df.empty and cid is not None and "cell_id" in df.columns:
            df = df[df["cell_id"] != cid]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(self.parquet_path, index=False)

    def write_raw(self, cell_id: str, name: str, content: str | bytes) -> Path:
        """Persist a raw artifact (bench JSON, DCGM CSV) under raw/<cell_id>/<name>."""
        self.ensure_dirs()
        d = self.raw_dir / cell_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(p, mode) as f:
            f.write(content)
        return p

    def pending(self, configs: Iterable, *, include_contaminated: bool = False) -> list:
        """Filter a list of RunConfig-like objects (with .cell_id) to those not yet done."""
        done = self.completed_cell_ids(include_contaminated=include_contaminated)
        return [c for c in configs if getattr(c, "cell_id", None) not in done]


def load_yaml(path: str | Path) -> dict:
    """Small helper so callers don't import yaml everywhere."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
