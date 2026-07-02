"""Resumable, parallel-safe result store.

Each cell's record is written as its own atomic JSON file under ``rows/<cell_id>.json``
(write-tmp + rename), so multiple SLURM array tasks can run ``vob run`` against the
same results dir without racing on a single file. The aggregate ``runs.parquet`` is a
*derived* export (``export_parquet`` / ``vob merge``) rebuilt from the per-cell rows.

Resumability: a cell is "complete" once it has a row (any status). Contaminated/failed
cells are NOT auto-re-run by default (that would loop forever on an inherently-flagged
cell); energy repeats get redundancy from distinct repeat_idx cells instead, and
analysis filters to ``status == 'ok'``. Pass ``include_contaminated=False`` to force
re-running contaminated cells.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_PARQUET_NAME = "runs.parquet"
_NONFINAL = {"contaminated", "failed"}


class ResultStore:
    def __init__(self, results_dir: str | Path):
        self.dir = Path(results_dir)
        self.rows_dir = self.dir / "rows"
        self.raw_dir = self.dir / "raw"
        self.parquet_path = self.dir / _PARQUET_NAME

    def ensure_dirs(self) -> None:
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # --- writing -----------------------------------------------------------
    def write_row(self, row: dict[str, Any]) -> Path:
        """Atomically write one cell's record to rows/<cell_id>.json.

        Overwrites any prior row for the same cell_id (a re-run replaces it). The
        tmp+rename keeps concurrent readers from seeing a half-written file."""
        self.ensure_dirs()
        cid = str(row.get("cell_id") or "unknown")
        p = self.rows_dir / f"{cid}.json"
        tmp = p.with_name(f"{cid}.json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(row, default=str, indent=0))
        tmp.replace(p)  # atomic on POSIX
        return p

    # append_row kept as an alias for older call sites / tests.
    append_row = write_row

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

    # --- reading -----------------------------------------------------------
    def _iter_rows(self) -> Iterable[dict]:
        if not self.rows_dir.exists():
            return
        for f in sorted(self.rows_dir.glob("*.json")):
            try:
                yield json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue  # skip a partial/corrupt file rather than crash the sweep

    def load(self) -> pd.DataFrame:
        """Merge all per-cell rows into a DataFrame (for analysis/export)."""
        recs = list(self._iter_rows())
        return pd.DataFrame(recs) if recs else pd.DataFrame()

    def completed_cell_ids(self, *, include_contaminated: bool = True) -> set[str]:
        """cell_ids that already have a row. By default contaminated/failed count as
        complete (no auto-re-run loop). Pass include_contaminated=False to re-run them."""
        ids: set[str] = set()
        for rec in self._iter_rows():
            cid = rec.get("cell_id")
            if cid is None:
                continue
            if not include_contaminated and rec.get("status") in _NONFINAL:
                continue
            ids.add(str(cid))
        return ids

    def is_complete(self, cell_id: str, **kw) -> bool:
        return cell_id in self.completed_cell_ids(**kw)

    def pending(self, configs: Iterable, *, include_contaminated: bool = True) -> list:
        """Filter RunConfig-like objects (with .cell_id) to those not yet complete."""
        done = self.completed_cell_ids(include_contaminated=include_contaminated)
        return [c for c in configs if getattr(c, "cell_id", None) not in done]

    # --- export ------------------------------------------------------------
    def export_parquet(self) -> pd.DataFrame:
        """Rebuild the aggregate runs.parquet from the per-cell rows. Idempotent."""
        df = self.load()
        if not df.empty:
            self.dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.parquet_path, index=False)
        return df


def load_yaml(path: str | Path) -> dict:
    """Small helper so callers don't import yaml everywhere."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
