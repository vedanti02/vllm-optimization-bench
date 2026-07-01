"""Analysis (Phase 7): aggregate the run store into comparison tables + figures.

Pure-CPU. Reads results/L40S/runs.parquet, computes median + IQR for energy
repeats, and emits comparison tables/plots. Energy aggregation counts only
`status == 'ok'` rows (contaminated/failed excluded), matching the validity gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_runs(results_dir: str | Path) -> pd.DataFrame:
    p = Path(results_dir) / "runs.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no results at {p}")
    return pd.read_parquet(p)


def valid_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for tokens/joule claims: gate-passed and non-null energy."""
    d = df
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    return d[d["tokens_per_joule"].notna()]


def energy_summary(df: pd.DataFrame, group_cols=("cell_name", "precision", "speculative", "workload")) -> pd.DataFrame:
    """Median + IQR of tokens/joule across repeats (residual contamination -> variance)."""
    d = valid_energy(df)
    group_cols = [c for c in group_cols if c in d.columns]
    g = d.groupby(list(group_cols))["tokens_per_joule"]
    out = g.agg(
        n="count",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    out["iqr"] = out["q75"] - out["q25"]
    return out.sort_values("median", ascending=False)


def ofat_table(df: pd.DataFrame, axis: str, metric: str = "output_throughput_tok_s") -> pd.DataFrame:
    """One OFAT axis vs a metric, across workloads."""
    d = df[df["cell_name"].isin([axis, "baseline"])] if "cell_name" in df.columns else df
    cols = [c for c in (axis if axis in d.columns else "precision", "workload", metric) if c in d.columns]
    return d[cols].sort_values(list(cols[:-1]))


def plot_energy_frontier(df: pd.DataFrame, out_path: str | Path,
                         x: str = "output_throughput_tok_s", y: str = "tokens_per_joule") -> Optional[Path]:
    """Scatter the throughput vs tokens/joule efficiency frontier (blog figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = valid_energy(df)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for prec, sub in d.groupby("precision"):
        ax.scatter(sub[x], sub[y], label=str(prec), alpha=0.7)
    ax.set_xlabel("output throughput (tok/s)")
    ax.set_ylabel("tokens / joule")
    ax.set_title("L40S efficiency frontier")
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)
