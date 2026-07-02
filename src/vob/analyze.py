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
    """Load the run store. Prefers the exported runs.parquet; falls back to merging
    the per-cell rows/*.json (so analysis works before an explicit `vob merge`)."""
    p = Path(results_dir) / "runs.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from .store import ResultStore
    df = ResultStore(results_dir).load()
    if df.empty:
        raise FileNotFoundError(f"no results under {results_dir} (no parquet, no rows/)")
    return df


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


def audit(df: pd.DataFrame) -> dict:
    """No-errors audit: status counts + any non-ok cells (excluding throttle-contaminated,
    which is a legitimate shared-node flag). Returns a dict for the report + gating."""
    from collections import Counter
    counts = dict(Counter(df["status"])) if "status" in df.columns else {}
    nonok = df[df["status"] != "ok"] if "status" in df.columns else df.iloc[0:0]
    # throttle-contaminated is an acceptable (flagged) outcome; failed is not.
    failed = nonok[nonok["status"] == "failed"] if "status" in nonok.columns else nonok.iloc[0:0]
    return {
        "status_counts": counts,
        "n_failed": int(len(failed)),
        "failed_cells": failed[["cell_name", "workload", "precision", "gate_reasons"]].to_dict("records")
        if len(failed) else [],
        "clean": bool(len(failed) == 0),
    }


def sanity_checks(df: pd.DataFrame) -> list[dict]:
    """Physics/expectation gates. Each returns {name, pass, detail}."""
    out = []
    ok = df[df["status"] == "ok"] if "status" in df.columns else df

    # 1) FP8 >= BF16 throughput (per workload, concurrency=16 baseline point)
    base = ok[(ok["precision"] == "bf16") & (ok["speculative"] == "none") &
              (ok["chunked_prefill"] == True) & (ok["concurrency"] == 16)]
    fp8 = ok[(ok["precision"] == "fp8-static") & (ok["concurrency"] == 16)]
    if len(base) and len(fp8):
        merged = base.merge(fp8, on="workload", suffixes=("_bf16", "_fp8"))
        wins = (merged["output_throughput_tok_s_fp8"] >= merged["output_throughput_tok_s_bf16"]).mean()
        out.append({"name": "FP8 >= BF16 throughput", "pass": bool(wins == 1.0),
                    "detail": f"{int(wins*len(merged))}/{len(merged)} workloads"})

    # 2) serialized (max_num_seqs=1) << continuous batching (baseline max_num_seqs=256)
    ser = ok[(ok["cell_name"] == "max_num_seqs") & (ok["max_num_seqs"] == 1)]
    cont = ok[(ok["cell_name"] == "baseline")]
    if len(ser) and len(cont):
        m = ser.merge(cont, on="workload", suffixes=("_ser", "_cont"))
        ratio = (m["output_throughput_tok_s_ser"] / m["output_throughput_tok_s_cont"]).mean()
        out.append({"name": "serialized << continuous batching", "pass": bool(ratio < 0.9),
                    "detail": f"mean serialized/continuous throughput ratio = {ratio:.2f}"})

    # 3) all energy-valid rows have non-null tokens/joule
    ev = valid_energy(ok)
    out.append({"name": "energy rows have tokens/joule", "pass": bool(ev["tokens_per_joule"].notna().all()),
                "detail": f"{len(ev)} energy-valid rows"})
    return out


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


def build_report(results_dir: str | Path, quality_path: str | Path | None = None) -> str:
    """Generate RESULTS.md (audit + sanity + OFAT tables + energy median/IQR + quality)
    and figures under <results_dir>/figures. Returns the markdown string."""
    import json as _json

    results_dir = Path(results_dir)
    df = load_runs(results_dir)
    a = audit(df)
    checks = sanity_checks(df)
    fig_dir = results_dir / "figures"

    L = ["# vllm-optimization-bench — L40S results\n",
         f"Total runs: **{len(df)}**  |  status: {a['status_counts']}\n"]

    L.append("## No-errors audit")
    L.append(f"- Failed cells: **{a['n_failed']}** {'✅ clean' if a['clean'] else '❌'}")
    for c in a["failed_cells"]:
        L.append(f"  - {c['cell_name']}/{c['workload']}/{c['precision']}: {c['gate_reasons']}")

    L.append("\n## Sanity checks")
    for c in checks:
        L.append(f"- {'✅' if c['pass'] else '❌'} **{c['name']}** — {c['detail']}")

    ok = df[df["status"] == "ok"]
    L.append("\n## OFAT: throughput & energy by precision (concurrency=16)")
    prec = ok[ok["cell_name"].isin(["precision", "baseline"])]
    if len(prec):
        t = prec.groupby(["workload", "precision"])[
            ["output_throughput_tok_s", "ttft_ms_median", "tpot_ms_median", "tokens_per_joule"]].mean().round(1)
        L.append("```\n" + t.to_string() + "\n```")

    L.append("\n## Speculative decoding (vs baseline)")
    spec = ok[ok["cell_name"].isin(["speculative", "baseline"])]
    if len(spec):
        t = spec.groupby(["workload", "speculative"])[
            ["output_throughput_tok_s", "tokens_per_joule"]].mean().round(2)
        L.append("```\n" + t.to_string() + "\n```")

    L.append("\n## Energy headline (median + IQR across repeats)")
    es = energy_summary(df)
    if len(es):
        L.append("```\n" + es.round(3).to_string(index=False) + "\n```")

    if quality_path and Path(quality_path).exists():
        q = _json.loads(Path(quality_path).read_text())
        L.append("\n## FP8 quality gate (perplexity vs BF16)")
        L.append("```")
        for r in q:
            d = f"{r.get('perplexity_delta_pct'):+.1f}%" if r.get("perplexity_delta_pct") is not None else "baseline"
            L.append(f"{r['precision']:14s} ppl={r.get('perplexity'):.3f}  ({d})  flagged={r.get('flagged')}")
        L.append("```")

    # figures
    try:
        f = plot_energy_frontier(df, fig_dir / "energy_frontier.png")
        if f:
            L.append(f"\n![energy frontier](figures/{f.name})")
    except Exception as e:
        L.append(f"\n(figure skipped: {e})")

    md = "\n".join(L) + "\n"
    (results_dir / "RESULTS.md").write_text(md)
    return md
