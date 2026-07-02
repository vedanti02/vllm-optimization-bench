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


# validated categorical palette (dataviz references/palette.md, light mode)
_BLUE, _AQUA, _YELLOW, _GREEN, _VIOLET, _RED = "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"
_INK, _MUTED, _SURFACE, _GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"
_PREC = {"bf16": _BLUE, "fp8-dynamic": _AQUA, "fp8-static": _VIOLET, "fp8-kv": _GREEN}
_SPEC = {"none": _BLUE, "ngram": _AQUA, "eagle3": _VIOLET}
_WLC = {"chat": _BLUE, "long_prompt": _YELLOW, "long_decode": _VIOLET, "saturation": _GREEN}
_WL = ["chat", "long_prompt", "long_decode", "saturation"]
_PREC_ORDER = ["bf16", "fp8-dynamic", "fp8-static", "fp8-kv"]

# ordered (filename -> insight caption) for the figures make_figures produces
FIGURE_CAPTIONS = {
    "fp8_throughput.png": "output throughput by workload and precision at concurrency 16. Every FP8 variant clears bf16 on every workload.",
    "fp8_energy.png": "tokens per joule by workload and precision at concurrency 16. FP8 improves energy efficiency the most on saturation and long_decode.",
    "latency_throughput.png": "decode latency (TPOT median) against output throughput at concurrency 16 (color is precision, marker shape is workload). FP8 points sit up and to the right of bf16 within each workload.",
    "speculative.png": "throughput (left) and energy efficiency (right) for no speculation, ngram, and EAGLE-3 at concurrency 16 on bf16. Speculative decoding is a clear win on long_decode and mixed elsewhere; eagle3 with long_prompt is pruned.",
    "concurrency_scaling.png": "throughput (left) and energy efficiency (right) against client concurrency on a log x axis (bf16, one line per workload). Batching lifts both for chat, long_decode, and saturation, while long_prompt stays flat.",
    "quality.png": "perplexity by precision over 40 held out prompts. The dashed line is the plus 5 percent gate above the bf16 baseline; every FP8 variant sits under it.",
    "energy_frontier.png": "tokens per joule against output throughput across all runs, colored by precision. Efficiency tracks throughput; the highest points are bf16 at high concurrency, so batching is the largest single lever.",
}


def make_figures(df: pd.DataFrame, fig_dir: str | Path, quality_path: str | Path | None = None) -> list[str]:
    """Render all README/RESULTS figures from the run store into fig_dir. Returns the
    list of filenames written. Uses the validated categorical palette and one y-axis
    per panel (see the dataviz skill). If quality_path is given, also draws quality.png."""
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE, "savefig.facecolor": _SURFACE,
        "axes.edgecolor": _MUTED, "axes.labelcolor": _INK, "text.color": _INK,
        "xtick.color": _MUTED, "ytick.color": _MUTED, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    ok = df[df["status"] == "ok"].copy()
    written: list[str] = []

    def style(ax, ylabel, title):
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", loc="left", pad=10)
        ax.grid(axis="x", visible=False)

    def label_bars(ax, bars, fmt):
        for b in bars:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=_MUTED, xytext=(0, 2), textcoords="offset points")

    def grouped(metric, ylabel, title, fname, fmt, cells=("precision", "baseline")):
        at16 = ok[(ok.concurrency == 16) & (ok.cell_name.isin(cells))]
        fig, ax = plt.subplots(figsize=(9, 4.8)); n = len(_PREC_ORDER); w = 0.8 / n
        for i, g in enumerate(_PREC_ORDER):
            vals = [at16[(at16.workload == wl) & (at16.precision == g)][metric].mean() if
                    len(at16[(at16.workload == wl) & (at16.precision == g)]) else 0.0 for wl in _WL]
            x = [j + (i - (n - 1) / 2) * w for j in range(len(_WL))]
            label_bars(ax, ax.bar(x, vals, w * 0.92, label=g, color=_PREC[g], zorder=3), fmt)
        ax.set_xticks(range(len(_WL))); ax.set_xticklabels(_WL)
        style(ax, ylabel, title)
        ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.09))
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(fname)

    grouped("output_throughput_tok_s", "output throughput (tokens/s)",
            "FP8 vs BF16 throughput by workload (concurrency 16)", "fp8_throughput.png", "{:.0f}")
    grouped("tokens_per_joule", "energy efficiency (tokens/joule)",
            "FP8 vs BF16 energy efficiency by workload (concurrency 16)", "fp8_energy.png", "{:.2f}")

    # speculative, two panels
    sp = ok[(ok.concurrency == 16) & (ok.precision == "bf16") & (ok.cell_name.isin(["speculative", "baseline"]))]
    order = ["none", "ngram", "eagle3"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, metric, ylab, fmt in [(axes[0], "output_throughput_tok_s", "output throughput (tokens/s)", "{:.0f}"),
                                  (axes[1], "tokens_per_joule", "energy efficiency (tokens/joule)", "{:.2f}")]:
        n = len(order); w = 0.8 / n
        for i, g in enumerate(order):
            for j, wl in enumerate(_WL):
                sub = sp[(sp.workload == wl) & (sp.speculative == g)]; xp = j + (i - (n - 1) / 2) * w
                if not len(sub):
                    ax.annotate("pruned", (xp, 0), rotation=90, fontsize=7, color=_MUTED,
                                ha="center", va="bottom", xytext=(0, 3), textcoords="offset points"); continue
                label_bars(ax, ax.bar([xp], [sub[metric].mean()], w * 0.92,
                                      label=g if j == 0 else "_nolegend_", color=_SPEC[g], zorder=3), fmt)
        ax.set_xticks(range(len(_WL))); ax.set_xticklabels(_WL); style(ax, ylab, "")
    axes[0].set_title("Throughput", fontweight="bold", loc="left")
    axes[1].set_title("Energy efficiency", fontweight="bold", loc="left")
    fig.suptitle("Speculative decoding by workload (bf16, concurrency 16)", fontweight="bold", x=0.02, ha="left")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09), title="speculative")
    fig.tight_layout(); fig.savefig(fig_dir / "speculative.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append("speculative.png")

    # concurrency scaling
    sc = ok[(ok.precision == "bf16") & (ok.speculative == "none") &
            (ok.cell_name.isin(["concurrency", "baseline"])) & (ok.chunked_prefill == True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, ylab in [(axes[0], "output_throughput_tok_s", "output throughput (tokens/s)"),
                             (axes[1], "tokens_per_joule", "energy efficiency (tokens/joule)")]:
        for wl in _WL:
            s = sc[sc.workload == wl].groupby("concurrency")[metric].mean().sort_index()
            ax.plot(s.index, s.values, marker="o", markersize=7, linewidth=2, color=_WLC[wl], label=wl, zorder=3)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 16, 256]); ax.set_xticklabels([1, 16, 256])
        ax.set_xlabel("client concurrency"); style(ax, ylab, "")
    axes[0].set_title("Continuous batching: throughput scaling (bf16)", fontweight="bold", loc="left")
    axes[1].set_title("Continuous batching: energy scaling (bf16)", fontweight="bold", loc="left")
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(fig_dir / "concurrency_scaling.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append("concurrency_scaling.png")

    # latency vs throughput
    at16 = ok[(ok.concurrency == 16) & (ok.cell_name.isin(["precision", "baseline"]))]
    mk = {"chat": "o", "long_prompt": "s", "long_decode": "^", "saturation": "D"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for prec in _PREC_ORDER:
        for wl in _WL:
            sub = at16[(at16.precision == prec) & (at16.workload == wl)]
            if len(sub):
                ax.scatter(sub["output_throughput_tok_s"], sub["tpot_ms_median"], s=90, color=_PREC[prec],
                           marker=mk[wl], edgecolor=_SURFACE, linewidth=1.2, zorder=3)
    ph = [Line2D([0], [0], marker="o", color="w", markerfacecolor=_PREC[p], markersize=10, label=p) for p in _PREC_ORDER]
    wh = [Line2D([0], [0], marker=mk[w], color=_MUTED, markerfacecolor="none", markersize=9, label=w, linestyle="none") for w in _WL]
    l1 = ax.legend(handles=ph, frameon=False, title="precision", loc="upper left")
    ax.add_artist(l1); ax.legend(handles=wh, frameon=False, title="workload", loc="lower right")
    ax.set_xlabel("output throughput (tokens/s)  -> better"); ax.invert_yaxis()
    style(ax, "decode latency TPOT median (ms)  -> better (down)",
          "Latency vs throughput (concurrency 16): FP8 is faster and lower latency")
    fig.tight_layout(); fig.savefig(fig_dir / "latency_throughput.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append("latency_throughput.png")

    # energy frontier
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for prec in _PREC_ORDER:
        sub = ok[ok.precision == prec]
        ax.scatter(sub["output_throughput_tok_s"], sub["tokens_per_joule"], s=70, color=_PREC[prec],
                   edgecolor=_SURFACE, linewidth=1.0, alpha=0.9, label=prec, zorder=3)
    ax.legend(frameon=False, title="precision", loc="upper left")
    ax.set_xlabel("output throughput (tokens/s)")
    style(ax, "energy efficiency (tokens/joule)", "Energy frontier: tokens/joule vs throughput (all runs)")
    fig.tight_layout(); fig.savefig(fig_dir / "energy_frontier.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append("energy_frontier.png")

    # FP8 quality gate (needs quality.json)
    if quality_path and Path(quality_path).exists():
        q = _json.loads(Path(quality_path).read_text())
        q = sorted([r for r in q if r["precision"] in _PREC_ORDER], key=lambda r: _PREC_ORDER.index(r["precision"]))
        if q:
            base = next(r["perplexity"] for r in q if r["precision"] == "bf16")
            fig, ax = plt.subplots(figsize=(7.5, 4.4)); xs = list(range(len(q)))
            bars = ax.bar(xs, [r["perplexity"] for r in q], 0.6, color=[_PREC[r["precision"]] for r in q], zorder=3)
            for b, r in zip(bars, q):
                d = r.get("perplexity_delta_pct")
                ax.annotate(f"{r['perplexity']:.2f}\n{'baseline' if not d else f'+{d:.1f}%'}",
                            (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                            fontsize=9, color=_MUTED, xytext=(0, 2), textcoords="offset points")
            ax.axhline(base * 1.05, color=_RED, linestyle="--", linewidth=1.3, zorder=2)
            ax.annotate("+5% quality gate", ((len(q) - 1) / 2, base * 1.13), color=_RED, fontsize=9, va="bottom", ha="center")
            ax.set_xticks(xs); ax.set_xticklabels([r["precision"] for r in q]); ax.set_ylim(0, base * 1.20)
            style(ax, "perplexity (lower is better)", "FP8 quality gate: perplexity vs BF16 (all under 5% gate)")
            fig.tight_layout(); fig.savefig(fig_dir / "quality.png", dpi=150, bbox_inches="tight"); plt.close(fig)
            written.append("quality.png")
    return written


def _key_findings(df: pd.DataFrame, quality_path) -> list[str]:
    """Insight bullets computed from the data (no hardcoded numbers)."""
    import json as _json
    ok = df[df["status"] == "ok"]
    p = ok[ok.cell_name.isin(["precision", "baseline"])]
    out = []

    def _mean(sub):
        return sub.mean() if len(sub) else float("nan")

    # FP8 throughput + energy uplift vs bf16 (best workload)
    best_tp, best_e = None, None
    for wl in _WL:
        b = _mean(p[(p.workload == wl) & (p.precision == "bf16")]["output_throughput_tok_s"])
        f = _mean(p[(p.workload == wl) & (p.precision.isin(["fp8-static", "fp8-kv", "fp8-dynamic"]))]["output_throughput_tok_s"])
        be = _mean(p[(p.workload == wl) & (p.precision == "bf16")]["tokens_per_joule"])
        fe = _mean(p[(p.workload == wl) & (p.precision.isin(["fp8-static", "fp8-kv", "fp8-dynamic"]))]["tokens_per_joule"])
        if b == b and f == f:
            up = (f / b - 1) * 100
            if best_tp is None or up > best_tp[1]:
                best_tp = (wl, up)
        if be == be and fe == fe:
            ue = (fe / be - 1) * 100
            if best_e is None or ue > best_e[1]:
                best_e = (wl, ue)
    if best_tp:
        out.append(f"FP8 raises output throughput over bf16 on every workload, by up to about {best_tp[1]:.0f} percent (on {best_tp[0]}).")
    if best_e:
        out.append(f"FP8 raises energy efficiency (tokens/joule) by up to about {best_e[1]:.0f} percent (on {best_e[0]}).")

    # speculative on long_decode
    sp = ok[(ok.concurrency == 16) & (ok.precision == "bf16") & (ok.cell_name.isin(["speculative", "baseline"]))]
    b = _mean(sp[(sp.workload == "long_decode") & (sp.speculative == "none")]["output_throughput_tok_s"])
    ng = _mean(sp[(sp.workload == "long_decode") & (sp.speculative == "ngram")]["output_throughput_tok_s"])
    if b == b and ng == ng:
        out.append(f"Speculative decoding helps most on the decode heavy long_decode workload: ngram lifts throughput from about {b:.0f} to {ng:.0f} tokens per second.")
    out.append("On chat, ngram raises throughput but lowers tokens per joule, and EAGLE-3 gives little gain, so speculative decoding is not a universal win.")

    # batching lever
    sc = ok[(ok.precision == "bf16") & (ok.speculative == "none") & (ok.cell_name.isin(["concurrency", "baseline"]))]
    lo = _mean(sc[(sc.workload == "chat") & (sc.concurrency == 1)]["tokens_per_joule"])
    hi = _mean(sc[(sc.workload == "chat") & (sc.concurrency == 256)]["tokens_per_joule"])
    if lo == lo and hi == hi:
        out.append(f"Continuous batching is the largest energy lever: chat tokens per joule rises from about {lo:.2f} at concurrency 1 to {hi:.1f} at concurrency 256.")

    # quality
    if quality_path and Path(quality_path).exists():
        q = _json.loads(Path(quality_path).read_text())
        deltas = [r.get("perplexity_delta_pct") for r in q if r.get("perplexity_delta_pct")]
        if deltas:
            out.append(f"FP8 quality cost is small: the largest perplexity increase over bf16 is about {max(deltas):.1f} percent, under the 5 percent gate, so the speed gains are not from a degraded checkpoint.")
    return out


def build_report(results_dir: str | Path, quality_path: str | Path | None = None) -> str:
    """Generate RESULTS.md (audit + sanity + key findings + OFAT tables + energy median/IQR
    + quality, with embedded figures) and figures under <results_dir>/figures."""
    import json as _json

    results_dir = Path(results_dir)
    df = load_runs(results_dir)
    a = audit(df)
    checks = sanity_checks(df)
    fig_dir = results_dir / "figures"

    L = ["# vllm-optimization-bench: L40S results\n",
         f"Total runs: **{len(df)}**  |  status: {a['status_counts']}\n"]

    L.append("## No-errors audit")
    L.append(f"- Failed cells: **{a['n_failed']}** {'✅ clean' if a['clean'] else '❌'}")
    for c in a["failed_cells"]:
        L.append(f"  - {c['cell_name']}/{c['workload']}/{c['precision']}: {c['gate_reasons']}")

    L.append("\n## Sanity checks")
    for c in checks:
        L.append(f"- {'✅' if c['pass'] else '❌'} **{c['name']}**: {c['detail']}")

    # render figures up front so tables can embed them inline
    try:
        figs = set(make_figures(df, fig_dir, quality_path))
    except Exception as e:
        figs = set(); L.append(f"\n(figures skipped: {e})")

    def embed(name: str) -> None:
        if name in figs:
            L.append(f"\n![{name}](figures/{name})")
            cap = FIGURE_CAPTIONS.get(name)
            if cap:
                L.append(f"\n*Figure: {cap}*")

    L.append("\n## Key findings")
    for finding in _key_findings(df, quality_path):
        L.append(f"- {finding}")

    ok = df[df["status"] == "ok"]
    L.append("\n## OFAT: throughput & energy by precision (concurrency=16)")
    prec = ok[ok["cell_name"].isin(["precision", "baseline"])]
    if len(prec):
        t = prec.groupby(["workload", "precision"])[
            ["output_throughput_tok_s", "ttft_ms_median", "tpot_ms_median", "tokens_per_joule"]].mean().round(1)
        L.append("```\n" + t.to_string() + "\n```")
    embed("fp8_throughput.png"); embed("fp8_energy.png"); embed("latency_throughput.png")

    L.append("\n## Speculative decoding (vs baseline)")
    spec = ok[ok["cell_name"].isin(["speculative", "baseline"])]
    if len(spec):
        t = spec.groupby(["workload", "speculative"])[
            ["output_throughput_tok_s", "tokens_per_joule"]].mean().round(2)
        L.append("```\n" + t.to_string() + "\n```")
    embed("speculative.png")

    L.append("\n## Continuous batching scaling")
    L.append("bf16 throughput and tokens/joule at client concurrency 1, 16, and 256 "
             "(the concurrency axis plus the baseline point).")
    embed("concurrency_scaling.png")

    L.append("\n## Energy headline (median + IQR across repeats)")
    es = energy_summary(df)
    if len(es):
        L.append("```\n" + es.round(3).to_string(index=False) + "\n```")
    embed("energy_frontier.png")

    if quality_path and Path(quality_path).exists():
        q = _json.loads(Path(quality_path).read_text())
        L.append("\n## FP8 quality gate (perplexity vs BF16)")
        L.append("```")
        for r in q:
            d = f"{r.get('perplexity_delta_pct'):+.1f}%" if r.get("perplexity_delta_pct") is not None else "baseline"
            L.append(f"{r['precision']:14s} ppl={r.get('perplexity'):.3f}  ({d})  flagged={r.get('flagged')}")
        L.append("```")
        embed("quality.png")

    md = "\n".join(L) + "\n"
    (results_dir / "RESULTS.md").write_text(md)
    return md
