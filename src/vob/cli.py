"""`vob` CLI: expand the matrix, run cells under a budget guard, submit SLURM jobs.

Two execution modes (plan-locked):
  * `vob run`    — runs the loop INSIDE the current allocation (call from sbatch/srun).
  * `vob submit` — generates + submits the sbatch array so sweeps run unattended.

Resumability: every cell is keyed by cell_id; completed cells are skipped, so a
killed sweep restarts cleanly across separate SLURM jobs.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import metrics as M
from . import runner as R
from . import server as S
from . import telemetry as T
from .matrix import attach_workload_shape, expand_matrix
from .store import ResultStore, load_yaml

app = typer.Typer(add_completion=False, help="vLLM optimization benchmark harness (L40S / Babel).")
console = Console()

_REPO = Path(__file__).resolve().parents[2]
_DEF_MATRIX = _REPO / "configs" / "matrix.yaml"
_DEF_WORKLOADS = _REPO / "configs" / "workloads.yaml"
_DEF_RESULTS = _REPO / "results" / "L40S"


def collect_pins() -> dict:
    """Reproducibility metadata stamped into every row: versions + node + GPU UUID."""
    pins: dict = {"node": os.uname().nodename}
    try:
        import vllm  # type: ignore
        pins["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        pins["vllm_version"] = None
    try:
        import torch  # type: ignore
        pins["torch_version"] = torch.__version__
        pins["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pins["torch_version"] = pins.get("cuda_version")
    # Driver version from /proc (works outside allocation too).
    try:
        drv = Path("/proc/driver/nvidia/version")
        if drv.exists():
            pins["driver_version"] = drv.read_text().splitlines()[0]
    except Exception:
        pass
    pins["gpu_uuid"] = T.resolve_gpu_uuid()
    pins["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return pins


def _load_configs(matrix_path: Path, workloads_path: Path):
    matrix = load_yaml(matrix_path)
    workloads = load_yaml(workloads_path)
    configs, pruned = expand_matrix(matrix, workloads)
    return matrix, workloads, configs, pruned


@app.command()
def plan(
    matrix: Path = typer.Option(_DEF_MATRIX),
    workloads: Path = typer.Option(_DEF_WORKLOADS),
    show: bool = typer.Option(False, help="print every expanded cell"),
):
    """Expand the matrix and print the plan (CPU-only, no GPU needed)."""
    _, _, configs, pruned = _load_configs(matrix, workloads)
    console.print(f"[bold green]{len(configs)}[/] cells; [yellow]{len(pruned)}[/] pruned as invalid.")
    for p in pruned:
        console.print(f"  [yellow]pruned[/] {p['knobs']} ({p['workload']}): {p['reason']}")
    if show:
        table = Table("cell_id", "source", "cell_name", "workload", "precision", "spec", "cp", "seqs", "conc", "rep")
        for c in configs:
            table.add_row(c.cell_id, c.source, c.cell_name, c.workload, c.precision,
                          c.speculative, str(c.chunked_prefill), str(c.max_num_seqs),
                          str(c.concurrency), str(c.repeat_idx))
        console.print(table)


@app.command()
def run(
    matrix: Path = typer.Option(_DEF_MATRIX),
    workloads: Path = typer.Option(_DEF_WORKLOADS),
    results: Path = typer.Option(_DEF_RESULTS),
    only_source: Optional[str] = typer.Option(None, help="baseline|ofat|interaction — restrict to one group"),
    only_cell: Optional[str] = typer.Option(None, help="run only this cell_name (axis/interaction name)"),
    limit: Optional[int] = typer.Option(None, help="cap number of cells this invocation (smoke tests)"),
    include_contaminated: bool = typer.Option(False, help="treat contaminated rows as complete (skip re-run)"),
):
    """Run pending cells INSIDE the current SLURM allocation, under the budget guard."""
    matrix_cfg, workloads_cfg, configs, pruned = _load_configs(matrix, workloads)
    store = ResultStore(results)
    store.ensure_dirs()

    budget = matrix_cfg.get("budget", {})
    gpu_hours_ceiling = float(budget.get("gpu_hours_ceiling", 1e9))
    per_run_cap_s = int(budget.get("per_run_wallclock_cap_s", 1800))
    gate_cfg = matrix_cfg.get("gate", {})

    pins = collect_pins()
    console.print(f"[bold]pins[/]: {pins}")

    # Filter to pending + requested subset.
    todo = store.pending(configs, include_contaminated=include_contaminated)
    if only_source:
        todo = [c for c in todo if c.source == only_source]
    if only_cell:
        todo = [c for c in todo if c.cell_name == only_cell]
    if limit:
        todo = todo[:limit]

    console.print(f"[bold]{len(todo)}[/] pending cells to run "
                  f"([green]{len(configs) - len(todo)}[/] already complete).")

    spent_gpu_s = _prior_gpu_seconds(store)
    for i, cfg in enumerate(todo, 1):
        if spent_gpu_s / 3600.0 >= gpu_hours_ceiling:
            console.print(f"[red]BUDGET GUARD[/]: {spent_gpu_s/3600:.1f} GPU-h >= ceiling "
                          f"{gpu_hours_ceiling} — stopping. {len(todo) - i + 1} cells left.")
            break
        console.print(f"[cyan][{i}/{len(todo)}][/] {cfg.cell_name}/{cfg.workload} "
                      f"prec={cfg.precision} spec={cfg.speculative} conc={cfg.concurrency} rep={cfg.repeat_idx}")
        elapsed = _run_one(cfg, workloads_cfg, store, pins, per_run_cap_s, gate_cfg)
        spent_gpu_s += elapsed

    console.print(f"[bold green]done[/]; ~{spent_gpu_s/3600:.2f} GPU-h this store.")


def _prior_gpu_seconds(store: ResultStore) -> float:
    df = store.load()
    if df.empty or "bench_elapsed_s" not in df.columns:
        return 0.0
    return float(df["bench_elapsed_s"].fillna(0).sum())


def _run_one(cfg, workloads_cfg, store: ResultStore, pins: dict,
             per_run_cap_s: int, gate_cfg: dict) -> float:
    """Launch server -> telemetry -> bench -> reduce -> store one cell. Returns wall seconds."""
    shape = attach_workload_shape(cfg, workloads_cfg)
    raw_result = str(store.raw_dir / cfg.cell_id / "bench.json")
    dcgm_csv = str(store.raw_dir / cfg.cell_id / "dcgm.csv")
    server_log = str(store.raw_dir / cfg.cell_id / "server.log")
    (store.raw_dir / cfg.cell_id).mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    record: dict
    try:
        with S.serving(cfg, log_path=server_log) as handle:
            tele = T.start(dcgm_csv)
            try:
                bench = R.run_benchmark(cfg, shape, base_url=handle.base_url,
                                        result_path=raw_result, wallclock_cap_s=per_run_cap_s)
            finally:
                T.stop(tele)
            extras = T.read_throttle_and_neighbors(tele.uuid)
            telemetry = M.reduce_telemetry(dcgm_csv, extras=extras, gate=gate_cfg)
            record = M.build_record(cfg, bench, telemetry, pins={**pins, "telemetry_backend": tele.backend})
    except Exception as e:  # server OOM / early exit / timeout — record as failed, keep going
        record = {
            "cell_id": cfg.cell_id, "cell_name": cfg.cell_name, "source": cfg.source,
            "workload": cfg.workload, "repeat_idx": cfg.repeat_idx,
            "precision": cfg.precision, "speculative": cfg.speculative,
            "concurrency": cfg.concurrency, "status": "failed", "gate_reasons": str(e)[:300],
            **pins,
        }
        console.print(f"    [red]failed[/]: {e}")

    store.append_row(record)
    elapsed = time.monotonic() - t0
    status = record.get("status", "?")
    tpj = record.get("tokens_per_joule")
    console.print(f"    -> status={status} tok/J={tpj} ({elapsed:.0f}s)")
    return elapsed


@app.command()
def submit(
    matrix: Path = typer.Option(_DEF_MATRIX),
    template: Path = typer.Option(_REPO / "slurm" / "bench.sbatch"),
    only_source: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(False, help="print sbatch command, don't submit"),
):
    """Submit the sweep as a SLURM job (reuses slurm/bench.sbatch). Resumable across jobs."""
    cmd = ["sbatch", str(template)]
    if only_source:
        cmd += ["--export", f"ALL,VOB_ONLY_SOURCE={only_source}"]
    console.print(f"[bold]submit[/]: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
