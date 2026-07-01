# vllm-optimization-bench

A rigorous, reproducible benchmark suite for **vLLM inference optimizations on the
NVIDIA L40S** — a GPU under-represented in published vLLM data (most is H100/A100).
It measures how FP8 quantization, EAGLE-3 / n-gram speculative decoding, chunked
prefill, continuous batching (`max-num-seqs`), and request concurrency trade off
across three axes: **latency, throughput, and energy (tokens-per-joule)**.

Built to run on the CMU LTI **Babel** shared SLURM cluster (8× L40S nodes), with
per-GPU DCGM telemetry and a validity gate so energy numbers are trustworthy even
on a shared node.

> Status: harness code complete + unit-tested (CPU). GPU phases (bring-up,
> telemetry wiring, sweeps) run inside a SLURM allocation — see `scratchpad.md`
> for the live progress log.

## What it produces
One tidy row per `(optimization config × workload × repeat)` with:
`TTFT / TPOT / throughput` (from `vllm bench serve`) **+** `tokens/joule`,
`SM-active`, `power`, clocks, throttle flags (from DCGM), a validity `status`
(`ok | contaminated | failed`), and full reproducibility pins (vllm/torch/CUDA/
driver/GPU-UUID/node). Results land in `results/L40S/runs.parquet`.

## Design in one screen
- **OFAT + interactions** (`configs/matrix.yaml`): each optimization is swept
  one-factor-at-a-time from a baseline, plus ~5 targeted interaction cells (the
  blog hooks, e.g. *EAGLE-3 × concurrency crossover*).
- **4 workloads** (`configs/workloads.yaml`): `chat`, `long_prompt` (prefill-heavy),
  `long_decode` (decode-heavy), `saturation` (scheduler stress).
- **Energy validity gate** (`src/vob/metrics.py`): a `tokens/joule` run counts only
  if GPU-bound (`SM_ACTIVE ≥ 0.80` in steady decode) with no thermal throttle and
  stable clocks; failing runs are marked `contaminated` and re-run, never averaged
  in. Energy-headline cells are repeated 3–5× and reported as **median + IQR**.
- **FP8 correctness gate** (`src/vob/quality.py`): before trusting any "FP8 is
  faster" claim, each FP8 variant's perplexity delta vs BF16 is recorded.
- **Resumable** (`src/vob/store.py`): cells are keyed by `cell_id`; a killed sweep
  restarts and skips completed cells.
- **Budget guard** (`src/vob/cli.py`): per-run wall-clock cap + cumulative
  GPU-hour ceiling (`configs/matrix.yaml → budget`).

## Reproduce on Babel

### 0. Build the CUDA env (once, inside a GPU allocation)
System Python is 3.9 (too old for recent vLLM); we provision ≥3.10 via `uv`.
```bash
srun --partition=general --gres=gpu:L40S:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
cd vllm-optimization-bench
bash scripts/setup_env.sh          # module load cuda-12.4; uv venv; pip install vllm + this pkg
```
Record the printed vllm/torch/CUDA versions in `scratchpad.md`, and confirm DCGM
field ids with `dcgmi dmon -l` (Phase 2).

### 1. Storage (off `$HOME` — home NFS has ~15 GB free)
```bash
export HF_HOME=/data/hf_cache                      # read existing Llama-3.1 weights
export HF_HUB_CACHE=/data/user_data/$USER/hf       # writable overlay for new FP8 downloads
```

### 2. Plan the matrix (CPU-only, no GPU)
```bash
vob plan --show          # expand OFAT + interaction cells; lists any pruned invalid combos
```

### 3. Smoke one cell end-to-end (inside the allocation)
```bash
vob run --limit 1        # launch server -> DCGM telemetry -> bench -> parquet row -> teardown
```

### 4. Run the sweep unattended
```bash
sbatch slurm/bench.sbatch                       # or: sbatch --array=0-3 slurm/bench.sbatch
# resumable: overlapping array tasks skip already-completed cells
```
Prioritize the blog hooks first if time is short:
```bash
vob run --only-source interaction
```

### 5. Analyze (CPU-only)
```bash
python -c "from vob.analyze import load_runs, energy_summary; \
           print(energy_summary(load_runs('results/L40S')))"
# or open notebooks/analysis.ipynb
```

## Retargeting to another GPU
The harness is not L40S-specific. To run elsewhere: change `--gres`/partition in
`slurm/bench.sbatch` (or run `vob run` directly on a bare GPU host), point the
`models:` block in `configs/matrix.yaml` at your checkpoints, and adjust
`sm_active_min` in the gate if your GPU's steady-decode utilization differs.
DCGM telemetry falls back to `pynvml` if `dcgmi` is unavailable.

## Development (no GPU required)
```bash
pip install -e ".[dev]"        # CPU tooling only; vllm is the ".[gpu]" extra
pytest -q                      # matrix expansion, validity gate, resumable store
```

## Shared-node caveat (read before citing energy numbers)
These runs share an 8× L40S node with other users. Per-GPU DCGM isolates the
joules denominator to *our* card, but the shared thermal/power envelope can
perturb clocks. We handle this with the validity gate + repeats above and report
`tokens/joule` alongside the logged clocks/SM-active so the operating point is
visible. Numbers are conditioned on the shared-node state, stated explicitly.

## License
MIT — see [LICENSE](LICENSE).
