"""GPU telemetry via DCGM (Phase 2), with a pynvml fallback.

Maps CUDA_VISIBLE_DEVICES -> a physical DCGM GPU-UUID, then samples that ONE GPU
with `dcgmi dmon` into a per-run CSV over the benchmark window. Pinning to the
UUID is what makes the metrics *my* GPU's even while neighbors run on the shared
node.

DCGM field IDs (confirm with `dcgmi dmon -l` in Phase 2 — the exact ids can vary
by DCGM version; these are the standard L40S set):
    155  POWER_USAGE (W)
    203  GPU_UTIL (%)
    1002 SM_ACTIVE (0-1)       <- the validity-gate signal
    1004 TENSOR_ACTIVE (0-1)
    1005 DRAM_ACTIVE (0-1)
    252  FB_USED (MiB)         (verified via `dcgmi dmon -l`: FBUSD=252, not 250)
    100  SM_CLOCK (MHz)
    101  MEM_CLOCK (MHz)
    150  GPU_TEMP (C)
    112  CLOCK_THROTTLE_REASONS (DVCCTR) — read via pynvml instead of dmon
Throttle reasons are read separately via nvml (CLOCK_THROTTLE_REASONS) since not
all DCGM builds expose them as a dmon field.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# (field_id, tidy_column) — order defines the dmon -e list and CSV columns.
DCGM_FIELDS = [
    (155, "power_w"),
    (203, "gpu_util"),
    (1002, "sm_active"),
    (1004, "tensor_active"),
    (1005, "dram_active"),
    (252, "fb_used_mib"),
    (100, "sm_clock_mhz"),
    (101, "mem_clock_mhz"),
    (150, "temp_c"),
]


def resolve_gpu_uuid(visible: Optional[str] = None) -> Optional[str]:
    """Map the first CUDA_VISIBLE_DEVICES entry to a DCGM/nvml GPU-UUID.

    Inside a SLURM allocation CUDA_VISIBLE_DEVICES may already be a UUID
    ("GPU-xxxx") or a physical index; handle both. Returns None if it can't be
    resolved (caller then falls back to pynvml or logs a warning)."""
    visible = visible if visible is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = visible.split(",")[0].strip() if visible else ""
    if first.startswith("GPU-") or first.startswith("MIG-"):
        return first

    # Physical index -> UUID via nvidia-smi (works inside the allocation).
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, timeout=15,
        )
    except Exception:
        return None
    idx = first if first else "0"
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == idx:
            return parts[1]
    return None


@dataclass
class TelemetryHandle:
    proc: Optional[subprocess.Popen]
    csv_path: str
    uuid: Optional[str]
    backend: str  # "dcgmi" | "pynvml" | "none"
    t_start: float


def _dmon_command(uuid: str, csv_path: str, interval_ms: int) -> list[str]:
    fields = ",".join(str(fid) for fid, _ in DCGM_FIELDS)
    # -i pins to the GPU UUID; -d is the sample interval; -e the field list.
    return ["dcgmi", "dmon", "-i", uuid, "-e", fields, "-d", str(interval_ms)]


def start(csv_path: str, *, uuid: Optional[str] = None, interval_ms: int = 200) -> TelemetryHandle:
    """Begin sampling the pinned GPU into csv_path. Returns a handle to stop().

    Prefers `dcgmi dmon`; if dcgmi/UUID are unavailable, returns a backend="none"
    handle and the caller may start the pynvml sampler instead."""
    uuid = uuid or resolve_gpu_uuid()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    if uuid and _has_dcgmi():
        f = open(csv_path, "w")
        proc = subprocess.Popen(
            _dmon_command(uuid, csv_path, interval_ms),
            stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        return TelemetryHandle(proc=proc, csv_path=csv_path, uuid=uuid,
                               backend="dcgmi", t_start=time.monotonic())

    return TelemetryHandle(proc=None, csv_path=csv_path, uuid=uuid,
                           backend="none", t_start=time.monotonic())


def stop(handle: TelemetryHandle) -> None:
    if handle.proc is None:
        return
    with_suppress(lambda: os.killpg(os.getpgid(handle.proc.pid), signal.SIGTERM))
    try:
        handle.proc.wait(timeout=10)
    except Exception:
        with_suppress(lambda: os.killpg(os.getpgid(handle.proc.pid), signal.SIGKILL))


def _has_dcgmi() -> bool:
    try:
        subprocess.check_output(["dcgmi", "--version"], stderr=subprocess.STDOUT, timeout=10)
        return True
    except Exception:
        return False


def with_suppress(fn) -> None:
    try:
        fn()
    except Exception:
        pass


def read_throttle_and_neighbors(uuid: Optional[str]) -> dict:
    """Snapshot clock-throttle reasons for my GPU + aggregate neighbor power via
    pynvml. Used by metrics.py to detect thermal/contention contamination.

    Returns {} if pynvml is unavailable."""
    try:
        import pynvml
    except Exception:
        return {}

    result: dict = {}
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        my_power_w = None
        neighbor_power_w = 0.0
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            dev_uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(dev_uuid, bytes):
                dev_uuid = dev_uuid.decode()
            p_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            if uuid and dev_uuid == uuid:
                my_power_w = p_w
                try:
                    result["throttle_reasons"] = int(
                        pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
                except Exception:
                    result["throttle_reasons"] = None
            else:
                neighbor_power_w += p_w
        result["my_power_w_nvml"] = my_power_w
        result["neighbor_power_w"] = round(neighbor_power_w, 1)
    except Exception:
        return result
    finally:
        with_suppress(pynvml.nvmlShutdown)
    return result
