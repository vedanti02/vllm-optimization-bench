"""Phase 2 (CPU-testable slice): validity gate + tokens/joule + DCGM CSV parse."""

from vob.metrics import (
    apply_validity_gate,
    parse_dcgm_csv,
    reduce_telemetry,
    tokens_per_joule,
)
from vob.telemetry import DCGM_FIELDS


def _write_dmon_csv(path, rows):
    # Emulate `dcgmi dmon` whitespace table: "GPU <id> <field values...>"
    names = [name for _, name in DCGM_FIELDS]
    header = "# Entity " + " ".join(names)
    lines = [header]
    for r in rows:
        vals = " ".join(str(r[n]) for n in names)
        lines.append(f"GPU 0 {vals}")
    path.write_text("\n".join(lines) + "\n")


def _row(sm, power, **over):
    base = {name: 0.0 for _, name in DCGM_FIELDS}
    base.update({"sm_active": sm, "power_w": power})
    base.update(over)
    return base


def test_effective_num_prompts_scales_with_parallelism():
    from vob.runner import effective_num_prompts
    # low concurrency -> floored small count so serialized cells finish
    assert effective_num_prompts(200, 1, 256) == 16
    assert effective_num_prompts(200, 4, 256) == 64
    # high concurrency -> full configured count
    assert effective_num_prompts(200, 16, 256) == 200
    assert effective_num_prompts(500, 256, 256) == 500
    # max_num_seqs=1 serializes even with many clients -> floored small count
    assert effective_num_prompts(200, 16, 1) == 16
    # never exceeds configured
    assert effective_num_prompts(100, 256, 256) == 100


def test_tokens_per_joule():
    assert tokens_per_joule(300.0, 300.0) == 1.0
    assert tokens_per_joule(None, 300.0) is None
    assert tokens_per_joule(300.0, 0) is None


def test_dcgm_csv_parse_columns(tmp_path):
    csv = tmp_path / "dcgm.csv"
    _write_dmon_csv(csv, [_row(0.9, 320.0), _row(0.95, 330.0)])
    cols = parse_dcgm_csv(csv)
    assert cols["sm_active"] == [0.9, 0.95]
    assert cols["power_w"] == [320.0, 330.0]


def test_low_sm_active_is_ok_but_not_gpu_bound(tmp_path):
    csv = tmp_path / "dcgm.csv"
    # sustained low SM-active is NOT contamination (memory-bound / low-concurrency);
    # the run is 'ok' but flagged not-gpu-bound (informational only).
    _write_dmon_csv(csv, [_row(0.2, 120.0) for _ in range(10)])
    red = reduce_telemetry(csv, extras={"throttle_reasons": 0})
    assert red["status"] == "ok"
    assert red["gate_reasons"] == ""
    assert red["energy_gpu_bound"] is False


def test_gate_passes_clean_run(tmp_path):
    csv = tmp_path / "dcgm.csv"
    _write_dmon_csv(csv, [_row(0.95, 330.0) for _ in range(10)])
    red = reduce_telemetry(csv, extras={"throttle_reasons": 0, "neighbor_power_w": 250.0})
    assert red["status"] == "ok"
    assert red["gate_reasons"] == ""
    assert red["power_w_steady_mean"] == 330.0
    assert red["energy_gpu_bound"] is True


def test_gate_flags_thermal_throttle():
    reduced = {"sm_active_steady_mean": 0.95}
    # bit 3 set (a HW/thermal throttle reason, not the benign idle bit 0)
    status, reasons = apply_validity_gate(reduced, {"throttle_reasons": 0x8},
                                          gate={"allow_throttle": False, "max_neighbor_power_w": None})
    assert status == "contaminated"
    assert any("throttle" in r for r in reasons)


def test_gate_ignores_benign_idle_bit():
    reduced = {"sm_active_steady_mean": 0.20}
    # low SM-active + only the benign idle bit => still ok (not contaminated)
    status, _ = apply_validity_gate(reduced, {"throttle_reasons": 0x1},
                                    gate={"allow_throttle": False, "max_neighbor_power_w": None})
    assert status == "ok"
