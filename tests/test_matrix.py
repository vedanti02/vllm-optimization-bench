"""Phase 3 exit: matrix expands to the expected cells + drops a planted invalid combo."""

from vob.matrix import expand_matrix, normalize_dependencies

WORKLOADS = {
    "defaults": {"dataset": "random", "seed": 1},
    "workloads": {
        "chat": {"input_len": 128, "output_len": 128, "num_prompts": 10},
        "long_decode": {"input_len": 64, "output_len": 256, "num_prompts": 10},
    },
}


def _tiny_matrix(invalid=None):
    return {
        "models": {
            "bf16": {"id": "unsloth/Llama-3.1-8B-Instruct", "quantization": None},
            "fp8-static": {"id": "RedHatAI/FP8", "quantization": "fp8"},
        },
        "speculative_models": {"eagle3": "yuhuili/EAGLE3", "ngram": None},
        "baseline": {
            "precision": "bf16", "speculative": "none", "chunked_prefill": True,
            "max_num_seqs": 256, "concurrency": 16, "kv_cache_dtype": "auto",
        },
        "ofat": {
            "precision": {"levels": ["bf16", "fp8-static", "fp8-kv"]},
            "concurrency": {"levels": [1, 16, 256]},
        },
        "interactions": [
            {"name": "eagle3_x_conc", "fixed": {"speculative": "eagle3"},
             "sweep": {"concurrency": [1, 256]}, "workloads": ["chat"], "repeats": 2},
        ],
        "ofat_workloads": ["chat", "long_decode"],
        "invalid_combos": invalid if invalid is not None else [],
    }


def test_baseline_emitted_once_per_workload():
    configs, _ = expand_matrix(_tiny_matrix(), WORKLOADS)
    baselines = [c for c in configs if c.source == "baseline"]
    assert {c.workload for c in baselines} == {"chat", "long_decode"}
    assert len(baselines) == 2


def test_ofat_skips_baseline_level_but_keeps_fp8_kv():
    configs, _ = expand_matrix(_tiny_matrix(), WORKLOADS)
    prec_cells = [c for c in configs if c.cell_name == "precision"]
    precisions = {c.precision for c in prec_cells}
    # bf16 == baseline precision -> not re-emitted as an OFAT precision cell
    assert "bf16" not in precisions
    assert {"fp8-static", "fp8-kv"} <= precisions
    # fp8-kv resolves to fp8-static weights + fp8 KV cache
    kv = next(c for c in prec_cells if c.precision == "fp8-kv")
    assert kv.model_id == "RedHatAI/FP8"
    assert kv.kv_cache_dtype == "fp8"


def test_eagle3_normalizes_chunked_prefill_off_not_pruned():
    configs, _ = expand_matrix(_tiny_matrix(), WORKLOADS)
    eagle = [c for c in configs if c.speculative == "eagle3"]
    assert eagle, "EAGLE-3 interaction cells must survive (normalized, not pruned)"
    assert all(c.chunked_prefill is False for c in eagle)
    # 2 concurrency levels x 1 workload x 2 repeats = 4 cells
    assert len(eagle) == 4
    assert {c.repeat_idx for c in eagle} == {0, 1}


def test_planted_invalid_combo_is_pruned_and_logged():
    # Plant a combo the matrix actually generates: EAGLE-3 at concurrency=256
    # (from the eagle3_x_conc interaction sweep). Simulates "this OOMs at conc=256".
    invalid = [{"when": {"speculative": "eagle3", "concurrency": 256},
                "reason": "planted: EAGLE-3 OOMs at conc=256"}]
    configs, pruned = expand_matrix(_tiny_matrix(invalid), WORKLOADS)
    assert any(p["knobs"]["speculative"] == "eagle3" and p["knobs"]["concurrency"] == 256
               for p in pruned), "planted invalid combo must be logged as pruned"
    # The conc=256 eagle3 cells are gone; the conc=1 eagle3 cells survive.
    assert not any(c.speculative == "eagle3" and c.concurrency == 256 for c in configs)
    assert any(c.speculative == "eagle3" and c.concurrency == 1 for c in configs)


def test_cell_ids_unique_and_stable():
    configs, _ = expand_matrix(_tiny_matrix(), WORKLOADS)
    ids = [c.cell_id for c in configs]
    assert len(ids) == len(set(ids)), "cell_ids must be unique (resumability key)"
    # stable across a re-expansion
    again, _ = expand_matrix(_tiny_matrix(), WORKLOADS)
    assert {c.cell_id for c in configs} == {c.cell_id for c in again}


def test_invalid_combo_can_key_on_workload():
    invalid = [{"when": {"speculative": "eagle3", "workload": "chat"},
                "reason": "planted: eagle3 x chat invalid"}]
    configs, pruned = expand_matrix(_tiny_matrix(invalid), WORKLOADS)
    # eagle3 chat cells pruned; eagle3 on other workloads (none here) unaffected
    assert any(p["knobs"]["speculative"] == "eagle3" and p["workload"] == "chat" for p in pruned)
    assert not any(c.speculative == "eagle3" and c.workload == "chat" for c in configs)


def test_normalize_dependencies_is_noop_without_eagle3():
    knobs = {"speculative": "none", "chunked_prefill": True}
    notes = normalize_dependencies(knobs)
    assert notes == []
    assert knobs["chunked_prefill"] is True
