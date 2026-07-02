"""Phase 3/4: resumable, parallel-safe store — per-cell rows, skip/resume, upsert."""

from types import SimpleNamespace

from vob.store import ResultStore


def _cfg(cell_id):
    return SimpleNamespace(cell_id=cell_id)


def test_write_and_completed(tmp_path):
    store = ResultStore(tmp_path)
    store.write_row({"cell_id": "a", "status": "ok", "tokens_per_joule": 1.0})
    store.write_row({"cell_id": "b", "status": "ok"})
    assert store.completed_cell_ids() == {"a", "b"}
    # one JSON file per cell (parallel-safe, no shared parquet write)
    assert {p.stem for p in (tmp_path / "rows").glob("*.json")} == {"a", "b"}


def test_pending_skips_completed(tmp_path):
    store = ResultStore(tmp_path)
    store.write_row({"cell_id": "a", "status": "ok"})
    pending = store.pending([_cfg("a"), _cfg("b"), _cfg("c")])
    assert {c.cell_id for c in pending} == {"b", "c"}


def test_contaminated_complete_by_default_but_rerunnable(tmp_path):
    store = ResultStore(tmp_path)
    store.write_row({"cell_id": "x", "status": "contaminated"})
    # default: contaminated counts as complete (no infinite re-run loop)
    assert store.completed_cell_ids() == {"x"}
    # explicit opt-in re-runs it
    assert store.completed_cell_ids(include_contaminated=False) == set()


def test_upsert_replaces_same_cell(tmp_path):
    store = ResultStore(tmp_path)
    store.write_row({"cell_id": "x", "status": "contaminated", "tokens_per_joule": 0.5})
    store.write_row({"cell_id": "x", "status": "ok", "tokens_per_joule": 1.2})
    df = store.load()
    assert len(df[df["cell_id"] == "x"]) == 1
    assert df[df["cell_id"] == "x"]["status"].iloc[0] == "ok"


def test_export_parquet_roundtrip(tmp_path):
    store = ResultStore(tmp_path)
    store.write_row({"cell_id": "a", "status": "ok", "tokens_per_joule": 2.0})
    df = store.export_parquet()
    assert store.parquet_path.exists() and len(df) == 1


def test_write_raw(tmp_path):
    store = ResultStore(tmp_path)
    p = store.write_raw("cell1", "bench.json", '{"ok": true}')
    assert p.exists()
    assert p.read_text() == '{"ok": true}'
