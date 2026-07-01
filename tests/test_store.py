"""Phase 3/4: resumable store — completed cells skipped, contaminated re-run, upsert."""

from types import SimpleNamespace

from vob.store import ResultStore


def _cfg(cell_id):
    return SimpleNamespace(cell_id=cell_id)


def test_append_and_completed(tmp_path):
    store = ResultStore(tmp_path)
    store.append_row({"cell_id": "a", "status": "ok", "tokens_per_joule": 1.0})
    store.append_row({"cell_id": "b", "status": "ok"})
    assert store.completed_cell_ids() == {"a", "b"}


def test_pending_skips_completed(tmp_path):
    store = ResultStore(tmp_path)
    store.append_row({"cell_id": "a", "status": "ok"})
    pending = store.pending([_cfg("a"), _cfg("b"), _cfg("c")])
    assert {c.cell_id for c in pending} == {"b", "c"}


def test_contaminated_not_complete_by_default(tmp_path):
    store = ResultStore(tmp_path)
    store.append_row({"cell_id": "x", "status": "contaminated"})
    assert store.completed_cell_ids() == set()          # re-run by default
    assert store.completed_cell_ids(include_contaminated=True) == {"x"}


def test_upsert_replaces_same_cell(tmp_path):
    store = ResultStore(tmp_path)
    store.append_row({"cell_id": "x", "status": "contaminated", "tokens_per_joule": 0.5})
    store.append_row({"cell_id": "x", "status": "ok", "tokens_per_joule": 1.2})
    df = store.load()
    assert len(df[df["cell_id"] == "x"]) == 1
    assert df[df["cell_id"] == "x"]["status"].iloc[0] == "ok"


def test_write_raw(tmp_path):
    store = ResultStore(tmp_path)
    p = store.write_raw("cell1", "bench.json", '{"ok": true}')
    assert p.exists()
    assert p.read_text() == '{"ok": true}'
