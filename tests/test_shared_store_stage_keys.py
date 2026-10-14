"""The stage-key alias must skip work without ever changing where bytes land.

The optimisation it guards is large -- 11.4s of a 22s sweep job -- but it is
only safe if a hit returns exactly the entry a full content-hash would have
produced. These tests pin that, plus the two ways the alias can be wrong:
a pruned entry, and an unwritable index.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.pipeline.shared_store import SharedStore, content_key


def _writer(payload: bytes):
    """A `put_written`-style writer producing a windows entry, counting calls."""
    calls = []

    def write(target):
        calls.append(target)
        (target / "windows.npz").write_bytes(payload)
        (target / "windows.json").write_text('{"episodes": [], "windows": []}')

    return write, calls


def test_alias_hit_returns_the_same_entry_and_skips_the_writer(tmp_path):
    store = SharedStore(tmp_path)
    write, calls = _writer(b"\x01" * 4096)

    first = store.put_written("windows", write, stage_key="cachekey1")
    assert len(calls) == 1

    # Same stage key: the writer must not run again, and the entry must be the
    # very same directory -- not merely an equivalent one.
    second = store.put_written("windows", write, stage_key="cachekey1")
    assert len(calls) == 1, "writer ran again despite a recorded stage key"
    assert second == first


def test_alias_agrees_with_the_content_hash(tmp_path):
    """A hit must land where content-addressing alone would have put it."""
    store = SharedStore(tmp_path)
    payload = b"\x02" * 8192
    write, _ = _writer(payload)
    entry = store.put_written("windows", write, stage_key="k")

    expected = content_key(
        {"windows.npz": entry / "windows.npz", "windows.json": entry / "windows.json"}
    )
    assert entry.name == expected


def test_no_stage_key_still_dedupes_by_content(tmp_path):
    """The old path is untouched: identical bytes converge with no alias."""
    store = SharedStore(tmp_path)
    write_a, calls_a = _writer(b"\x03" * 2048)
    write_b, calls_b = _writer(b"\x03" * 2048)
    a = store.put_written("windows", write_a)
    b = store.put_written("windows", write_b)
    assert a == b
    # Both writers run -- that is the cost the stage key exists to remove -- but
    # only one entry results.
    assert len(calls_a) == 1 and len(calls_b) == 1


def test_different_stage_keys_same_content_share_one_entry(tmp_path):
    store = SharedStore(tmp_path)
    write, calls = _writer(b"\x04" * 1024)
    a = store.put_written("windows", write, stage_key="one")
    b = store.put_written("windows", write, stage_key="two")
    assert a == b
    assert len(calls) == 2  # distinct keys, so the second could not be skipped


def test_pruned_entry_is_rewritten_not_pointed_at(tmp_path):
    """A stale alias must fall back to writing, not return a missing directory."""
    import shutil

    store = SharedStore(tmp_path)
    write, calls = _writer(b"\x05" * 512)
    entry = store.put_written("windows", write, stage_key="k")
    shutil.rmtree(entry)

    again = store.put_written("windows", write, stage_key="k")
    assert len(calls) == 2, "a pruned entry was pointed at instead of rewritten"
    assert (again / "windows.npz").exists()
    assert again == entry  # same content, so the same key


def test_lookup_ignores_a_partial_entry(tmp_path):
    store = SharedStore(tmp_path)
    write, calls = _writer(b"\x06" * 512)
    entry = store.put_written("windows", write, stage_key="k")
    (entry / "windows.json").unlink()  # half of the file set

    assert store.lookup("windows", "k") is None
    store.put_written("windows", write, stage_key="k")
    assert len(calls) == 2


def test_unwritable_index_does_not_break_the_write(tmp_path, caplog):
    """A failed alias costs one redundant serialise; it must not fail the run."""
    store = SharedStore(tmp_path)
    write, calls = _writer(b"\x07" * 256)
    # Make the index location a file, so mkdir of its parent fails.
    (tmp_path / "_shared").mkdir(parents=True, exist_ok=True)
    (tmp_path / "_shared" / ".stage_keys").write_text("not a directory")

    entry = store.put_written("windows", write, stage_key="k")
    assert (entry / "windows.npz").exists()
    assert len(calls) == 1
    assert store.lookup("windows", "k") is None


def test_round_trips_a_real_array(tmp_path):
    """End to end with npz payloads, since that is what the pipeline writes."""
    store = SharedStore(tmp_path)
    arr = np.arange(3000, dtype=np.float32).reshape(100, 30)

    def write(target):
        np.savez_compressed(target / "windows.npz", features=arr)
        (target / "windows.json").write_text('{"episodes": [], "windows": []}')

    entry = store.put_written("windows", write, stage_key="real")
    hit = store.put_written("windows", write, stage_key="real")
    assert hit == entry
    with np.load(hit / "windows.npz") as z:
        np.testing.assert_array_equal(z["features"], arr)
