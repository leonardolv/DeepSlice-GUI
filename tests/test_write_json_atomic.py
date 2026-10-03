"""write_json_atomic must never destroy an existing file on failure."""
import json
import os

import pytest

from DeepSlice.gui.state import write_json_atomic


def test_writes_payload(tmp_path):
    target = tmp_path / "a.json"
    write_json_atomic(str(target), {"x": 1}, indent=2)
    assert json.loads(target.read_text()) == {"x": 1}
    assert os.listdir(tmp_path) == ["a.json"]


def test_unserialisable_payload_keeps_previous_file(tmp_path):
    target = tmp_path / "a.json"
    target.write_text('{"old": true}')
    with pytest.raises(TypeError):
        write_json_atomic(str(target), {"bad": object()})
    assert json.loads(target.read_text()) == {"old": True}
    assert os.listdir(tmp_path) == ["a.json"]


def test_replace_failure_cleans_temp_and_keeps_old(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    target.write_text('{"old": true}')

    def boom(*_a, **_k):
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_json_atomic(str(target), {"new": 1})
    assert json.loads(target.read_text()) == {"old": True}
    assert os.listdir(tmp_path) == ["a.json"]
