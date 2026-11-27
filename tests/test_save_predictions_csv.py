"""save_predictions: bad format writes nothing; CSV write is atomic."""
import os

import pandas as pd
import pytest

from DeepSlice.main import DSModel


def _model():
    m = DSModel.__new__(DSModel)
    m.species = "mouse"
    m.config = {
        "target_volumes": {"mouse": {"name": "t"}},
        "DeepSlice_version": {"prerelease": "x"},
    }
    cols = {k: [1.0] for k in ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]}
    m.predictions = pd.DataFrame(
        {"Filenames": ["a"], **cols, "height": [5], "width": [6], "markers": [[]]}
    )
    return m


def test_invalid_format_leaves_no_csv(tmp_path):
    with pytest.raises(ValueError):
        _model().save_predictions(str(tmp_path / "out"), output_format="yaml")
    assert os.listdir(tmp_path) == []


def test_failed_csv_write_keeps_previous_file(tmp_path, monkeypatch):
    (tmp_path / "out.csv").write_text("old")

    def boom(self, *_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        _model().save_predictions(str(tmp_path / "out"))
    assert (tmp_path / "out.csv").read_text() == "old"
    assert os.listdir(tmp_path) == ["out.csv"]


def test_valid_save_writes_csv_and_json(tmp_path):
    _model().save_predictions(str(tmp_path / "out"))
    assert sorted(os.listdir(tmp_path)) == ["out.csv", "out.json"]
