"""QUINT JSON / QuickNII XML exports must not destroy a previous file on failure."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from DeepSlice.read_and_write import QuickNII_functions as q


def _df(markers):
    cols = {k: [1.0] for k in ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]}
    return pd.DataFrame(
        {"Filenames": ["a"], **cols, "height": [5], "width": [6], "markers": [markers]}
    )


def test_ndarray_markers_are_serialised(tmp_path):
    base = str(tmp_path / "out")
    q.write_QUINT_JSON(_df(np.array([[1, 2]])), base, "x", "t")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["slices"][0]["markers"] == [[1, 2]]
    assert os.listdir(tmp_path) == ["out.json"]


def test_failed_json_export_keeps_previous_file(tmp_path):
    base = str(tmp_path / "out")
    (tmp_path / "out.json").write_text('{"old": 1}')
    with pytest.raises(TypeError):
        q.write_QUINT_JSON(_df(object()), base, "x", "t")
    assert json.loads((tmp_path / "out.json").read_text()) == {"old": 1}
    assert os.listdir(tmp_path) == ["out.json"]


def test_failed_xml_export_keeps_previous_file(tmp_path, monkeypatch):
    base = str(tmp_path / "out")
    (tmp_path / "out.xml").write_text("<old/>")

    def boom(self, target, *_a, **_k):
        with open(target, "w") as handle:
            handle.write("<partial")
        raise OSError("disk full")

    monkeypatch.setattr(q.ET.ElementTree, "write", boom)
    df = _df([]).drop(columns="markers")
    with pytest.raises(OSError):
        q.write_QuickNII_XML(df, base, "x")
    assert (tmp_path / "out.xml").read_text() == "<old/>"
    assert os.listdir(tmp_path) == ["out.xml"]
