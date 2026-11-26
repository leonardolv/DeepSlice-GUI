import json

import pytest

from DeepSlice.training import training_utils


def test_metadata_write_keeps_previous_file_on_failure(tmp_path):
    path = tmp_path / "meta.json"
    training_utils.save_training_run_metadata(str(path), {"a": 1})
    with pytest.raises(TypeError):
        training_utils.save_training_run_metadata(str(path), {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]
