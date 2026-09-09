"""Tests for the neural network builder and the name-based weight loader."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def tf_module():
    """Skip this module when TensorFlow is unavailable (e.g., CPU-only CI)."""
    return pytest.importorskip("tensorflow")


@pytest.mark.parametrize("species", ["mouse", "rat"])
def test_initialise_network_produces_named_layers(tf_module, species):
    from DeepSlice.neural_network.neural_network import (
        DENSE_HEAD_LAYER_NAMES,
        XCEPTION_BASE_LAYER_NAME,
        initialise_network,
    )

    model = initialise_network(xception_weights=None, weights=None, species=species)

    for expected in (XCEPTION_BASE_LAYER_NAME, *DENSE_HEAD_LAYER_NAMES):
        model.get_layer(expected)


@pytest.mark.parametrize("species", ["mouse", "rat"])
def test_forward_pass_produces_9_vector(tf_module, species):
    import numpy as np

    from DeepSlice.neural_network.neural_network import initialise_network

    model = initialise_network(xception_weights=None, weights=None, species=species)
    dummy = np.zeros((1, 299, 299, 3), dtype=np.float32)
    output = model.predict(dummy, verbose=0)

    assert output.shape == (1, 9)
    assert np.isfinite(output).all()


def test_initialise_network_rejects_unknown_species(tf_module):
    from DeepSlice.neural_network.neural_network import initialise_network

    with pytest.raises(ValueError, match="species must be one of"):
        initialise_network(xception_weights=None, weights=None, species="hamster")


def test_load_xception_weights_rejects_unknown_species(tf_module):
    from DeepSlice.neural_network.neural_network import load_xception_weights

    with pytest.raises(ValueError, match="species must be one of"):
        load_xception_weights(model=None, weights="/tmp/ignored.h5", species="hamster")


def test_load_xception_weights_missing_dense_layer_raises(tf_module, tmp_path):
    """The loader must emit a clear error when the model lacks expected layers."""
    import h5py
    import numpy as np
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Input

    from DeepSlice.neural_network.neural_network import load_xception_weights

    inputs = Input(shape=(4,))
    outputs = Dense(2, name="unrelated")(inputs)
    bad_model = Model(inputs, outputs)

    weights_path = tmp_path / "weights.h5"
    with h5py.File(weights_path, "w") as file_handle:
        group = file_handle.create_group("dense/dense")
        group.create_dataset("kernel:0", data=np.zeros((2048, 256), dtype=np.float32))
        group.create_dataset("bias:0", data=np.zeros((256,), dtype=np.float32))

    with pytest.raises(RuntimeError, match="missing expected layer 'dense'"):
        load_xception_weights(bad_model, str(weights_path))


def test_xception_is_never_called_with_a_name_kwarg():
    """`keras.applications.xception.Xception` is a plain builder function with
    a fixed signature (include_top/weights/input_tensor/input_shape/pooling/
    classes/classifier_activation) — it accepts no `name`/`**kwargs` on the
    tensorflow>=2.13,<2.16 range this project pins in setup.py, and raises
    TypeError if one is passed. `d216d79` added exactly that kwarg while
    wiring up name-based weight resolution (DS-007), which made
    `initialise_network()` — the function BOTH species call for every single
    prediction run — crash unconditionally with
    ``TypeError: Xception() got an unexpected keyword argument 'name'``.

    This is a static AST check, deliberately independent of whether
    tensorflow is importable in the current environment (the two behavioural
    tests above are `pytest.importorskip`-gated and would silently skip in a
    TF-less environment, which is exactly how this regression could ship
    unnoticed) — it inspects the source directly rather than exercising the
    real call.
    """
    import ast
    import pathlib

    source_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "DeepSlice"
        / "neural_network"
        / "neural_network.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    xception_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Xception"
    ]
    assert xception_calls, "expected at least one Xception(...) call site to check"

    for call in xception_calls:
        passed_kwargs = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "name" not in passed_kwargs, (
            "Xception(...) was called with name=..., which the real "
            "keras.applications Xception() builder function does not "
            "accept and raises TypeError for (see test docstring)"
        )


def test_xception_base_layer_name_matches_the_real_default(tf_module):
    """`load_xception_weights` resolves the backbone via
    `model.get_layer(XCEPTION_BASE_LAYER_NAME)`, which only works because a
    freshly built `Xception(...)` model's own `.name` already defaults to
    "xception" — nothing in `initialise_network` sets it explicitly (see the
    test above). Pins that assumption directly, independent of the constant
    matching by coincidence.
    """
    from tensorflow.keras.applications.xception import Xception

    from DeepSlice.neural_network.neural_network import XCEPTION_BASE_LAYER_NAME

    model = Xception(include_top=True, weights=None)
    assert model.name == XCEPTION_BASE_LAYER_NAME
