import sys
from unittest.mock import patch


def test_state_and_neural_network_import_without_tensorflow():
    """Ensure DeepSlice.gui.state and DeepSlice.neural_network.neural_network
    can be imported safely even in environments where TensorFlow is not installed."""
    with patch.dict(sys.modules, {"tensorflow": None, "tensorflow.keras": None, "tensorflow.keras.models": None}):
        import DeepSlice.neural_network.neural_network as nn
        import DeepSlice.gui.state as state
        import DeepSlice.training.train_runner as tr

        assert hasattr(nn, "XCEPTION_INPUT_SIZE")
        assert tr.XCEPTION_INPUT_SIZE == (299, 299, 3)
        assert hasattr(state, "DeepSliceAppState")
