# GUI & UX Maintenance Log

## In Progress

## Blocked / Needs Review

## Completed

### Task: Complete named-layer resolution refactor for weight loader (DS-007)
- **Completed**: 2026-09-07 20:25:00
- **Changes**:
  - `DeepSlice/neural_network/neural_network.py`:
    - Exported `XCEPTION_BASE_LAYER_NAME = "xception"` and `DENSE_HEAD_LAYER_NAMES = ("dense", "dense_1", "dense_2")`.
    - Updated `initialise_network`: Added species validation `("mouse", "rat")`, assigned explicit `name=XCEPTION_BASE_LAYER_NAME` to `Xception(...)` and explicit names `DENSE_HEAD_LAYER_NAMES[0..2]` to the 3 dense head layers in both mouse and rat architectures.
    - Updated `load_xception_weights`: Validated species at entry, resolved dense layers by name via `model.get_layer(layer_name)` instead of fragile positional indices, supported both `"kernel"` / `"kernel:0"` and `"bias"` / `"bias:0"` group keys, and raised clear `RuntimeError("missing expected layer '<name>'")` when layers are missing.
    - Preserved complete Xception layer weight mapping and sub-layer verification.
- **Verification**: Verified with `pytest tests/test_weight_loader.py -v` (7 passed in 35.87s) and test suite `pytest tests/ -k "not training"` (172 passed, 0 failures in 53.77s).
- **Status**: Completed

### Task: Fix Series index alignment and shape broadcast crash in calculate_average_section_thickness
- **Completed**: 2026-09-07 14:17:00
- **Changes**:
  - `DeepSlice/coord_post_processing/spacing_and_indexing.py`:
    - Converted `section_numbers` and `section_depth` via `np.asarray(...)` at function entry in `calculate_average_section_thickness`.
    - Resolved critical pandas Series index alignment bug where slice subtraction `section_depth[:-1] - section_depth[1:]` produced length N Series with NaNs and zeros instead of length N-1 consecutive differences, causing broadcast shape mismatches `(N,) / (N-1,)` and crashing section thickness estimation.
    - Resolved `.values` assumption on `section_numbers` which raised `AttributeError` on list inputs.
    - Handled boolean masking for `bad_sections` on ndarrays safely.
    - Enforced float return type for `average_thickness`.
  - `tests/test_spacing_and_indexing.py`:
    - Updated `test_calculate_average_section_thickness_happy_path_no_inf` assertion to expected positive thickness `10.0`.
    - Added `test_calculate_average_section_thickness_supports_python_lists_and_arrays` validating both raw Python lists and NumPy arrays without errors.
    - Added `test_calculate_average_section_thickness_with_bad_sections_filtering` verifying bad section exclusion across Series/lists.
- **Verification**: Verified with `pytest tests/test_spacing_and_indexing.py -v` (8 passed in 6.40s) and full test suites `pytest tests/test_curation_state.py tests/test_session_roundtrip.py tests/test_curation_state_rat.py tests/test_quicknii_roundtrip.py tests/test_training_utils.py -v` (40 passed, 0 failures).
- **Status**: Completed

## Backlog
