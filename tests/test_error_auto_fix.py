"""Tests for `DeepSlice.error_auto_fix.ErrorAutoFixer`.

This module previously had zero test coverage despite being wired to a
real, user-triggerable action ("Try Auto-Fix Last Error",
`gui/main_window.py`) that runs `pip install` for real via
`_install_and_verify`. The tests below cover its rule-based error
classification, and in particular pin the fix for a real bug: the
`tensorflow` entry in `MODULE_PACKAGE_MAP` used to hardcode
`"tensorflow<3.0"`, independent of and looser than `setup.py`'s actual
pin (`tensorflow>=2.13,<2.16`). Had a user's TensorFlow install ever gone
missing and they clicked "Try Auto-Fix", the auto-fixer would have `pip
install`ed a current TF 2.16+ release, landing on Keras 3 -- the exact
API-incompatibility class `setup.py`'s own comment says this project is
pinned against, and the same class of bug PR #17 fixed
(`initialise_network()` crashing against an unpinned Xception kwarg).
"""
from __future__ import annotations

import pathlib
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DeepSlice.error_auto_fix import MODULE_PACKAGE_MAP, ErrorAutoFixer


def _setup_py_tensorflow_pin() -> str:
    """Extract the exact tensorflow spec string from setup.py's install_requires."""
    setup_path = pathlib.Path(__file__).resolve().parents[1] / "setup.py"
    text = setup_path.read_text(encoding="utf-8")
    match = re.search(r'"(tensorflow[^"]*)"', text)
    assert match, "setup.py no longer declares a quoted tensorflow dependency spec"
    return match.group(1)


def test_tensorflow_install_spec_matches_setup_py_pin():
    """The auto-fixer must never install a TensorFlow outside the app's own pin.

    This is the regression pin for the bug this session fixed: before it,
    `MODULE_PACKAGE_MAP["tensorflow"]` was `"tensorflow<3.0"`, which does
    not match `setup.py`'s real `tensorflow>=2.13,<2.16` pin and would
    resolve to a Keras-3-era TensorFlow release today.
    """
    setup_pin = _setup_py_tensorflow_pin()
    package_spec, import_name = MODULE_PACKAGE_MAP["tensorflow"]

    assert package_spec == setup_pin
    assert import_name == "tensorflow"
    # The specific incompatibility this guards against: an unbounded-above
    # (or too-loosely-bounded) spec that would happily resolve past 2.16.
    assert "<2.16" in package_spec


@pytest.mark.parametrize(
    "module_key",
    ["matplotlib", "pyside6", "numpy", "pandas", "requests", "pil", "skimage",
     "scipy", "h5py", "nibabel", "reportlab", "lxml"],
)
def test_other_module_specs_have_no_upper_bound(module_key):
    """Sanity check the other entries: none of these are upper-bounded in
    setup.py, so a bare package name (which `pip install` resolves to
    "latest compatible") is the correct spec for them -- unlike
    tensorflow, which genuinely needs an explicit upper bound."""
    package_spec, _import_name = MODULE_PACKAGE_MAP[module_key]
    assert "<" not in package_spec


class TestExtractMissingModule:
    def test_extracts_module_name_from_standard_message(self):
        fixer = ErrorAutoFixer()
        result = fixer._extract_missing_module("No module named 'tensorflow'")
        assert result == "tensorflow"

    def test_extracts_dotted_submodule_root_is_kept_intact(self):
        fixer = ErrorAutoFixer()
        # _extract_missing_module itself returns the raw dotted name; root
        # resolution happens in _resolve_install_target.
        result = fixer._extract_missing_module("No module named 'skimage.morphology'")
        assert result == "skimage.morphology"

    def test_returns_none_when_no_match(self):
        fixer = ErrorAutoFixer()
        assert fixer._extract_missing_module("ValueError: bad shape") is None

    def test_rejects_shell_metacharacters(self):
        """A module name containing shell metacharacters must never reach a
        generated `pip install` command string."""
        fixer = ErrorAutoFixer()
        result = fixer._extract_missing_module(
            "No module named 'foo; rm -rf ~'"
        )
        assert result is None


class TestResolveInstallTarget:
    def test_resolves_known_root_module(self):
        fixer = ErrorAutoFixer()
        assert fixer._resolve_install_target("tensorflow") == (
            "tensorflow>=2.13,<2.16",
            "tensorflow",
        )

    def test_resolves_dotted_submodule_by_root(self):
        fixer = ErrorAutoFixer()
        assert fixer._resolve_install_target("skimage.morphology") == (
            "scikit-image",
            "skimage",
        )

    def test_is_case_insensitive(self):
        fixer = ErrorAutoFixer()
        assert fixer._resolve_install_target("PySide6") == ("PySide6", "PySide6")

    def test_unknown_module_returns_none(self):
        fixer = ErrorAutoFixer()
        assert fixer._resolve_install_target("some_random_unmapped_module") is None


class TestAnalyzeError:
    def test_missing_dependency_analysis_names_the_pinned_tensorflow_spec(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("prediction", "No module named 'tensorflow'")

        assert analysis["category"] == "missing_dependency"
        assert analysis["auto_fix_available"] is True
        assert "tensorflow>=2.13,<2.16" in analysis["auto_fix_plan"]
        assert "tensorflow<3.0" not in analysis["auto_fix_plan"]

    def test_missing_dependency_with_no_safe_mapping(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("prediction", "No module named 'some_obscure_pkg'")

        assert analysis["category"] == "missing_dependency"
        assert analysis["auto_fix_available"] is False

    def test_workflow_precondition_detected(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("curation", "RuntimeError: no predictions available")

        assert analysis["category"] == "workflow_precondition"
        assert analysis["auto_fix_available"] is False

    def test_filename_validation_detected(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error(
            "ingest", "ValueError: no section number found in filename slide1.tif"
        )

        assert analysis["category"] == "filename_validation"
        assert analysis["auto_fix_available"] is False

    def test_permission_error_detected(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("export", "PermissionError: [Errno 13] Permission denied")

        assert analysis["category"] == "permission_error"
        assert analysis["auto_fix_available"] is False

    def test_unknown_error_falls_back(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("misc", "Something totally unrecognized happened")

        assert analysis["category"] == "unknown"
        assert analysis["auto_fix_available"] is False


class TestFormatAnalysis:
    def test_includes_summary_category_and_plan(self):
        fixer = ErrorAutoFixer()
        analysis = fixer.analyze_error("prediction", "No module named 'tensorflow'")
        text = fixer.format_analysis(analysis)

        assert "Summary:" in text
        assert "Category: missing_dependency" in text
        assert "Auto-fix available: yes" in text
        assert "tensorflow>=2.13,<2.16" in text


class TestTryAutoFix:
    def test_returns_not_attempted_when_no_safe_fix(self):
        fixer = ErrorAutoFixer()
        result = fixer.try_auto_fix("misc", "Something totally unrecognized happened")

        assert result["attempted"] is False
        assert result["succeeded"] is False

    def test_returns_not_attempted_for_non_missing_dependency_category(self):
        fixer = ErrorAutoFixer()
        # workflow_precondition is auto_fix_available=False already, but
        # confirm the category gate explicitly reads as documented for a
        # category that isn't "missing_dependency".
        result = fixer.try_auto_fix("curation", "RuntimeError: no predictions available")

        assert result["attempted"] is False
        assert "Auto-fix currently supports missing dependency errors only" in result["summary"] \
            or "No safe automatic fix" in result["summary"]

    def test_missing_dependency_with_no_mapping_is_not_attempted(self):
        fixer = ErrorAutoFixer()
        result = fixer.try_auto_fix("prediction", "No module named 'some_obscure_pkg'")

        assert result["attempted"] is False
        assert result["succeeded"] is False

    def test_missing_tensorflow_drives_pip_install_with_the_pinned_spec(self):
        """End-to-end wiring check (subprocess mocked, nothing is actually
        installed): confirms the exact string handed to `pip install` for a
        real 'No module named tensorflow' error is the pinned spec, not the
        old unbounded one."""
        fixer = ErrorAutoFixer()

        install_result = MagicMock(returncode=0, stdout="Successfully installed", stderr="")
        verify_result = MagicMock(returncode=0, stdout="", stderr="")

        with patch(
            "DeepSlice.error_auto_fix.subprocess.run",
            side_effect=[install_result, verify_result],
        ) as mock_run:
            result = fixer.try_auto_fix("prediction", "No module named 'tensorflow'")

        assert result["attempted"] is True
        assert result["succeeded"] is True

        install_call_args = mock_run.call_args_list[0].args[0]
        assert install_call_args[-1] == "tensorflow>=2.13,<2.16"
        assert "tensorflow<3.0" not in install_call_args

    def test_failed_pip_install_is_reported_not_attempted_as_success(self):
        fixer = ErrorAutoFixer()

        install_result = MagicMock(returncode=1, stdout="", stderr="ERROR: no matching distribution")

        with patch(
            "DeepSlice.error_auto_fix.subprocess.run",
            return_value=install_result,
        ):
            result = fixer.try_auto_fix("prediction", "No module named 'tensorflow'")

        assert result["attempted"] is True
        assert result["succeeded"] is False
        assert "could not install" in result["summary"]

    def test_install_command_exception_is_reported_not_raised(self):
        fixer = ErrorAutoFixer()

        with patch(
            "DeepSlice.error_auto_fix.subprocess.run",
            side_effect=OSError("pip not found"),
        ):
            result = fixer.try_auto_fix("prediction", "No module named 'tensorflow'")

        assert result["attempted"] is True
        assert result["succeeded"] is False
        assert "pip not found" in result["details"]
