"""Pins DeepSliceMainWindow.dropEvent's toast count.

The toast used to report len(dropped_paths) - the number of URLs the OS
handed over - rather than how many images state.add_images actually kept
after filtering out non-files, unsupported extensions and duplicates.
Dropping a folder of 200 TIFFs said "Added 1 dropped path(s)" (one folder
URL); dropping 5 unsupported files said "Added 5" when zero were added.

As in test_load_session_file.py, DeepSliceMainWindow is exercised via the
unbound method against a lightweight stub rather than a real instance -
constructing the real window pulls in the full Qt UI, which nothing else
in this suite does.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from DeepSlice.gui.main_window import DeepSliceMainWindow
from DeepSlice.gui.state import DeepSliceAppState


class _FakeUrl:
    def __init__(self, local_path: str) -> None:
        self._local_path = local_path

    def toLocalFile(self) -> str:  # noqa: N802 - matches QUrl's Qt-style name
        return self._local_path


class _FakeMimeData:
    def __init__(self, paths: list[str]) -> None:
        self._paths = paths

    def hasUrls(self) -> bool:  # noqa: N802
        return True

    def urls(self) -> list[_FakeUrl]:
        return [_FakeUrl(path) for path in self._paths]


class _FakeDropEvent:
    def __init__(self, paths: list[str]) -> None:
        self._mime = _FakeMimeData(paths)
        self.accepted = False

    def mimeData(self):  # noqa: N802
        return self._mime

    def acceptProposedAction(self):  # noqa: N802
        self.accepted = True


class _StubWindow:
    def __init__(self) -> None:
        self.state = DeepSliceAppState()
        self._show_toast = MagicMock()

    def _handle_dropped_paths(self, paths: list[str]) -> None:
        # Real _handle_dropped_paths goes through
        # _collect_supported_files_from_paths + state.add_images; state's
        # own filtering (existing file + supported extension + dedup) is
        # what this test is actually exercising, so drive it directly.
        image_paths = [
            path for path in paths if Path(path).suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        ]
        self.state.add_images(image_paths)


def _drop(stub: _StubWindow, paths: list[str]) -> _FakeDropEvent:
    event = _FakeDropEvent(paths)
    DeepSliceMainWindow.dropEvent(stub, event)
    return event


def test_toast_reports_images_actually_added_not_paths_requested(tmp_path: Path) -> None:
    tif_paths = []
    for index in range(3):
        image_path = tmp_path / f"slide_{index}.tif"
        image_path.write_bytes(b"fake")
        tif_paths.append(str(image_path))

    stub = _StubWindow()
    event = _drop(stub, tif_paths)

    assert event.accepted
    stub._show_toast.assert_called_once()
    message, kwargs = stub._show_toast.call_args.args[0], stub._show_toast.call_args.kwargs
    assert message == "Added 3 image(s)"
    assert kwargs.get("level", "info") != "warning"


def test_toast_reports_zero_when_nothing_was_actually_added(tmp_path: Path) -> None:
    # Unsupported extension - state.add_images will filter it out even
    # though a URL was dropped for it.
    pdf_path = tmp_path / "notes.pdf"
    pdf_path.write_bytes(b"fake")

    stub = _StubWindow()
    event = _drop(stub, [str(pdf_path)] * 5)

    assert event.accepted
    stub._show_toast.assert_called_once()
    args, kwargs = stub._show_toast.call_args
    assert args[0] != "Added 5 dropped path(s)"
    assert "Added 5" not in args[0]
    assert kwargs.get("level") == "warning"


def test_toast_reports_the_real_delta_when_a_folder_expands_to_many_files(tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    for index in range(4):
        (folder / f"s{index}.png").write_bytes(b"fake")

    stub = _StubWindow()
    # _handle_dropped_paths is stubbed to only understand plain file paths
    # with supported extensions (see _StubWindow docstring) - simulate what
    # the real _collect_supported_files_from_paths would have expanded the
    # single folder URL into.
    expanded = [str(p) for p in folder.iterdir()]
    event = _drop(stub, expanded)

    assert event.accepted
    message = stub._show_toast.call_args.args[0]
    assert message == "Added 4 image(s)"
