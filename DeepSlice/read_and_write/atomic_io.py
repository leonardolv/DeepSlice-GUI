"""Crash-safe file writers shared by the GUI and the export helpers."""
from __future__ import annotations

import json
import os
from collections.abc import Callable


def write_json_atomic(path: str, payload: object, **dump_kwargs) -> None:
    """Write ``payload`` as JSON to ``path`` without ever truncating it first.

    The payload is serialised in memory and written to a sibling temp file,
    then moved over ``path`` with ``os.replace``. A serialisation or I/O
    failure therefore leaves any previous file at ``path`` untouched.
    """
    text = json.dumps(payload, **dump_kwargs)
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_file_atomic(path: str, writer: Callable[[str], None]) -> None:
    """Run ``writer(tmp_path)`` and move the result over ``path``.

    Any previous file at ``path`` survives a failing ``writer``.
    """
    directory = os.path.dirname(os.path.abspath(path))
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
