"""download_file must never leave a truncated file at the destination."""
import hashlib
import os

import pytest

from DeepSlice.metadata import metadata_loader


class _Resp:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = chunks, fail_after
        self.headers = {"content-length": str(sum(map(len, chunks)))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("dropped")
            yield c


def test_success_writes_file_and_no_part(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_loader.requests, "get", lambda *a, **k: _Resp([b"ab", b"cd"]))
    dest = tmp_path / "w.bin"
    metadata_loader.download_file("u", str(dest), retries=1)
    assert dest.read_bytes() == b"abcd"
    assert os.listdir(tmp_path) == ["w.bin"]


def test_interrupted_download_keeps_previous_file(tmp_path, monkeypatch):
    dest = tmp_path / "w.bin"
    dest.write_bytes(b"old")
    monkeypatch.setattr(
        metadata_loader.requests, "get",
        lambda *a, **k: _Resp([b"ab", b"cd"], fail_after=1),
    )
    with pytest.raises(RuntimeError):
        metadata_loader.download_file("u", str(dest), retries=1)
    assert dest.read_bytes() == b"old"
    assert os.listdir(tmp_path) == ["w.bin"]


def test_checksum_mismatch_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_loader.requests, "get", lambda *a, **k: _Resp([b"abcd"]))
    dest = tmp_path / "w.bin"
    with pytest.raises(RuntimeError):
        metadata_loader.download_file("u", str(dest), retries=1, expected_sha256="0" * 64)
    assert os.listdir(tmp_path) == []


def test_checksum_match_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_loader.requests, "get", lambda *a, **k: _Resp([b"abcd"]))
    dest = tmp_path / "w.bin"
    metadata_loader.download_file("u", str(dest), retries=1,
                                  expected_sha256=hashlib.sha256(b"abcd").hexdigest())
    assert dest.read_bytes() == b"abcd"
