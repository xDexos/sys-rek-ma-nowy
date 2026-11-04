import pytest

from sys_rek_mag.config import Paths
from sys_rek_mag.download import download


def test_download_skips_existing(tmp_path):
    paths = Paths(tmp_path / "data")
    paths.raw.mkdir(parents=True)
    (paths.raw / "a.gz").write_bytes(b"old")
    calls = []
    result = download(paths, {"a.gz": "u"}, lambda url, target: calls.append(url))
    assert calls == [] and (paths.raw / "a.gz").read_bytes() == b"old"
    assert result == [paths.raw / "a.gz"]


def test_download_fetches_missing(tmp_path):
    paths = Paths(tmp_path / "data")
    result = download(paths, {"a.gz": "u"}, lambda url, target: target.write_bytes(b"new"))
    assert result == [paths.raw / "a.gz"] and (paths.raw / "a.gz").read_bytes() == b"new"


def test_download_atomic_on_failure(tmp_path):
    paths = Paths(tmp_path / "data")

    def broken(url, target):
        target.write_bytes(b"half")
        raise OSError("przerwane")

    with pytest.raises(OSError):
        download(paths, {"a.gz": "u"}, broken)
    assert list(paths.raw.iterdir()) == []
