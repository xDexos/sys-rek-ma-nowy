from sys_rek_mag.storage import read_json, read_pickle, write_json, write_pickle


def test_write_pickle_roundtrip_atomic(tmp_path):
    path = tmp_path / "sub" / "x.pkl"
    write_pickle({"a": [1, 2]}, path)
    assert read_pickle(path) == {"a": [1, 2]}
    assert [p.name for p in path.parent.iterdir()] == ["x.pkl"]


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "x.json"
    write_json({"zażółć": 1}, path)
    assert read_json(path) == {"zażółć": 1}
    assert not (tmp_path / ".x.json.tmp").exists()
