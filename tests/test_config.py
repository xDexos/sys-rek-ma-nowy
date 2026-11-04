import pytest

from sys_rek_mag.config import Paths, model_id


def test_model_id():
    assert model_id("gemma4:e2b") == "gemma4_e2b"
    with pytest.raises(ValueError):
        model_id("  ")


def test_paths_layout(tmp_path):
    p = Paths(tmp_path)
    assert p.results("no_llm", "dense") == tmp_path / "results" / "no_llm" / "dense"
    assert p.models("llm_x") == tmp_path / "models" / "llm_x"
    assert p.item_texts("no_llm") == tmp_path / "item_texts" / "no_llm"
    assert p.comparisons == tmp_path / "results" / "comparisons"
    assert p.results_root == tmp_path / "results"
    assert p.raw == tmp_path / "raw" and p.prepared == tmp_path / "prepared"
