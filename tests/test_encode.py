import json

import numpy as np
import pandas as pd
import pytest

from conftest import PREPARE_SIZES, FakeClient, FakeEncoder
from sys_rek_mag import llm_texts
from sys_rek_mag.config import llm_variant
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import complete_llm_variants, encode_all, load_variant, variant_texts
from sys_rek_mag.llm_texts import generate
from sys_rek_mag.storage import read_json

DESC = " ".join(["word"] * 25)
VALID = json.dumps({"description": DESC, "features": [{"aspect": f"a{i}", "value": f"v{i}"} for i in range(6)]})
LLM = llm_variant("fake_model")


@pytest.fixture
def prepared(paths, raw, monkeypatch):
    monkeypatch.setattr(llm_texts, "server_info", lambda *a: {"server": "fake"})
    prepare(paths, **PREPARE_SIZES)
    generate("fake:model", paths, client=FakeClient([VALID]))
    return paths


def _items(paths):
    return pd.read_parquet(paths.prepared / "items.parquet")


def test_encode_all_writes_variants(prepared):
    manifest = encode_all(prepared, FakeEncoder())
    n = len(_items(prepared))
    assert set(manifest) == {"no_llm", LLM}
    for v in manifest:
        assert len(pd.read_parquet(prepared.item_texts(v) / "texts.parquet")) == n
        data = load_variant(prepared, v)
        assert data["embeddings"].shape == (n, 32) and data["embeddings"].dtype == np.float32
        assert np.allclose(np.linalg.norm(data["embeddings"], axis=1), 1, atol=1e-5)
        assert data["tfidf"].shape[0] == n and data["vectorizer"] is not None
        info = read_json(prepared.models(v) / "manifest.json")
        assert info["rows"] == n and "tokens_max" in info
    assert "Features:" in pd.read_parquet(prepared.item_texts(LLM) / "texts.parquet")["text"][0]


def test_cache_skips_unchanged(prepared):
    encode_all(prepared, FakeEncoder())
    enc = FakeEncoder()
    encode_all(prepared, enc)
    assert enc.calls == 0


def test_incomplete_llm_variant_skipped(prepared, monkeypatch):
    other = prepared.item_texts(llm_variant("partial"))
    other.mkdir(parents=True)
    (other / "manifest.json").write_text('{"complete": false}')
    assert complete_llm_variants(prepared) == [LLM]
    assert set(variant_texts(prepared, _items(prepared))) == {"no_llm", LLM}


def test_row_order_matches_items(prepared):
    items = _items(prepared)
    texts = variant_texts(prepared, items)
    enc = FakeEncoder()
    encode_all(prepared, enc)
    emb = load_variant(prepared, "no_llm")["embeddings"]
    assert np.allclose(emb, enc.encode(texts["no_llm"]))
    assert list(pd.read_parquet(prepared.item_texts("no_llm") / "texts.parquet")["parent_asin"]) == list(items["parent_asin"])
