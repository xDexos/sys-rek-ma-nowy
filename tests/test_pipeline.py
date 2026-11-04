import json
import re

import pandas as pd

from conftest import GROUP_WORDS, PREPARE_SIZES, FakeClient, FakeEncoder
from sys_rek_mag import llm_texts
from sys_rek_mag.config import BASELINE, MECHANISMS, NO_LLM, POPULARITY, llm_variant
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import encode_all
from sys_rek_mag.evaluate import evaluate
from sys_rek_mag.llm_texts import generate
from sys_rek_mag.storage import read_json
from sys_rek_mag.two_tower import train_variant

LLM = llm_variant("fake_model")


def responder(messages):
    """Opis z prawdziwego słowa grupy, które występuje w opiniach produktu."""
    review_text = messages[1]["content"].lower()
    word = max(GROUP_WORDS, key=lambda w: len(re.findall(w, review_text)))
    description = f"This {word} is a reliable {word} that customers like. " + " ".join([word] * 10)
    features = [{"aspect": f"{word} aspect {i}", "value": "good"} for i in range(6)]
    return json.dumps({"description": description, "features": features})


def test_full_pipeline(paths, raw, monkeypatch):
    monkeypatch.setattr(llm_texts, "server_info", lambda *a: {"server": "fake"})
    prepare(paths, **PREPARE_SIZES)
    generate("fake:model", paths, client=FakeClient(responder=responder))
    assert set(encode_all(paths, FakeEncoder())) == {NO_LLM, LLM}
    for variant in (NO_LLM, LLM):
        train_variant(paths, variant)

    evaluate(paths)
    first = pd.read_csv(paths.comparisons / "comparisons.csv")
    for variant, mechanism in [*((v, m) for v in (NO_LLM, LLM) for m in MECHANISMS), (BASELINE, POPULARITY)]:
        assert (paths.results(variant, mechanism) / "metrics.json").is_file()
    ndcg = {k: read_json(paths.results(*k) / "metrics.json")["ndcg_at_10"]
            for k in ((NO_LLM, "dense"), (BASELINE, POPULARITY))}
    assert ndcg[(NO_LLM, "dense")] > ndcg[(BASELINE, POPULARITY)]
    assert set(first["role"]) == {"primary", "supporting"} and (first["role"] == "primary").sum() == 3

    evaluate(paths)
    pd.testing.assert_frame_equal(first, pd.read_csv(paths.comparisons / "comparisons.csv"))
