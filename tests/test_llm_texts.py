import json

import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import PREPARE_SIZES, FakeClient
from sys_rek_mag import llm_texts
from sys_rek_mag.config import llm_variant
from sys_rek_mag.dataset import prepare
from sys_rek_mag.llm_texts import ItemDescription, generate, load_generations
from sys_rek_mag.storage import read_jsonl, read_json
from sys_rek_mag.texts import llm_input_text

DESC = " ".join(["word"] * 25)
FEATURES = [{"aspect": f" Aspect{i} ", "value": f"Value {i}"} for i in range(6)]
VALID = json.dumps({"description": DESC, "features": FEATURES})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(llm_texts, "server_info", lambda *a: {"server": "fake"})


@pytest.fixture
def prepared(paths, raw):
    prepare(paths, **PREPARE_SIZES)
    return paths


def _out(paths):
    return paths.item_texts(llm_variant("fake_model"))


def test_labels_normalized_and_counts_validated():
    parsed = ItemDescription.model_validate_json(VALID)
    assert parsed.features[0].aspect == "aspect0" and parsed.features[0].value == "value 0"
    with pytest.raises(ValidationError):
        ItemDescription.model_validate({"description": DESC, "features": FEATURES[:4]})
    with pytest.raises(ValidationError):
        ItemDescription.model_validate({"description": "too short", "features": FEATURES})
    with pytest.raises(ValidationError):
        ItemDescription.model_validate({"description": DESC, "features": FEATURES * 3})


def test_feature_count_validated_retries(prepared):
    bad = json.dumps({"description": DESC, "features": FEATURES[:2]})
    client = FakeClient([bad, VALID])
    generate("fake:model", prepared, client=client, limit=1)
    assert client.calls == 2
    assert read_jsonl(_out(prepared) / "generations.jsonl")[0]["attempts"] == 2


def test_generate_writes_and_caches(prepared):
    client = FakeClient([VALID])
    manifest = generate("fake:model", prepared, client=client)
    n = len(pd.read_parquet(prepared.prepared / "items.parquet"))
    assert client.calls == n and manifest["complete"] is True
    gens = load_generations(prepared, llm_variant("fake_model"))
    assert len(gens) == n
    first = next(iter(gens.values()))
    assert first["description"] == DESC
    again = FakeClient([VALID])
    generate("fake:model", prepared, client=again)
    assert again.calls == 0


def test_invalid_json_retries_and_errors_resume(prepared):
    client = FakeClient(["not json"])
    with pytest.raises(SystemExit):
        generate("fake:model", prepared, client=client, limit=2)
    assert client.calls == 6  # 2 produkty x 3 próby
    assert len(read_jsonl(_out(prepared) / "errors.jsonl")) == 2
    assert read_json(_out(prepared) / "manifest.json")["complete"] is False
    manifest = generate("fake:model", prepared, client=FakeClient([VALID]))
    assert manifest["complete"] is True
    assert not (_out(prepared) / "errors.jsonl").exists()


def test_truncated_response_retried(prepared):
    client = FakeClient([VALID], finish_reason="length")
    with pytest.raises(SystemExit):
        generate("fake:model", prepared, client=client, limit=1)
    assert client.calls == 3


def test_input_never_contains_product_title(prepared):
    seen = []
    def responder(messages):
        seen.append(messages[1]["content"])
        return VALID
    generate("fake:model", prepared, client=FakeClient(responder=responder), limit=3)
    items = pd.read_parquet(prepared.prepared / "items.parquet")
    assert seen and all(t not in s for s in seen for t in items["title"])
