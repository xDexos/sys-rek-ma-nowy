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


def _run_after_llm(paths):
    """encode → train → evaluate tak jak CLI: warianty z current_variants (nie ze skanu folderów)."""
    from sys_rek_mag.encode import current_variants

    encode_all(paths, FakeEncoder())
    for variant in current_variants(paths):
        train_variant(paths, variant)
    return evaluate(paths)


def _full_run(paths, monkeypatch):
    monkeypatch.setattr(llm_texts, "server_info", lambda *a: {"server": "fake"})
    prepare(paths, **PREPARE_SIZES)
    generate("fake:model", paths, client=FakeClient(responder=responder))
    assert _run_after_llm(paths)["variants"] == [NO_LLM, LLM, BASELINE]


def test_prompt_change_with_pilot_drops_stale_llm_variant(paths, raw, monkeypatch, capsys):
    _full_run(paths, monkeypatch)
    monkeypatch.setattr(llm_texts, "PROMPT_HASH", "new-prompt")
    generate("fake:model", paths, client=FakeClient(responder=responder), limit=2)
    capsys.readouterr()
    assert _run_after_llm(paths)["variants"] == [NO_LLM, BASELINE]  # stare models/llm_* nie są oceniane
    assert f"UWAGA: pomijam {LLM}" in capsys.readouterr().out


def test_reprepare_with_changed_inputs_drops_stale_generations(paths, raw, monkeypatch):
    from sys_rek_mag import config
    from sys_rek_mag.encode import complete_llm_variants

    _full_run(paths, monkeypatch)
    monkeypatch.setattr(config, "REVIEW_MAX_CHARS", 10)  # te same produkty, inne wejścia LLM
    prepare(paths, **PREPARE_SIZES)
    assert complete_llm_variants(paths) == [] and llm_texts.current_generations(paths, LLM) == {}
    assert _run_after_llm(paths)["variants"] == [NO_LLM, BASELINE]


def test_current_variant_with_stale_models_fails_clearly(paths, raw, monkeypatch):
    import pytest

    from sys_rek_mag.encode import current_variants

    _full_run(paths, monkeypatch)
    monkeypatch.setattr(llm_texts, "PROMPT_HASH", "new-prompt")
    other = lambda messages: responder(messages).replace("customers like", "drivers praise")  # noqa: E731
    generate("fake:model", paths, client=FakeClient(responder=other))  # komplet nowych tekstów, bez encode
    with pytest.raises(ValueError, match="sysrekmag-encode"):
        current_variants(paths)
    with pytest.raises(ValueError, match="sysrekmag-encode"):
        evaluate(paths)
