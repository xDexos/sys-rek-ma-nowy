"""Dashboard: odkrywanie wyników, profil tekstowy bez LLM i start aplikacji (AppTest) na danych z potoku testowego."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import sys_rek_mag.dashboard as dashboard
from conftest import PREPARE_SIZES, FakeEncoder, write_raw
from sys_rek_mag import llm_texts
from sys_rek_mag.config import NO_LLM, Paths, llm_variant
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import encode_all
from sys_rek_mag.evaluate import evaluate
from sys_rek_mag.llm_texts import generate
from sys_rek_mag.two_tower import train_variant
from test_pipeline import LLM, responder
from conftest import FakeClient

LAUNCHER = str(Path(__file__).resolve().parents[1] / "dashboard.py")


@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> Paths:
    root = tmp_path_factory.mktemp("dash")
    paths = Paths(root / "data")
    write_raw(paths.raw)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llm_texts, "server_info", lambda *a: {"server": "fake"})
        prepare(paths, **PREPARE_SIZES)
        generate("fake:model", paths, client=FakeClient(responder=responder))
    encode_all(paths, FakeEncoder())
    for variant in (NO_LLM, LLM):
        train_variant(paths, variant)
    evaluate(paths)
    return paths


class Boom:
    def encode(self, *a, **k):
        raise AssertionError("encoder nie powinien być wołany")


def test_discover_results_from_folders(paths):
    found = dashboard.discover_results(paths)
    assert {"variant", "mechanism", "users", "ndcg_at_10"} <= set(found.columns)
    assert set(zip(found["variant"], found["mechanism"])) >= {(NO_LLM, "dense"), (LLM, "two_tower"), ("baseline", "popularity")}
    extra = paths.results("llm_other", "dense")
    shutil.copytree(paths.results(LLM, "dense"), extra)
    try:
        assert ("llm_other", "dense") in set(zip(*[dashboard.discover_results(paths)[c] for c in ("variant", "mechanism")]))
    finally:
        shutil.rmtree(paths.results("llm_other", "dense").parent)


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_custom_profile_returns_nothing(paths, text):
    assert dashboard.custom_top_n(paths, NO_LLM, "dense", text, 5, Boom()).empty


@pytest.mark.parametrize("variant", [NO_LLM, LLM])
@pytest.mark.parametrize("mechanism", ["tfidf", "dense", "two_tower"])
def test_custom_top_n_relevant_group(paths, variant, mechanism):
    top = dashboard.custom_top_n(paths, variant, mechanism, "led bulb bright", 5, FakeEncoder())
    assert list(top.columns) == ["parent_asin", "title", "score"] and len(top) == 5
    if mechanism != "two_tower":  # sieć dwuwieżowa na mini-danych bywa przypadkowa
        assert top["title"].str.contains("Led Bulb").sum() >= 4


def test_llm_cost(paths):
    cost = dashboard.llm_cost(paths)
    assert cost["wariant"].tolist() == [LLM] and cost["średni czas [s]"].iloc[0] >= 0


def test_app_starts_and_profile_tab(paths, monkeypatch):
    monkeypatch.setattr(dashboard, "PATHS", paths)
    monkeypatch.setattr(dashboard, "load_encoder", lambda: FakeEncoder())
    dashboard._encoder.clear()
    app = AppTest.from_file(LAUNCHER, default_timeout=60).run()
    assert not app.exception and len(app.tabs) == 4
    assert len(app.dataframe) >= 2
    app.text_area[0].set_value("led bulb bright").run()
    assert not app.exception
    assert len(app.expander) >= 1


def test_app_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "PATHS", Paths(tmp_path / "data"))
    app = AppTest.from_file(LAUNCHER, default_timeout=60).run()
    assert not app.exception and any("sysrekmag-evaluate" in w.value for w in app.warning)
