"""Dashboard: odkrywanie wyników, profil tekstowy bez LLM i start aplikacji (AppTest) na danych z potoku testowego."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import sys_rek_mag.dashboard as dashboard
from sys_rek_mag import charts
from conftest import PREPARE_SIZES, FakeEncoder, write_raw
from sys_rek_mag import llm_texts
from sys_rek_mag.config import NO_LLM, Paths, llm_variant
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import encode_all
from sys_rek_mag.evaluate import evaluate, evaluate_valid
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
    evaluate_valid(paths)
    return paths


class Boom:
    def encode(self, *a, **k):
        raise AssertionError("encoder nie powinien być wołany")


def test_discover_results_from_folders(paths):
    found = dashboard.discover_results(paths)
    assert {"variant", "mechanism", "users", "ndcg_at_10"} <= set(found.columns)
    assert set(zip(found["variant"], found["mechanism"])) >= {(NO_LLM, "dense"), (LLM, "two_tower"), ("baseline", "popularity")}
    unknown = paths.results(NO_LLM, "new_mech")
    shutil.copytree(paths.results(NO_LLM, "dense"), unknown)
    try:
        assert "new_mech" in dashboard.mechanism_options(dashboard.discover_results(paths))
        assert "popularity" not in dashboard.mechanism_options(dashboard.discover_results(paths))
    finally:
        shutil.rmtree(unknown)


def test_discover_valid_results(paths):
    found = dashboard.discover_results(paths, "valid")
    assert set(zip(found["variant"], found["mechanism"])) >= {(NO_LLM, "dense"), (LLM, "two_tower"), ("baseline", "popularity")}
    assert dashboard.comparisons_path(paths, "valid") == paths.results_valid / "comparisons.csv"
    assert dashboard.discover_results(Paths(paths.root / "brak"), "valid").empty


def test_stale_result_folder_hidden(paths):
    stale = paths.results("llm_old", "dense")  # z poprzedniego przebiegu, nie ma go w manifeście porównań
    shutil.copytree(paths.results(LLM, "dense"), stale)
    try:
        assert "llm_old" not in set(dashboard.discover_results(paths)["variant"])
    finally:
        shutil.rmtree(stale.parent)


def test_custom_top_n_rejects_rows_not_matching_items(paths, monkeypatch):
    data = dashboard.load_variant(paths, NO_LLM)
    monkeypatch.setattr(dashboard, "load_variant", lambda *a: data | {"embeddings": data["embeddings"][:-1]})
    with pytest.raises(ValueError, match="sysrekmag-encode"):
        dashboard.custom_top_n(paths, NO_LLM, "dense", "led bulb", 5, FakeEncoder())


def test_variant_colors_stable():
    one = charts.variant_colors([NO_LLM, "baseline", "llm_b"])
    two = charts.variant_colors([NO_LLM, "baseline", "llm_b", "llm_a"])
    assert one[NO_LLM] == two[NO_LLM] and one["baseline"] == two["baseline"]
    assert len(set(two.values())) == 4
    assert charts.variant_colors([NO_LLM, "llm_b", "llm_b", "llm_b"])["llm_b"] == one["llm_b"]  # powtórzone wiersze


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
    app.sidebar.selectbox[0].set_value("valid").run()
    assert not app.exception and len(app.dataframe) >= 2


def test_app_without_results(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "PATHS", Paths(tmp_path / "data"))
    app = AppTest.from_file(LAUNCHER, default_timeout=60).run()
    assert not app.exception and any("sysrekmag-evaluate" in w.value for w in app.warning)


def test_app_defaults_to_valid_when_only_valid_results(paths, tmp_path, monkeypatch):
    only_valid = Paths(tmp_path / "data")
    shutil.copytree(paths.results_valid, only_valid.results_valid)
    monkeypatch.setattr(dashboard, "PATHS", only_valid)
    app = AppTest.from_file(LAUNCHER, default_timeout=60).run()
    assert not app.exception and not app.warning and app.sidebar.selectbox[0].value == "valid"


def test_selected_rows_order_and_labels(paths):
    rows = dashboard.selected_rows(dashboard.discover_results(paths), LLM, "dense")
    assert list(zip(rows["variant"], rows["mechanism"])) == [(NO_LLM, "dense"), (LLM, "dense"), ("baseline", "popularity")]
    assert rows["wariant"].tolist() == [dashboard.label(NO_LLM), dashboard.label(LLM), dashboard.label("baseline")]
    assert dashboard.selected_rows(dashboard.discover_results(paths), None, "dense")["variant"].tolist() == [NO_LLM, "baseline"]


def test_mechanism_rows_fixed_order(paths):
    rows = dashboard.mechanism_rows(dashboard.discover_results(paths), LLM)
    assert list(zip(rows["mechanism"], rows["variant"]))[:2] == [("tfidf", NO_LLM), ("tfidf", LLM)]
    assert "popularity" not in set(rows["mechanism"])


def test_results_table_joins_comparisons(paths):
    found = dashboard.discover_results(paths)
    rows = dashboard.selected_rows(found, LLM, "dense")
    cmp = dashboard.load_comparisons(paths, "test")
    cmp = cmp[(cmp["variant"] == LLM) & (cmp["mechanism"] == "dense")]
    table = dashboard.results_table(rows, cmp)
    primary = cmp[cmp["metric"] == "ndcg_at_10"].iloc[0]
    assert table.loc["NDCG@10", "Zmiana [%]"] == pytest.approx(primary["delta_relative"] * 100)
    assert table.loc["NDCG@10", "Wniosek"] == primary["conclusion"]
    assert table.loc["NDCG@10", dashboard.label(NO_LLM)] == pytest.approx(primary["baseline_mean"])
    assert "Wniosek" not in dashboard.results_table(rows, cmp.iloc[0:0])  # bez porównań: same metryki


def test_charts_build_valid_specs(paths):
    found = dashboard.discover_results(paths)
    rows = dashboard.selected_rows(found, LLM, "dense")
    cmp = dashboard.load_comparisons(paths, "test")
    cmp = cmp[(cmp["variant"] == LLM) & (cmp["mechanism"] == "dense")]
    for chart in (charts.metric_bars(rows, ["ndcg_at_10", "mrr_at_10"]), charts.delta_forest(cmp),
                  charts.mechanism_bars(dashboard.mechanism_rows(found, LLM), "ndcg_at_10", {}, 0.01)):
        spec = chart.to_dict()  # walidacja schematu Vega-Lite
        assert spec["config"]["locale"]["number"]["decimal"] == ","
    assert charts.metric_label("hit_rate_at_5") == "HitRate@5"
