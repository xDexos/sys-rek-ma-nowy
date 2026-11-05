import numpy as np
import pandas as pd
import pytest

from conftest import PREPARE_SIZES, FakeEncoder, write_raw
from sys_rek_mag import evaluate
from sys_rek_mag.config import BASELINE, MECHANISMS, NO_LLM, POPULARITY, T1, T2, Paths
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import encode_all
from sys_rek_mag.storage import read_json
from sys_rek_mag.two_tower import train_variant


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    paths = Paths(tmp_path_factory.mktemp("ev") / "data")
    write_raw(paths.raw)
    prepare(paths, **PREPARE_SIZES)
    encode_all(paths, FakeEncoder())
    train_variant(paths, NO_LLM)
    return paths


def test_valid_period_items_masked():
    items = pd.DataFrame({"parent_asin": ["A", "B", "C", "D"]})
    rows = [("u", "A", 5.0, T1 - 1, "train"), ("u", "B", 5.0, T1, "valid"), ("u", "C", 5.0, T2, "test"),
            ("u", "D", 2.0, T1 - 5, "train"), ("v", "A", 5.0, T1 - 2, "train"), ("w", "C", 5.0, T2 + 1, "test"),
            ("u", "Z", 5.0, T2, "test")]
    interactions = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp", "split"])
    users = pd.DataFrame({"user_id": ["u", "v", "w"], "eval": [True, False, False]})
    m = evaluate.build_matrices(items, interactions, users)
    assert m["eval_users"].tolist() == ["u"]
    assert m["history"].toarray().tolist() == [[1, 0, 0, 0]]  # tylko pozytywy < T1
    assert m["seen"].toarray().tolist() == [[1, 1, 0, 1]]  # wszystko < T2, także walidacja i negatywy
    assert m["relevant"].tolist() == [[False, False, True, False]]  # B z walidacji nie jest celem
    assert m["train_counts"].tolist() == [2, 0, 0, 1]
    users.loc[1, "eval"] = True  # v nie ma celu testowego -> błąd zamiast cichego pominięcia
    with pytest.raises(ValueError, match="cel"):
        evaluate.build_matrices(items, interactions, users)



def test_valid_split_targets_validation_period():
    items = pd.DataFrame({"parent_asin": ["A", "B", "C", "D"]})
    rows = [("u", "A", 5.0, T1 - 1, "train"), ("u", "B", 5.0, T1, "valid"), ("u", "C", 5.0, T2, "test"),
            ("v", "A", 5.0, T1 - 2, "train"), ("v", "A", 5.0, T1 + 1, "valid"),  # v: walidacja tylko powtórka
            ("w", "B", 5.0, T1 + 2, "valid"), ("x", "A", 5.0, T1 - 3, "train"), ("x", "D", 2.0, T1 + 3, "valid")]
    interactions = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp", "split"])
    users = pd.DataFrame({"user_id": ["u", "v", "w", "x"], "eval": [True, False, False, False]})
    m = evaluate.build_matrices(items, interactions, users, split="valid")
    assert m["eval_users"].tolist() == ["u"]  # w bez historii, v bez nowego celu, x bez pozytywnego celu
    assert m["seen"].toarray().tolist() == [[1, 0, 0, 0]]  # tylko trening; test w ogóle nie wchodzi
    assert m["relevant"].tolist() == [[False, True, False, False]]
    with pytest.raises(ValueError, match="split"):
        evaluate.build_matrices(items, interactions, users, split="tset")


def test_evaluate_valid_writes_separately(trained):
    before = sorted(p.relative_to(trained.root) for p in trained.results_root.rglob("*")) if trained.results_root.exists() else []
    comparisons = evaluate.evaluate_valid(trained, seed=2026)
    assert (trained.root / "results_valid" / "metrics.csv").is_file()
    assert list(comparisons.columns) == list(evaluate.COMPARISON_COLUMNS)
    after = sorted(p.relative_to(trained.root) for p in trained.results_root.rglob("*")) if trained.results_root.exists() else []
    assert before == after  # wyniki testu nietknięte

def test_only_eval_users_scored(trained):
    per_user = evaluate.per_user_metrics(trained, seed=2026)
    eval_users = pd.read_parquet(trained.prepared / "users.parquet").query("eval")["user_id"]
    assert set(zip(per_user["variant"], per_user["mechanism"])) == {
        *((NO_LLM, m) for m in MECHANISMS), (BASELINE, POPULARITY)}
    for _, group in per_user.groupby(["variant", "mechanism"]):
        assert sorted(group["user_id"]) == sorted(eval_users)
    assert per_user[list(evaluate.METRICS)].notna().all().all()


def test_never_recommends_seen_items(trained, monkeypatch):
    m = evaluate.build_matrices(*(pd.read_parquet(trained.prepared / f"{n}.parquet")
                                  for n in ("items", "interactions", "users")))
    assert not (m["relevant"] & m["seen"].toarray().astype(bool)).any()
    tops = []
    original = evaluate.top_k
    monkeypatch.setattr(evaluate, "top_k", lambda s, v, k: tops.append((v, original(s, v, k))) or tops[-1][1])
    evaluate.per_user_metrics(trained, seed=2026)
    assert len(tops) == 4
    for valid, (top, top_valid) in tops:
        assert np.take_along_axis(valid, top, axis=1)[top_valid].all()


def test_comparison_specs_primary_family():
    one = evaluate.comparison_specs([NO_LLM, "llm_a"])
    primary = [s for s in one if s["role"] == "primary"]
    assert len(primary) == 3 and {s["mechanism"] for s in primary} == set(MECHANISMS)
    assert all(s["metric"] == "ndcg_at_10" and s["baseline"] == NO_LLM for s in primary)
    assert primary[0]["confidence"] == pytest.approx(1 - 0.05 / 3)
    assert all(s["confidence"] == 0.95 for s in one if s["role"] == "supporting")
    assert len(one) == len(MECHANISMS) * len(evaluate.METRICS)
    two = evaluate.comparison_specs([NO_LLM, "llm_b", "llm_a"])
    assert [s for s in two if s["variant"] == "llm_a"] == one
    assert evaluate.comparison_specs([NO_LLM]) == []


def test_compare_pairs_by_user():
    users = [f"u{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    base = rng.random(30)
    frames = []
    for variant, values in ((NO_LLM, base), ("llm_a", base + 0.1)):
        for mechanism in MECHANISMS:
            frame = pd.DataFrame({"user_id": users, "variant": variant, "mechanism": mechanism})
            for metric in evaluate.METRICS:
                frame[metric] = values
            frames.append(frame if variant == NO_LLM else frame.iloc[::-1])  # inna kolejność wierszy
    result = evaluate.compare(pd.concat(frames, ignore_index=True), seed=2026)
    assert len(result) == len(MECHANISMS) * len(evaluate.METRICS)
    assert result["delta"].to_numpy() == pytest.approx(0.1)
    assert result["ci_low"].to_numpy() == pytest.approx(0.1) and result["ci_high"].to_numpy() == pytest.approx(0.1)
    assert set(result["conclusion"]) == {"poprawa"} and (result["users"] == 30).all()


def test_results_layout(trained):
    manifest = evaluate.evaluate(trained, seed=2026)
    for variant, mechanism in [*((NO_LLM, m) for m in MECHANISMS), (BASELINE, POPULARITY)]:
        out = trained.results(variant, mechanism)
        assert {p.name for p in out.iterdir()} == {"per_user.parquet", "metrics.json", "manifest.json"}
        metrics = read_json(out / "metrics.json")
        assert metrics["users"] == manifest["users"] and set(evaluate.METRICS) <= set(metrics)
        assert read_json(out / "manifest.json")["seed"] == 2026
    assert (trained.comparisons / "comparisons.csv").is_file()
    assert (trained.comparisons / "manifest.json").is_file()


def test_missing_two_tower_fails_clearly(trained):
    weights = trained.models(NO_LLM) / "two_tower.pt"
    saved = weights.read_bytes()
    weights.unlink()
    try:
        with pytest.raises(FileNotFoundError, match="sysrekmag-train"):
            evaluate.per_user_metrics(trained)
    finally:
        weights.write_bytes(saved)


@pytest.mark.parametrize("change", ["values", "rows"])
def test_stale_models_fail_clearly(trained, change):
    path = trained.models(NO_LLM) / "embeddings.npy"
    saved = path.read_bytes()
    embeddings = np.load(path)
    replaced = embeddings[::-1] if change == "values" else embeddings[:-1]  # encode bez train / inna próbka
    np.save(path, np.ascontiguousarray(replaced))
    try:
        with pytest.raises(ValueError, match="sysrekmag-train" if change == "values" else "sysrekmag-encode"):
            evaluate.per_user_metrics(trained)
    finally:
        path.write_bytes(saved)
    evaluate.per_user_metrics(trained)  # po przywróceniu znów aktualne
