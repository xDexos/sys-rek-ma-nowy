import numpy as np
import pytest
import torch

from conftest import PREPARE_SIZES, FakeEncoder
from sys_rek_mag import two_tower
from sys_rek_mag.dataset import prepare
from sys_rek_mag.encode import encode_all
from sys_rek_mag.recommend import ranking_metrics, top_k
from sys_rek_mag.storage import read_json

VARIANT = "no_llm"


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    from conftest import write_raw
    from sys_rek_mag.config import Paths

    paths = Paths(tmp_path_factory.mktemp("tt") / "data")
    write_raw(paths.raw)
    prepare(paths, **PREPARE_SIZES)
    encode_all(paths, FakeEncoder())
    return paths


def _weights(paths):
    return torch.load(paths.models(VARIANT) / "two_tower.pt", weights_only=True)


def test_defaults_match_brief():
    assert (two_tower.HIDDEN, two_tower.OUT, two_tower.TEMPERATURE, two_tower.LR, two_tower.BATCH,
            two_tower.MAX_EPOCHS, two_tower.PATIENCE) == (256, 128, 0.05, 1e-3, 256, 50, 5)
    model = two_tower.TwoTower(32)
    u, i = model(torch.randn(3, 32), torch.randn(4, 32))
    assert u.shape == (3, 128) and i.shape == (4, 128)
    assert torch.allclose(u.norm(dim=1), torch.ones(3), atol=1e-5)


def test_training_deterministic(encoded):
    first = two_tower.train_variant(encoded, VARIANT)
    w1 = _weights(encoded)
    (encoded.models(VARIANT) / "two_tower_manifest.json").unlink()
    second = two_tower.train_variant(encoded, VARIANT)
    w2 = _weights(encoded)
    assert first["validation_ndcg_at_10"] == second["validation_ndcg_at_10"]
    assert w1.keys() == w2.keys() and all(torch.equal(w1[k], w2[k]) for k in w1)


def test_two_tower_learns_groups(encoded):
    manifest = two_tower.train_variant(encoded, VARIANT)
    data = two_tower.training_data(encoded, VARIANT)
    users = data["valid_users"]
    relevant = data["valid_relevant"][users].toarray()
    valid = ~data["train_seen"][users].toarray()
    rng = np.random.default_rng(0)
    random_ndcg = np.mean([np.nanmean(ranking_metrics(*top_k(rng.random(valid.shape), valid, 10), relevant, ks=(10,))[
        "ndcg_at_10"]) for _ in range(50)])
    model = two_tower.load_two_tower(encoded, VARIANT)
    scores = two_tower.two_tower_scores(model, data["train_history"][users], data["embeddings"])
    ndcg = np.nanmean(ranking_metrics(*top_k(scores, valid, 10), relevant, ks=(10,))["ndcg_at_10"])
    assert ndcg == pytest.approx(manifest["best_ndcg_at_10"])
    assert ndcg > random_ndcg + 0.1
    assert 1 <= manifest["best_epoch"] <= len(manifest["validation_ndcg_at_10"]) <= two_tower.MAX_EPOCHS
    query = two_tower.two_tower_query(model, data["embeddings"][0], data["embeddings"])
    assert query.shape == (1, data["embeddings"].shape[0])


def test_cache_reused(encoded, monkeypatch):
    two_tower.train_variant(encoded, VARIANT)
    monkeypatch.setattr(two_tower, "_fit", lambda *a, **k: pytest.fail("trening mimo zgodnego odcisku"))
    assert two_tower.train_variant(encoded, VARIANT) == read_json(encoded.models(VARIANT) / "two_tower_manifest.json")


def test_main_trains_every_variant(encoded, monkeypatch):
    calls = []
    monkeypatch.setattr(two_tower, "PATHS", encoded)
    monkeypatch.setattr(two_tower, "train_variant", lambda paths, variant: calls.append(variant) or {})
    monkeypatch.setattr("sys.argv", ["sysrekmag-train"])
    two_tower.main()
    assert calls == [VARIANT]


def test_examples_match_naive_leave_one_out():
    import scipy.sparse as sp

    history = sp.csr_matrix(np.array([[1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=float))
    embeddings = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    inputs, targets = two_tower._examples(history, embeddings)
    expected = [(np.delete(row, k).tolist(), t) for u in (0, 3) for row in [history[u].indices]
                for k, t in enumerate(row)]
    assert targets.tolist() == [t for _, t in expected]  # użytkownik z 1 pozytywem i bez pozytywów pominięci
    assert np.allclose(inputs, [embeddings[rest].mean(axis=0) for rest, _ in expected])


def test_fingerprint_includes_positive_rating(encoded, monkeypatch):
    before = two_tower.fingerprint(encoded, VARIANT)
    monkeypatch.setattr(two_tower.config, "POSITIVE_RATING", 5)
    assert two_tower.fingerprint(encoded, VARIANT) != before
