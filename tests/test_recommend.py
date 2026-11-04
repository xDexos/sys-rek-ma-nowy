import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from conftest import FakeEncoder
from sys_rek_mag.recommend import (
    dense_scores,
    mean_profiles,
    normalize_rows,
    popularity_scores,
    query_scores,
    ranking_metrics,
    tfidf_scores,
    top_k,
)


def _dcg(positions):
    return sum(1 / np.log2(p + 1) for p in positions)


def test_ranking_metrics_known_values():
    # 2 użytkowników, 6 produktów; top ma tylko 6 kolumn (mniej niż max(ks) = 10)
    relevant = np.zeros((2, 6), dtype=bool)
    relevant[0, [1, 5]] = True  # pozycje 2 i 6
    relevant[1, [0]] = True  # pozycja 1, ale niedozwolona -> brak trafienia
    relevant[1, [3]] = True  # pozycja 4
    top = np.tile(np.arange(6), (2, 1))
    top_valid = np.ones_like(top, dtype=bool)
    top_valid[1, 0] = False
    m = ranking_metrics(top, top_valid, relevant)
    assert set(m) == {f"{name}_at_{k}" for name in ("ndcg", "precision", "recall", "hit_rate") for k in (5, 10)} | {
        "mrr_at_10"}
    assert m["ndcg_at_5"][0] == pytest.approx(_dcg([2]) / _dcg([1, 2]))
    assert m["ndcg_at_10"][0] == pytest.approx(_dcg([2, 6]) / _dcg([1, 2]))
    assert m["precision_at_5"][0] == pytest.approx(1 / 5) and m["precision_at_10"][0] == pytest.approx(2 / 10)
    assert m["recall_at_5"][0] == pytest.approx(1 / 2) and m["recall_at_10"][0] == pytest.approx(1.0)
    assert m["hit_rate_at_5"][0] == 1.0 and m["mrr_at_10"][0] == pytest.approx(1 / 2)
    assert m["ndcg_at_5"][1] == pytest.approx(_dcg([4]) / _dcg([1, 2]))
    assert m["precision_at_10"][1] == pytest.approx(1 / 10) and m["recall_at_10"][1] == pytest.approx(1 / 2)
    assert m["mrr_at_10"][1] == pytest.approx(1 / 4) and m["hit_rate_at_10"][1] == 1.0


def test_ties_break_by_lower_index():
    order, order_valid = top_k(np.array([[0.5, 0.9, 0.9, 0.1]]), np.array([[True, True, True, False]]), k=4)
    assert order.tolist() == [[1, 2, 0, 3]] and order_valid.tolist() == [[True, True, True, False]]


def test_rows_without_targets_nan():
    relevant = np.array([[False, False, False], [False, True, False]])
    top = np.tile(np.arange(3), (2, 1))
    m = ranking_metrics(top, np.ones_like(top, dtype=bool), relevant)
    assert all(np.isnan(v[0]) for v in m.values())
    assert m["hit_rate_at_5"][1] == 1.0 and m["mrr_at_10"][1] == pytest.approx(1 / 2)


def test_mean_profiles_dense_and_sparse():
    history = sparse.csr_matrix(np.array([[1, 1, 0], [0, 0, 1]], dtype=float))
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    assert mean_profiles(history, vectors).tolist() == [[0.5, 0.5], [2.0, 2.0]]
    out = mean_profiles(history, sparse.csr_matrix(vectors))
    assert isinstance(out, np.ndarray) and out.tolist() == [[0.5, 0.5], [2.0, 2.0]]
    with pytest.raises(ValueError):
        mean_profiles(sparse.csr_matrix(np.zeros((1, 3))), vectors)


TEXTS = ["led bulb bright light", "led bulb cold light", "wiper blade rain glass", "wiper blade rubber glass",
         "car wax shine paint", "car wax polish paint"]


def test_dense_and_tfidf_prefer_same_group():
    history = sparse.csr_matrix(np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=float))
    valid = np.ones((2, 6), dtype=bool)
    valid[0, 0] = valid[1, 2] = False
    tfidf = TfidfVectorizer().fit_transform(TEXTS).tocsr()
    for scores in (dense_scores(history, FakeEncoder().encode(TEXTS)), tfidf_scores(history, tfidf)):
        assert scores.shape == (2, 6)
        assert top_k(scores, valid, 1)[0].ravel().tolist() == [1, 3]


def test_cosine_scores_bounded_and_query_matches_profile():
    emb = normalize_rows(np.random.default_rng(0).normal(size=(6, 4)))
    history = sparse.csr_matrix(np.array([[1, 1, 0, 0, 0, 0]], dtype=float))
    scores = dense_scores(history, emb)
    assert np.abs(scores).max() <= 1 + 1e-9
    assert query_scores(emb[:2].mean(axis=0), emb) == pytest.approx(scores)
    tfidf = TfidfVectorizer().fit_transform(TEXTS).tocsr()
    assert query_scores(tfidf[0], tfidf).shape == (1, 6)
    assert query_scores(tfidf[0], tfidf)[0, 0] == pytest.approx(1.0)


def test_popularity_scores():
    out = popularity_scores(3, np.array([0, 1, 9]))
    assert out.shape == (3, 3) and out[2].tolist() == pytest.approx(np.log1p([0, 1, 9]).tolist())
