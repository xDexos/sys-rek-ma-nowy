"""Ranking i metryki jako czyste funkcje na macierzach (wiersze = użytkownicy, kolumny = produkty).

Kolumny są zawsze w porządku rosnącego parent_asin, więc remisy rozstrzyga niższy parent_asin.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .config import TOP_KS


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def mean_profiles(history: sparse.csr_matrix, item_vectors) -> np.ndarray:
    """Profil użytkownika = średnia wektorów produktów z historii (wiersz macierzy 0/1); zawsze gęsty."""
    counts = np.asarray(history.sum(axis=1)).ravel()
    if (counts == 0).any():
        raise ValueError("Każdy użytkownik musi mieć co najmniej jeden produkt w historii")
    sums = history @ item_vectors
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    return sums.astype(np.float64) / counts[:, None]


def _cosine(profiles: np.ndarray, item_vectors) -> np.ndarray:
    return np.asarray(normalize(item_vectors) @ normalize_rows(profiles).T, dtype=np.float64).T


def dense_scores(history: sparse.csr_matrix, embeddings: np.ndarray) -> np.ndarray:
    """Cosinus średniego profilu z embeddingami produktów."""
    return _cosine(mean_profiles(history, embeddings), embeddings)


def tfidf_scores(history: sparse.csr_matrix, tfidf: sparse.csr_matrix) -> np.ndarray:
    """Cosinus średniego profilu TF-IDF (L2-normalizowanego) z wektorami TF-IDF produktów."""
    return _cosine(mean_profiles(history, tfidf), tfidf)


def popularity_scores(n_users: int, counts: np.ndarray) -> np.ndarray:
    """log1p(liczba ocen przed T1) — ten sam wiersz dla każdego użytkownika (widok tylko do odczytu)."""
    row = np.log1p(np.asarray(counts, dtype=np.float64))
    return np.broadcast_to(row, (n_users, len(row)))


def query_scores(query_vector, item_vectors) -> np.ndarray:
    """Cosinus jednego profilu tekstowego (gęsty wektor albo wiersz rzadki TF-IDF) z produktami; 1 × N."""
    query = query_vector.toarray() if sparse.issparse(query_vector) else np.atleast_2d(query_vector)
    return _cosine(query, item_vectors)


def top_k(scores: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indeksy k najlepszych produktów i maska, czy pozycja jest ważna (gdy ważnych < k)."""
    masked = np.where(valid, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(valid, order, axis=1)


def ranking_metrics(top: np.ndarray, top_valid: np.ndarray, relevant: np.ndarray,
                    ks: tuple[int, ...] = TOP_KS) -> dict[str, np.ndarray]:
    """Binarne NDCG, Precision, Recall, HitRate @k i MRR@max(ks) per wiersz; wiersze bez celów -> NaN.

    `top` ma max(ks) kolumn (mniej, gdy produktów jest mniej); Precision@k = trafienia / k."""
    hits = np.take_along_axis(relevant, top, axis=1) & top_valid
    n_relevant = relevant.sum(axis=1)
    has = n_relevant > 0
    max_k = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    ideal_cum = np.concatenate([[0.0], np.cumsum(discounts)])
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in ks:
            h = hits[:, :k]
            n_hits = h.sum(axis=1)
            dcg = (h * discounts[: h.shape[1]]).sum(axis=1)
            out[f"ndcg_at_{k}"] = np.where(has, dcg / ideal_cum[np.minimum(n_relevant, k)], np.nan)
            out[f"precision_at_{k}"] = np.where(has, n_hits / k, np.nan)
            out[f"recall_at_{k}"] = np.where(has, n_hits / n_relevant, np.nan)
            out[f"hit_rate_at_{k}"] = np.where(has, (n_hits > 0).astype(float), np.nan)
        h = hits[:, :max_k]
        reciprocal = np.where(h.any(axis=1), 1.0 / (h.argmax(axis=1) + 1), 0.0)
        out[f"mrr_at_{max_k}"] = np.where(has, reciprocal, np.nan)
    return out
