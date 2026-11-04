"""Etap encode: teksty wariantów, embeddingi (MiniLM, CPU) i TF-IDF dla każdego wariantu."""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .config import PATHS, Paths
from .llm_texts import current_generations
from .storage import read_json, read_pickle, text_hash, write_json, write_npy, write_parquet, write_pickle
from .texts import llm_text, no_llm_text

BATCH_SIZE = 64


def load_encoder(name: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(name, device="cpu")
    encoder.max_seq_length = config.MAX_SEQ_LENGTH
    return encoder


def encode(encoder, texts: list[str]) -> np.ndarray:
    vectors = np.asarray(
        encoder.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=len(texts) > BATCH_SIZE),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Enkoder zwrócił niepoprawną macierz")
    return vectors


def complete_llm_variants(paths: Paths) -> list[str]:
    """Warianty llm_*, w których każdy produkt próbki ma generację ważną dla bieżących wejść, promptu i modelu."""
    root = paths.root / "item_texts"
    manifests = sorted(root.glob(f"{config.LLM_PREFIX}*/manifest.json")) if root.is_dir() else []
    n_items = len(pd.read_parquet(paths.prepared / "items.parquet", columns=["parent_asin"]))
    return [m.parent.name for m in manifests if len(current_generations(paths, m.parent.name)) == n_items]


def variant_texts(paths: Paths, items: pd.DataFrame) -> dict[str, list[str]]:
    """no_llm + kompletne warianty LLM, teksty w porządku wierszy items."""
    records = items.to_dict(orient="records")
    texts = {config.NO_LLM: [no_llm_text(r) for r in records]}
    for variant in complete_llm_variants(paths):
        generated = current_generations(paths, variant)
        texts[variant] = [llm_text(generated[r["parent_asin"]]) for r in records]
    return texts


def _fingerprint(asins: list[str], texts: list[str], encoder_name: str) -> str:
    return text_hash("\0".join([encoder_name, str(config.MAX_SEQ_LENGTH), *asins, *texts]))


def current_fingerprints(paths: Paths, encoder_name: str = config.EMBEDDING_MODEL) -> dict[str, str]:
    """Wariant -> odcisk, jaki encode zapisałby teraz (no_llm + kompletne warianty LLM)."""
    items = pd.read_parquet(paths.prepared / "items.parquet")
    asins = list(items["parent_asin"])
    return {v: _fingerprint(asins, t, encoder_name) for v, t in variant_texts(paths, items).items()}


def current_variants(paths: Paths, encoder_name: str = config.EMBEDDING_MODEL) -> list[str]:
    """Warianty dla train i evaluate: no_llm, potem llm_* alfabetycznie, z embeddingami zgodnymi z bieżącymi tekstami.

    Nadmiarowe foldery models/<wariant> (np. po zmianie promptu) są pomijane z ostrzeżeniem; bieżący wariant
    z nieaktualnymi embeddingami to błąd.
    """
    current = current_fingerprints(paths, encoder_name)
    for variant, fingerprint in current.items():
        manifest = paths.models(variant) / "manifest.json"
        if not manifest.is_file() or read_json(manifest).get("fingerprint") != fingerprint:
            raise ValueError(f"Embeddingi wariantu {variant} są nieaktualne albo ich brak; uruchom sysrekmag-encode")
    root = paths.root / "models"
    found = {p.parent.name for p in root.glob("*/embeddings.npy")} if root.is_dir() else set()
    for extra in sorted(found - set(current)):
        print(f"UWAGA: pomijam {extra}: teksty wariantu nie są kompletne dla bieżącej próbki, promptu i wejść LLM")
    return [config.NO_LLM, *sorted(v for v in current if v != config.NO_LLM)]


def _save_tfidf(matrix: sp.csr_matrix, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp.{path.name}")  # save_npz dopisuje .npz, gdy go brak
    sp.save_npz(temporary, matrix)
    os.replace(temporary, path)


def encode_all(paths: Paths = PATHS, encoder=None, encoder_name: str = config.EMBEDDING_MODEL) -> dict[str, Any]:
    items = pd.read_parquet(paths.prepared / "items.parquet")
    asins = list(items["parent_asin"])
    result: dict[str, Any] = {}
    for variant, texts in variant_texts(paths, items).items():
        out = paths.models(variant)
        fingerprint = _fingerprint(asins, texts, encoder_name)
        manifest_path = out / "manifest.json"
        if (manifest_path.is_file() and read_json(manifest_path).get("fingerprint") == fingerprint
                and all((out / f).is_file() for f in ("embeddings.npy", "tfidf.npz", "tfidf_vectorizer.pkl"))
                and (paths.item_texts(variant) / "texts.parquet").is_file()):
            result[variant] = read_json(manifest_path)
            continue
        encoder = encoder or load_encoder(encoder_name)
        print(f"Koduję wariant {variant} ({len(texts)} tekstów)")
        write_parquet(pd.DataFrame({"parent_asin": asins, "text": texts}), paths.item_texts(variant) / "texts.parquet")
        write_npy(encode(encoder, texts), out / "embeddings.npy")
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
        _save_tfidf(vectorizer.fit_transform(texts).tocsr(), out / "tfidf.npz")
        write_pickle(vectorizer, out / "tfidf_vectorizer.pkl")
        lengths = np.array([len(ids) for ids in encoder.tokenizer(texts)["input_ids"]])
        info = {
            "fingerprint": fingerprint, "encoder": encoder_name, "max_seq_length": config.MAX_SEQ_LENGTH,
            "rows": len(texts), "tfidf_terms": len(vectorizer.vocabulary_),
            "texts_over_max_seq_length": int((lengths > config.MAX_SEQ_LENGTH).sum()),
            "tokens_median": float(np.median(lengths)), "tokens_max": int(lengths.max()),
        }
        write_json(info, manifest_path)
        result[variant] = info
        print(f"{variant}: tokeny mediana {info['tokens_median']:.0f}, maks. {info['tokens_max']}, "
              f"ucięte {info['texts_over_max_seq_length']}")
    return result


def load_variant(paths: Paths, variant: str) -> dict[str, Any]:
    out = paths.models(variant)
    return {"embeddings": np.load(out / "embeddings.npy"), "tfidf": sp.load_npz(out / "tfidf.npz"),
            "vectorizer": read_pickle(out / "tfidf_vectorizer.pkl")}


def main() -> None:
    argparse.ArgumentParser(description="Teksty wariantów, embeddingi i TF-IDF (bez LLM)").parse_args()
    encode_all()


if __name__ == "__main__":
    main()
