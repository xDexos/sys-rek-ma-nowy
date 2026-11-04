"""Etap evaluate: metryki per użytkownik dla każdego wariantu × mechanizmu i sparowany bootstrap różnic."""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import bootstrap

from . import config
from .config import PATHS, Paths
from .encode import load_variant
from .recommend import dense_scores, popularity_scores, ranking_metrics, tfidf_scores, top_k
from .storage import file_hash, write_csv, write_json, write_parquet
from .two_tower import load_two_tower, two_tower_scores

METRICS = tuple(f"{m}_at_{k}" for k in config.TOP_KS for m in ("ndcg", "precision", "recall", "hit_rate")) + (
    f"mrr_at_{max(config.TOP_KS)}",)
PRIMARY_METRIC = "ndcg_at_10"
BATCH_USERS = 4000
PRIMARY_CONFIDENCE = 1 - config.FAMILY_ALPHA / config.PRIMARY_COMPARISONS_PER_MODEL
PREPARED_FILES = ("items", "interactions", "users")
COMPARISON_COLUMNS = ("variant", "baseline", "mechanism", "metric", "role", "confidence", "users", "baseline_mean",
                      "variant_mean", "delta", "delta_relative", "ci_low", "ci_high", "conclusion")
MECHANISM_FILES = {"tfidf": ("tfidf.npz",), "dense": ("embeddings.npy",), "two_tower": ("embeddings.npy", "two_tower.pt")}


def build_matrices(items: pd.DataFrame, interactions: pd.DataFrame, users: pd.DataFrame) -> dict[str, Any]:
    """Macierze użytkownik ewaluacyjny × produkt (kolumny w porządku items, czyli rosnącego parent_asin)."""
    asins = pd.Index(items["parent_asin"])
    if not asins.is_monotonic_increasing or not asins.is_unique:
        raise ValueError("items.parquet musi być posortowany po parent_asin bez duplikatów")
    frame = interactions[interactions["parent_asin"].isin(asins)]
    eval_users = np.sort(users.loc[users["eval"], "user_id"].to_numpy(dtype=str))
    cols = asins.get_indexer(frame["parent_asin"])
    train_counts = np.bincount(cols[frame["split"].eq("train").to_numpy()], minlength=len(asins))
    rows = pd.Index(eval_users).get_indexer(frame["user_id"])
    keep = rows >= 0
    frame, rows, cols = frame[keep], rows[keep], cols[keep]
    shape = (len(eval_users), len(asins))
    positive = frame["rating"].ge(config.POSITIVE_RATING).to_numpy()
    split = frame["split"]

    def matrix(mask: np.ndarray) -> sparse.csr_matrix:
        out = sparse.csr_matrix((np.ones(mask.sum()), (rows[mask], cols[mask])), shape=shape)
        out.data[:] = 1.0  # duplikaty par sumowałyby się
        return out

    history = matrix(positive & split.eq("train").to_numpy())
    seen = matrix(split.isin(["train", "valid"]).to_numpy())  # każda interakcja przed T2
    relevant = (matrix(positive & split.eq("test").to_numpy()).toarray() > 0) & ~(seen.toarray() > 0)
    if (np.diff(history.indptr) == 0).any():
        raise ValueError("Użytkownik ewaluacyjny bez pozytywu w treningu; uruchom ponownie sysrekmag-prepare")
    if not relevant.any(axis=1).all():
        raise ValueError("Użytkownik ewaluacyjny bez celu testowego; uruchom ponownie sysrekmag-prepare")
    return {"eval_users": eval_users, "history": history, "seen": seen, "relevant": relevant,
            "train_counts": train_counts}


def evaluated_variants(paths: Paths) -> list[str]:
    """no_llm, potem warianty llm_* alfabetycznie — tylko te z embeddingami."""
    root = paths.root / "models"
    found = {p.parent.name for p in root.glob("*/embeddings.npy")} if root.is_dir() else set()
    llm = sorted(v for v in found if v.startswith(config.LLM_PREFIX))
    return [config.NO_LLM, *llm] if config.NO_LLM in found else llm


def _load_prepared(paths: Paths) -> list[pd.DataFrame]:
    return [pd.read_parquet(paths.prepared / f"{name}.parquet") for name in PREPARED_FILES]


def per_user_metrics(paths: Paths = PATHS, seed: int = config.SEED) -> pd.DataFrame:
    """Metryki per użytkownik ewaluacyjny; seed tylko do manifestu — ranking jest deterministyczny."""
    m = build_matrices(*_load_prepared(paths))
    variants = evaluated_variants(paths)
    if not variants:
        raise FileNotFoundError("Brak embeddingów w models/; uruchom sysrekmag-encode")
    models: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if not (paths.models(variant) / "two_tower.pt").is_file():
            raise FileNotFoundError(f"Brak sieci dwuwieżowej dla wariantu {variant}; uruchom sysrekmag-train")
        models[variant] = load_variant(paths, variant) | {"two_tower": load_two_tower(paths, variant)}
    scorers = {
        "tfidf": lambda data, history: tfidf_scores(history, data["tfidf"]),
        "dense": lambda data, history: dense_scores(history, data["embeddings"]),
        "two_tower": lambda data, history: two_tower_scores(data["two_tower"], history, data["embeddings"]),
    }
    rows: list[pd.DataFrame] = []
    for start in range(0, len(m["eval_users"]), BATCH_USERS):
        batch = slice(start, start + BATCH_USERS)
        history, valid, relevant = m["history"][batch], ~(m["seen"][batch].toarray() > 0), m["relevant"][batch]

        def score(scores: np.ndarray, variant: str, mechanism: str) -> None:
            top, top_valid = top_k(scores, valid, max(config.TOP_KS))
            result = ranking_metrics(top, top_valid, relevant, config.TOP_KS)
            rows.append(pd.DataFrame({"user_id": m["eval_users"][batch], "variant": variant, "mechanism": mechanism,
                                      **{k: result[k] for k in METRICS}}))

        for variant in variants:
            for mechanism in config.MECHANISMS:
                score(scorers[mechanism](models[variant], history), variant, mechanism)
        score(popularity_scores(history.shape[0], m["train_counts"]), config.BASELINE, config.POPULARITY)
    return pd.concat(rows, ignore_index=True)


def _bootstrap(deltas: np.ndarray, confidence: float, label: str, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng([seed, zlib.crc32(label.encode())])  # ziarno zależy od porównania, nie od kolejności
    result = bootstrap((deltas,), np.mean, confidence_level=confidence, n_resamples=config.BOOTSTRAP_RESAMPLES,
                       method="percentile", batch=500, rng=rng)
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def comparison_specs(variants: list[str]) -> list[dict]:
    """Każdy llm_* vs no_llm dla każdego mechanizmu i metryki; NDCG@10 to rodzina główna (3 na model)."""
    if config.NO_LLM not in variants:
        return []
    specs = []
    for variant in sorted(v for v in variants if v.startswith(config.LLM_PREFIX)):
        for mechanism in config.MECHANISMS:
            for metric in METRICS:
                primary = metric == PRIMARY_METRIC
                specs.append({"variant": variant, "baseline": config.NO_LLM, "mechanism": mechanism, "metric": metric,
                              "role": "primary" if primary else "supporting",
                              "confidence": PRIMARY_CONFIDENCE if primary else config.SUPPORTING_CONFIDENCE})
    return specs


def compare(per_user: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    out = []
    for spec in comparison_specs(sorted(per_user["variant"].unique())):
        group = per_user[per_user["mechanism"].eq(spec["mechanism"])]
        wide = group.pivot(index="user_id", columns="variant", values=spec["metric"])  # parowanie po user_id
        paired = wide[[spec["variant"], spec["baseline"]]].dropna()
        deltas = (paired[spec["variant"]] - paired[spec["baseline"]]).to_numpy()
        label = "|".join(spec[k] for k in ("mechanism", "metric", "variant", "baseline"))
        low, high = _bootstrap(deltas, spec["confidence"], label, seed)
        baseline_mean = paired[spec["baseline"]].mean()
        out.append({**spec, "users": len(deltas), "baseline_mean": baseline_mean,
                    "variant_mean": paired[spec["variant"]].mean(), "delta": deltas.mean(),
                    "delta_relative": deltas.mean() / baseline_mean if baseline_mean else np.nan,
                    "ci_low": low, "ci_high": high,
                    "conclusion": "poprawa" if low > 0 else "pogorszenie" if high < 0 else "brak istotnej różnicy"})
    return pd.DataFrame(out, columns=list(COMPARISON_COLUMNS))  # nagłówek także bez wariantów LLM


def _fingerprints(files: list[Path]) -> dict[str, str]:
    return {f"{path.parent.name}/{path.name}": file_hash(path) for path in files}


def evaluate(paths: Paths = PATHS, seed: int = config.SEED) -> dict:
    per_user = per_user_metrics(paths, seed)
    prepared = [paths.prepared / f"{name}.parquet" for name in PREPARED_FILES]
    common = {"seed": seed, "k": list(config.TOP_KS), "positive_rating": config.POSITIVE_RATING}
    for (variant, mechanism), group in per_user.groupby(["variant", "mechanism"], sort=False):
        out = paths.results(variant, mechanism)
        model_files = [paths.models(variant) / f for f in MECHANISM_FILES.get(mechanism, ())]
        write_parquet(group.reset_index(drop=True), out / "per_user.parquet")
        write_json({"users": len(group), **{k: float(group[k].mean()) for k in METRICS}}, out / "metrics.json")
        write_json({**common, "variant": variant, "mechanism": mechanism,
                    "input_fingerprints": _fingerprints(prepared + model_files)}, out / "manifest.json")
    comparisons = compare(per_user, seed)
    write_csv(comparisons, paths.comparisons / "comparisons.csv")
    manifest = {**common, "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES, "primary_confidence": PRIMARY_CONFIDENCE,
                "supporting_confidence": config.SUPPORTING_CONFIDENCE, "users": int(per_user["user_id"].nunique()),
                "variants": list(dict.fromkeys(per_user["variant"])),
                "input_fingerprints": _fingerprints([*prepared, *(paths.models(v) / f for v in evaluated_variants(paths)
                                                                  for f in ("embeddings.npy", "tfidf.npz", "two_tower.pt"))])}
    write_json(manifest, paths.comparisons / "manifest.json")
    return manifest


def main() -> None:
    argparse.ArgumentParser(description="Ewaluacja: metryki per użytkownik i sparowany bootstrap llm_* vs no_llm").parse_args()
    manifest = evaluate()
    print(f"Ocenieni użytkownicy: {manifest['users']}; wyniki w {PATHS.results_root}")
    comparisons = pd.read_csv(PATHS.comparisons / "comparisons.csv")
    if comparisons.empty:
        print("Brak wariantów LLM do porównania z no_llm")
    else:
        print(comparisons[comparisons["role"].eq("primary")].to_string(index=False))


if __name__ == "__main__":
    main()
