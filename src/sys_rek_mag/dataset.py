"""Etap prepare: próbka produktów Automotive, interakcje z podziałem czasowym i wejścia dla LLM.

Kandydaci pochodzą wyłącznie z treningowej części 5-core (Automotive.train.csv.gz); pliki valid/test
5-core nie są czytane. Wybór produktów korzysta wyłącznie z opinii sprzed T1; dane z okresu walidacji i testu trafiają
tylko do interactions.parquet (z kolumną split) i nie wpływają na próbkę ani na wejścia LLM.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from sys_rek_mag import config
from sys_rek_mag.config import PATHS, Paths
from sys_rek_mag.storage import write_json, write_jsonl, write_parquet

REVIEWS_FILE = "Automotive.jsonl.gz"
META_FILE = "meta_Automotive.jsonl.gz"
CORE_TRAIN_FILE = "Automotive.train.csv.gz"
META_FIELDS = ("parent_asin", "title", "features", "description", "categories")
REVIEW_SORT = (["helpful_vote", "timestamp", "user_id"], [False, False, True])


def _stream(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def read_candidates(raw: Path) -> set[str]:
    """parent_asin z treningowej części 5-core (valid/test pominięte: zależą od danych po T1)."""
    return set(pd.read_csv(raw / CORE_TRAIN_FILE, usecols=["parent_asin"], dtype=str)["parent_asin"])


def read_meta(raw: Path, candidates: set[str]) -> pd.DataFrame:
    """Metadane kandydatów z niepustym tytułem, tylko pola z kontraktu."""
    rows = []
    for row in _stream(raw / META_FILE):
        title = (row.get("title") or "").strip()
        if row.get("parent_asin") in candidates and title:
            rows.append({"parent_asin": row["parent_asin"], "title": title,
                         **{f: list(row.get(f) or []) for f in ("features", "description", "categories")}})
    return pd.DataFrame(rows, columns=list(META_FIELDS))


def _has_text(review: dict) -> bool:
    return bool((review.get("text") or "").strip())


def count_train_reviews(raw: Path, candidates: set[str]) -> pd.Series:
    """Przebieg 1: liczba opinii z niepustym tekstem i timestamp < T1."""
    counts = Counter(r["parent_asin"] for r in _stream(raw / REVIEWS_FILE)
                     if r["timestamp"] < config.T1 and r["parent_asin"] in candidates and _has_text(r))
    return pd.Series(counts, dtype="int64")


def select_items(candidates: set[str], meta: pd.DataFrame, train_counts: pd.Series, n_items: int,
                 subcategory: str | None) -> pd.DataFrame:
    """Top n_items wg liczby opinii treningowych (remis: niższy parent_asin); wynik posortowany po parent_asin."""
    pool = meta[meta["parent_asin"].isin(candidates)].copy()
    pool["train_reviews"] = pool["parent_asin"].map(train_counts).fillna(0).astype("int64")
    pool = pool[pool["train_reviews"].ge(config.MIN_ITEM_TRAIN_REVIEWS)]
    if subcategory is not None:
        pool = pool[pool["categories"].map(lambda c: subcategory in c)]
    top = pool.sort_values(["train_reviews", "parent_asin"], ascending=[False, True]).head(n_items)
    return top.sort_values("parent_asin").reset_index(drop=True)


def pick_reviews(reviews: pd.DataFrame) -> list[dict]:
    """Opinie dla LLM: sort (-helpful_vote, -timestamp, user_id), pierwsze MAX_REVIEWS_PER_ITEM, tekst obcięty."""
    top = reviews.sort_values(REVIEW_SORT[0], ascending=REVIEW_SORT[1], kind="stable").head(config.MAX_REVIEWS_PER_ITEM)
    return [{"rating": float(r.rating), "title": str(r.title or ""), "text": r.text.strip()[: config.REVIEW_MAX_CHARS]}
            for r in top.itertuples()]


def _review_key(review: dict) -> tuple:
    return -review["helpful_vote"], -review["timestamp"], review["user_id"]


def scan_selected(raw: Path, selected: set[str]) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Przebieg 2: najwcześniejsza interakcja każdej pary użytkownik–produkt i opinie treningowe z tekstem."""
    earliest: dict[tuple[str, str], tuple[int, float]] = {}
    texts: dict[str, list[dict]] = defaultdict(list)
    for r in _stream(raw / REVIEWS_FILE):
        asin = r["parent_asin"]
        if asin not in selected:
            continue
        pair, timestamp = (r["user_id"], asin), int(r["timestamp"])
        if pair not in earliest or timestamp < earliest[pair][0]:  # remis czasu: zostaje pierwsza w pliku
            earliest[pair] = (timestamp, float(r["rating"]))
        if timestamp < config.T1 and _has_text(r):
            bucket = texts[asin]
            bucket.append({k: r.get(k) for k in ("rating", "title", "text", "helpful_vote", "user_id")}
                          | {"timestamp": timestamp, "helpful_vote": int(r.get("helpful_vote") or 0)})
            if len(bucket) > 4 * config.MAX_REVIEWS_PER_ITEM:  # stabilne przycięcie = ta sama kolejność co pick_reviews
                texts[asin] = sorted(bucket, key=_review_key)[: config.MAX_REVIEWS_PER_ITEM]
    interactions = pd.DataFrame(
        [(u, a, rating, ts) for (u, a), (ts, rating) in earliest.items()],
        columns=["user_id", "parent_asin", "rating", "timestamp"],
    ).astype({"user_id": str, "parent_asin": str, "rating": "float64", "timestamp": "int64"})
    interactions["split"] = assign_split(interactions["timestamp"])
    interactions = interactions.sort_values(["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)
    return interactions, texts


def assign_split(timestamps: pd.Series) -> pd.Series:
    split = pd.Series("valid", index=timestamps.index)
    split[timestamps < config.T1] = "train"
    split[timestamps >= config.T2] = "test"
    return split


def build_users(interactions: pd.DataFrame) -> pd.DataFrame:
    """eval: ≥1 pozytyw w treningu i ≥1 pozytyw w teście; para to najwcześniejsza interakcja,
    więc split == test oznacza produkt bez interakcji użytkownika przed T2."""
    positive = interactions["rating"].ge(config.POSITIVE_RATING)
    train = interactions[positive & interactions["split"].eq("train")].groupby("user_id").size()
    test_users = set(interactions.loc[positive & interactions["split"].eq("test"), "user_id"])
    users = pd.DataFrame({"user_id": sorted(set(interactions["user_id"]))})
    users["train_positives"] = users["user_id"].map(train).fillna(0).astype("int64")
    users["eval"] = users["train_positives"].ge(1) & users["user_id"].isin(test_users)
    return users


def top_subcategories(pool: pd.DataFrame, n: int = 10) -> list[list]:
    """Podkategorie z największą sumą train_reviews (tylko dane sprzed T1)."""
    exploded = pool[["categories", "train_reviews"]].explode("categories").dropna()
    sums = exploded.groupby("categories")["train_reviews"].sum()
    sums = sums.drop("Automotive", errors="ignore")  # kategoria główna, nie podkategoria
    ranked = sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [[name, int(total)] for name, total in ranked]


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare(paths: Paths = PATHS, n_items: int = config.N_ITEMS, subcategory: str | None = None) -> dict:
    raw = paths.raw
    candidates = read_candidates(raw)
    meta = read_meta(raw, candidates)
    train_counts = count_train_reviews(raw, set(meta["parent_asin"]))
    items = select_items(candidates, meta, train_counts, n_items, subcategory)
    if items.empty:
        raise ValueError("Brak produktów spełniających warunki próbki")
    interactions, texts = scan_selected(raw, set(items["parent_asin"]))
    llm_inputs = [{"parent_asin": a, "reviews": pick_reviews(pd.DataFrame(texts[a]))} for a in items["parent_asin"]]
    users = build_users(interactions)

    eligible = select_items(candidates, meta, train_counts, len(meta), None)
    manifest = {
        "params": {"T1": config.T1, "T2": config.T2, "positive_rating": config.POSITIVE_RATING, "n_items": n_items,
                   "min_item_train_reviews": config.MIN_ITEM_TRAIN_REVIEWS,
                   "max_reviews_per_item": config.MAX_REVIEWS_PER_ITEM, "review_max_chars": config.REVIEW_MAX_CHARS},
        "subcategory": subcategory,
        "counts": {
            "candidates_from_train_5core": len(candidates), "candidates_with_title": len(meta),
            "eligible_items": len(eligible), "items": len(items),
            "interactions": {k: int(v) for k, v in interactions["split"].value_counts().items()},
            "users": len(users), "eval_users": int(users["eval"].sum()),
        },
        "top_subcategories": top_subcategories(eligible),
        "sources": {name: _fingerprint(raw / name) for name in (REVIEWS_FILE, META_FILE, CORE_TRAIN_FILE)},
    }
    out = paths.prepared
    write_parquet(items, out / "items.parquet")
    write_parquet(interactions, out / "interactions.parquet")
    write_jsonl(llm_inputs, out / "llm_inputs.jsonl")
    write_parquet(users, out / "users.parquet")
    write_json(manifest, out / "manifest.json")
    if manifest["counts"]["eval_users"] < config.MIN_EVAL_USERS_WARN:
        print(f"UWAGA: tylko {manifest['counts']['eval_users']} użytkowników ewaluacyjnych "
              f"(< {config.MIN_EVAL_USERS_WARN}). Podkategorie z największą sumą opinii treningowych "
              "(decyzję o --subcategory wpisz do protokołu):")
        for name, total in manifest["top_subcategories"]:
            print(f"  {name}: {total}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Przygotuj próbkę produktów Automotive, interakcje i wejścia LLM")
    parser.add_argument("--items", type=int, default=config.N_ITEMS, help="liczba produktów w próbce")
    parser.add_argument("--subcategory", default=None, help="ogranicz próbkę do jednej podkategorii")
    args = parser.parse_args()
    manifest = prepare(n_items=args.items, subcategory=args.subcategory)
    counts = manifest["counts"]
    print(f"Gotowe: {counts['items']} produktów, interakcje {counts['interactions']}, "
          f"{counts['users']:,} użytkowników, w tym {counts['eval_users']:,} ewaluacyjnych")
