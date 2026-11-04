import gzip
import json

import pandas as pd
import pytest

from conftest import (
    BLANK_ID, BLANK_TITLE_MARKER, BOUNDARY_USER, DUPLICATE_PAIR, EVAL_USER, FEW_REVIEWS_ID, ITEM_IDS, LATE_5CORE_ID,
    NO_TITLE_ID, NO_TRAIN_POSITIVE_USER, OUTSIDE_5CORE_ID, PREPARE_SIZES, VALID_THEN_TEST_USER, write_raw,
)
from sys_rek_mag.config import MAX_REVIEWS_PER_ITEM, REVIEW_MAX_CHARS, T1, T2, Paths
from sys_rek_mag.dataset import pick_reviews, prepare, read_meta, select_items
from sys_rek_mag.storage import read_jsonl


def _meta(asins, categories=None):
    return pd.DataFrame({
        "parent_asin": asins, "title": [f"T {a}" for a in asins], "features": [[] for _ in asins],
        "description": [[] for _ in asins], "categories": categories or [[] for _ in asins],
    })


def test_select_items_order_and_threshold():
    meta = _meta(["A", "B", "C", "D", "E"], [["x"], ["x"], ["y"], ["x"], ["x"]])
    counts = pd.Series({"A": 7, "B": 9, "C": 9, "D": 4, "E": 7})
    selected = select_items({"A", "B", "C", "D", "E"}, meta, counts, n_items=3, subcategory=None)
    # top 3 malejąco: B, C (9), potem A wygrywa remis 7 z E (niższy parent_asin); wynik po parent_asin
    assert selected["parent_asin"].tolist() == ["A", "B", "C"]
    assert selected["train_reviews"].tolist() == [7, 9, 9]
    selected = select_items({"A", "B", "C", "D", "E"}, meta, counts, n_items=10, subcategory="x")
    assert selected["parent_asin"].tolist() == ["A", "B", "E"]  # D poniżej progu 5, C spoza podkategorii
    assert select_items({"A"}, meta, counts, 10, None)["parent_asin"].tolist() == ["A"]


def _prepare(paths):
    manifest = prepare(paths, **PREPARE_SIZES)
    p = paths.prepared
    return (manifest, pd.read_parquet(p / "items.parquet"), pd.read_parquet(p / "interactions.parquet"),
            read_jsonl(p / "llm_inputs.jsonl"), pd.read_parquet(p / "users.parquet"))


def _raw_reviews(paths):
    with gzip.open(paths.raw / "Automotive.jsonl.gz", "rt") as file:
        return [json.loads(line) for line in file]


def test_items_without_title_or_outside_5core_excluded(paths, raw):
    manifest, items, interactions, llm_inputs, _ = _prepare(paths)
    assert len(items) == PREPARE_SIZES["n_items"] and manifest["counts"]["items"] == len(items)
    assert items["parent_asin"].is_monotonic_increasing
    assert set(items["parent_asin"]) <= set(ITEM_IDS)
    for excluded in (NO_TITLE_ID, OUTSIDE_5CORE_ID, FEW_REVIEWS_ID, LATE_5CORE_ID):  # LATE: tylko valid/test 5-core
        assert excluded not in set(items["parent_asin"]) | set(interactions["parent_asin"])
    assert set(interactions["parent_asin"]) <= set(items["parent_asin"])
    assert [r["parent_asin"] for r in llm_inputs] == items["parent_asin"].tolist()
    assert list(items.columns) == ["parent_asin", "title", "features", "description", "categories", "train_reviews"]
    assert set(manifest["sources"]) == {"Automotive.jsonl.gz", "meta_Automotive.jsonl.gz", "Automotive.train.csv.gz"}
    assert manifest["counts"]["candidates_from_train_5core"] == len(ITEM_IDS) + 3


def test_blank_reviews_ignored(paths, raw):
    _, items, _, llm_inputs, _ = _prepare(paths)
    assert BLANK_ID not in set(items["parent_asin"])  # 4 opinie z tekstem + 1 biała < 5
    assert ITEM_IDS[1] in set(items["parent_asin"])
    texts = [r for record in llm_inputs for r in record["reviews"]]
    assert all(r["text"].strip() for r in texts)
    assert BLANK_TITLE_MARKER not in {r["title"] for r in texts}


def test_duplicate_pair_keeps_earliest(paths, raw):
    _, _, interactions, _, _ = _prepare(paths)
    user, asin = DUPLICATE_PAIR
    pair = interactions[interactions["user_id"].eq(user) & interactions["parent_asin"].eq(asin)]
    assert len(pair) == 1 and pair.iloc[0]["rating"] == 5.0 and pair.iloc[0]["split"] == "train"
    assert not interactions.duplicated(["user_id", "parent_asin"]).any()


def test_splits_by_cutoffs(paths, raw):
    manifest, _, interactions, _, _ = _prepare(paths)
    edge = interactions[interactions["user_id"].eq(BOUNDARY_USER)].set_index("timestamp")["split"]
    assert edge.to_dict() == {T1 - 1: "train", T1: "valid", T2 - 1: "valid", T2: "test"}
    expected = pd.cut(interactions["timestamp"], [-1, T1 - 1, T2 - 1, 2**62], labels=["train", "valid", "test"])
    assert (interactions["split"] == expected.astype(str)).all()
    assert manifest["counts"]["interactions"] == interactions["split"].value_counts().to_dict()
    assert interactions["timestamp"].dtype == "int64" and interactions["rating"].dtype == "float64"


def test_pick_reviews_order_and_truncation():
    reviews = pd.DataFrame({  # helpful_vote zależy od głosów po T1, więc nie wpływa na kolejność
        "rating": [1.0, 2.0, 3.0, 4.0, 5.0], "title": ["a", "b", "c", "d", "e"],
        "text": ["x" * (REVIEW_MAX_CHARS + 50), "short", "tie u2", " tie u1 ", "newest"],
        "helpful_vote": [0, 100, 3, 3, 0], "timestamp": [10, 99, 20, 20, 30], "user_id": ["u", "u", "u2", "u1", "u"],
    })
    picked = pick_reviews(reviews)  # dłuższy tekst, potem nowsza, potem user_id
    assert [r["title"] for r in picked] == ["a", "e", "d", "c", "b"]
    assert picked[0] == {"rating": 1.0, "title": "a", "text": "x" * REVIEW_MAX_CHARS}
    many = pd.DataFrame({"rating": 5.0, "title": "t", "text": "y", "timestamp": range(30), "user_id": "u"})
    assert len(pick_reviews(many)) == MAX_REVIEWS_PER_ITEM


def test_llm_inputs_only_train_and_no_product_title(paths, raw):
    _, items, _, llm_inputs, _ = _prepare(paths)
    late = {r["text"] for r in _raw_reviews(paths) if r["timestamp"] >= T1}
    product_strings = {s for _, row in items.iterrows() for s in [row["title"], *row["description"]]}
    for record in llm_inputs:
        assert set(record) == {"parent_asin", "reviews"} and record["reviews"]
        for review in record["reviews"]:
            assert set(review) == {"rating", "title", "text"}
            assert review["text"] not in late
            assert not product_strings & {review["title"], review["text"]}


def test_eval_users_definition(paths, raw):
    manifest, _, interactions, _, users = _prepare(paths)
    flags = users.set_index("user_id")["eval"]
    assert flags[EVAL_USER] and not flags[VALID_THEN_TEST_USER] and not flags[NO_TRAIN_POSITIVE_USER]
    assert not flags[DUPLICATE_PAIR[0]]  # późniejsza ocena tego samego produktu nie jest celem
    positive = interactions["rating"].ge(4)
    train = interactions[positive & interactions["split"].eq("train")].groupby("user_id").size()
    test = set(interactions.loc[positive & interactions["split"].eq("test"), "user_id"])
    expected = {u for u in train.index if u in test}
    assert set(users.loc[users["eval"], "user_id"]) == expected and manifest["counts"]["eval_users"] == len(expected)
    assert users.set_index("user_id")["train_positives"].reindex(train.index).tolist() == train.tolist()
    assert set(users["user_id"]) == set(interactions["user_id"])


def test_selection_ignores_test_data(paths, raw, tmp_path):
    _, items, _, llm_inputs, _ = _prepare(paths)
    loser = ITEM_IDS[-1]
    assert loser not in set(items["parent_asin"])
    other = Paths(tmp_path / "other")
    extra = [{"user": f"Z{k}", "asin": loser, "rating": 5, "timestamp": T1 + k * (T2 - T1) // 50} for k in range(100)]
    write_raw(other.raw, extra_reviews=extra)
    _, items2, _, llm_inputs2, _ = _prepare(other)
    pd.testing.assert_frame_equal(items, items2)
    assert llm_inputs == llm_inputs2


def test_streaming_prune_matches_pick_reviews(paths, raw, monkeypatch):
    monkeypatch.setattr("sys_rek_mag.config.MAX_REVIEWS_PER_ITEM", 2)  # przycinanie w przebiegu 2 od > 8 opinii
    _, _, _, llm_inputs, _ = _prepare(paths)
    train = pd.DataFrame([r for r in _raw_reviews(paths) if r["timestamp"] < T1 and r["text"].strip()])
    for record in llm_inputs:
        expected = pick_reviews(train[train["parent_asin"].eq(record["parent_asin"])])
        assert record["reviews"] == expected and len(expected) == 2


def test_duplicate_meta_parent_asin_fails(tmp_path):
    rows = [{"parent_asin": "A", "title": "First"}, {"parent_asin": "A", "title": "Second"}]
    with gzip.open(tmp_path / "meta_Automotive.jsonl.gz", "wt", encoding="utf-8") as file:
        file.writelines(json.dumps(r) + "\n" for r in rows)
    with pytest.raises(ValueError, match="parent_asin"):
        read_meta(tmp_path, {"A"})
