"""Syntetyczne mini-pliki Amazon Reviews 2023 (Automotive) — bez prawdziwych danych i bez LLM."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sys_rek_mag.config import T1, T2, Paths

DAY = 86_400_000
GROUP_WORDS = ["led bulb", "wiper blade", "car wax"]
ITEM_IDS = [f"B{i:09d}" for i in range(30)]
NO_TITLE_ID = "B100000000"  # w 5-core, dużo opinii, ale pusty tytuł w meta -> odrzucony
FEW_REVIEWS_ID = "B100000001"  # 4 opinie treningowe -> poniżej progu
BLANK_ID = "B100000002"  # 4 opinie z tekstem + 1 z tekstem "   " -> poniżej progu
OUTSIDE_5CORE_ID = "B100000003"  # ma meta i dużo opinii, ale nie ma go w 5-core
LATE_5CORE_ID = "B100000004"  # dużo opinii przed T1, ale w 5-core tylko w valid/test -> nie kandydat
BLANK_TITLE_MARKER = "BLANKREVIEWTITLE"  # tytuł opinii z białym tekstem na produkcie z próbki
DUPLICATE_PAIR = ("U_DUP", ITEM_IDS[0])  # pierwsza opinia 5.0 w treningu, późniejsza 1.0 w teście
EVAL_USER = "U000"
VALID_THEN_TEST_USER = "U_VALID"  # ocena w walidacji i ponownie pozytywna w teście -> nie eval
NO_TRAIN_POSITIVE_USER = "U_NEG"  # tylko negatyw w treningu + pozytyw w teście -> nie eval
BOUNDARY_USER = "U_EDGE"  # opinie dokładnie w T1 i w T2
PREPARE_SIZES = {"n_items": 20}


def group_of(index: int) -> int:
    return index % 3


def _write_gz_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def _meta(asin: str, title: str, group: int) -> dict:
    word = GROUP_WORDS[group]
    return {
        "main_category": "Automotive", "title": title, "average_rating": 4.5, "rating_number": 10,
        "features": [f"Bright {word} feature", "Easy install"], "description": [f"Store description of {word}."],
        "price": 9.99, "images": [], "videos": [], "store": "Shop", "categories": ["Automotive", f"{word.title()}s"],
        "details": {}, "parent_asin": asin, "bought_together": None,
    }


def write_raw(root: Path, extra_reviews: list[dict] | None = None) -> Path:
    """Zapisuje pliki z config.AMAZON_FILES; extra_reviews: dict(user, asin, rating, timestamp)."""
    root.mkdir(parents=True, exist_ok=True)
    reviews: list[dict] = []

    def add(user: str, asin: str, rating: float, timestamp: int, text: str | None = None, title: str = "",
            helpful: int = 0) -> None:
        number = len(reviews)
        reviews.append({
            "rating": float(rating), "title": title or f"Review title {number}",
            "text": f"Review {number} about this product." if text is None else text, "images": [],
            "asin": asin, "parent_asin": asin, "user_id": user, "timestamp": int(timestamp),
            "helpful_vote": helpful, "verified_purchase": True,
        })

    train_day = T1 - 400 * DAY
    for index, asin in enumerate(ITEM_IDS):  # wypełniacze: malejąca liczba opinii treningowych
        word = GROUP_WORDS[group_of(index)]
        for k in range(6 + (29 - index) // 2):
            add(f"F{index:02d}_{k:02d}", asin, 5, train_day + k * DAY, f"Good {word}, review {len(reviews)}.",
                helpful=k % 4)
        add(f"FV{index:02d}", asin, 4, T1 + 10 * DAY, f"Valid {word} review {len(reviews)}.")
        add(f"FT{index:02d}", asin, 4, T2 + 10 * DAY, f"Test {word} review {len(reviews)}.")
    for user in range(40):  # użytkownicy w swojej grupie: pozytywy train, negatyw, valid, test
        uid = f"U{user:03d}"
        group = [a for i, a in enumerate(ITEM_IDS[:18]) if group_of(i) == user % 3]
        pick = [group[(user + j) % len(group)] for j in range(6)]
        word = GROUP_WORDS[user % 3]
        for j, asin in enumerate(pick[:3]):
            add(uid, asin, 5, train_day + (j + 30) * DAY, f"I like this {word}, u{user} r{len(reviews)}.")
        add(uid, pick[3], 2, train_day + 40 * DAY, f"Bad {word}, r{len(reviews)}.")
        add(uid, pick[4], 5, T1 + 20 * DAY, f"Valid {word} r{len(reviews)}.")
        add(uid, pick[5], 5, T2 + 20 * DAY, f"Test {word} r{len(reviews)}.")
    for k in range(10):  # NO_TITLE i OUTSIDE_5CORE mają dużo opinii treningowych
        add(f"NT{k}", NO_TITLE_ID, 5, train_day + k * DAY)
        add(f"NC{k}", OUTSIDE_5CORE_ID, 5, train_day + k * DAY)
    for k in range(40):
        add(f"NL{k:02d}", LATE_5CORE_ID, 5, train_day + k * DAY)
    for k in range(4):
        add(f"FR{k}", FEW_REVIEWS_ID, 5, train_day + k * DAY)
        add(f"BL{k}", BLANK_ID, 5, train_day + k * DAY)
    add("BL_BLANK", BLANK_ID, 5, train_day + 5 * DAY, "   ")
    add("BL_SEL", ITEM_IDS[1], 5, train_day + 50 * DAY, "   ", title=BLANK_TITLE_MARKER, helpful=99)
    add(*DUPLICATE_PAIR, 5, train_day + 60 * DAY, "First opinion.")
    add(*DUPLICATE_PAIR, 1, T2 + 30 * DAY, "Changed my mind.")
    add(VALID_THEN_TEST_USER, ITEM_IDS[2], 5, train_day + 60 * DAY, "Train positive.")
    add(VALID_THEN_TEST_USER, ITEM_IDS[5], 3, T1 + 30 * DAY, "Valid opinion.")
    add(VALID_THEN_TEST_USER, ITEM_IDS[5], 5, T2 + 30 * DAY, "Test opinion again.")
    add(NO_TRAIN_POSITIVE_USER, ITEM_IDS[3], 2, train_day + 60 * DAY, "Train negative.")
    add(NO_TRAIN_POSITIVE_USER, ITEM_IDS[6], 5, T2 + 30 * DAY, "Test positive.")
    add(BOUNDARY_USER, ITEM_IDS[4], 5, T1 - 1, "Just before T1.")
    add(BOUNDARY_USER, ITEM_IDS[7], 5, T1, "Exactly T1.")
    add(BOUNDARY_USER, ITEM_IDS[8], 5, T2 - 1, "Just before T2.")
    add(BOUNDARY_USER, ITEM_IDS[9], 5, T2, "Exactly T2.")
    for row in extra_reviews or []:
        add(**row)
    _write_gz_jsonl(root / "Automotive.jsonl.gz", reviews)

    meta = [_meta(a, f"{GROUP_WORDS[group_of(i)].title()} model {i}", group_of(i)) for i, a in enumerate(ITEM_IDS)]
    meta += [_meta(NO_TITLE_ID, "", 0), _meta(FEW_REVIEWS_ID, "Led Bulb few", 0),
             _meta(BLANK_ID, "Led Bulb blank", 0), _meta(OUTSIDE_5CORE_ID, "Led Bulb outside", 0),
             _meta(LATE_5CORE_ID, "Led Bulb late", 0)]
    _write_gz_jsonl(root / "meta_Automotive.jsonl.gz", meta)

    core = ITEM_IDS + [NO_TITLE_ID, FEW_REVIEWS_ID, BLANK_ID]
    for split, part in (("train", core), ("valid", [LATE_5CORE_ID, *ITEM_IDS[:3]]), ("test", [LATE_5CORE_ID])):
        with gzip.open(root / f"Automotive.{split}.csv.gz", "wt", encoding="utf-8") as file:
            file.write("user_id,parent_asin,rating,timestamp,history\n")
            for asin in part:
                file.write(f"X,{asin},5.0,{train_day},\n")
    return root


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(tmp_path / "data")


@pytest.fixture
def raw(paths: Paths) -> Path:
    return write_raw(paths.raw)


class FakeClient:
    """Atrapa klienta OpenAI: kolejka odpowiedzi (ostatnia się powtarza) albo responder(messages) -> str.

    prompt_tokens trafia do response.usage (None = serwer nie podaje usage)."""

    base_url = "http://127.0.0.1:9/v1"

    def __init__(self, responses: list[str] | None = None, responder=None, finish_reason: str = "stop",
                 prompt_tokens: int | None = 100):
        assert (responses is None) != (responder is None), "dokładnie jedno z: responses, responder"
        self.responses, self.responder, self.finish_reason, self.calls = list(responses or []), responder, finish_reason, 0
        self.usage = None if prompt_tokens is None else type("U", (), {"prompt_tokens": prompt_tokens})()
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        self.last_kwargs = kw
        if self.responder:
            content = self.responder(kw["messages"])
        else:
            content = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message, "finish_reason": self.finish_reason})()
        return type("R", (), {"choices": [choice], "usage": self.usage})()


class FakeEncoder:
    """Atrapa enkodera: deterministyczny worek słów haszowany (crc32) do 32 wymiarów, L2-normalizowany."""

    DIM = 32

    def __init__(self):
        self.calls = 0
        self.tokenizer = lambda texts: {"input_ids": [t.lower().split() for t in texts]}

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM

    def encode(self, texts, **kw):
        self.calls += 1
        import numpy as np
        import zlib

        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % self.DIM] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
