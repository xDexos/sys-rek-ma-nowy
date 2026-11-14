"""Jedno źródło ścieżek i stałych protokołu."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file(PROJECT_ROOT / ".env")

# Inny katalog danych (np. data/demo do pracy nad dashboardem); ścieżka względna liczona od katalogu projektu
DATA_DIR = PROJECT_ROOT / os.getenv("SYSREKMAG_DATA_DIR", "data")

# --- dane ---
AMAZON_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
_BENCH = "benchmark/5core/timestamp_w_his"
AMAZON_FILES: dict[str, str] = {
    "Automotive.jsonl.gz": f"{AMAZON_BASE}/raw/review_categories/Automotive.jsonl.gz",
    "meta_Automotive.jsonl.gz": f"{AMAZON_BASE}/raw/meta_categories/meta_Automotive.jsonl.gz",
    **{f"Automotive.{s}.csv.gz": f"{AMAZON_BASE}/{_BENCH}/Automotive.{s}.csv.gz" for s in ("train", "valid", "test")},
}

# --- protokół ---
SEED = 2026
T1 = 1628643414042  # koniec train (ms)
T2 = 1658002729837  # początek test (ms)
POSITIVE_RATING = 4.0
N_ITEMS = 1500
MIN_ITEM_TRAIN_REVIEWS = 5
MAX_REVIEWS_PER_ITEM = 20
REVIEW_MAX_CHARS = 800
MIN_EVAL_USERS_WARN = 500
TOP_KS = (5, 10)
BOOTSTRAP_RESAMPLES = 10_000
FAMILY_ALPHA = 0.05
PRIMARY_COMPARISONS_PER_MODEL = 3
SUPPORTING_CONFIDENCE = 0.95
NO_LLM = "no_llm"
LLM_PREFIX = "llm_"
MECHANISMS = ("tfidf", "dense", "two_tower")
BASELINE = "baseline"
POPULARITY = "popularity"

# --- teksty i enkoder ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_SEQ_LENGTH = 512

# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODELS = tuple(n.strip() for n in os.getenv("LLM_MODELS", "gemma4:e2b").split(",") if n.strip())
LLM_HARDWARE = os.getenv("LLM_HARDWARE", "")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def model_id(llm_model: str) -> str:
    """Bezpieczna nazwa katalogu dla modelu LLM, np. 'gemma4:e2b' -> 'gemma4_e2b'."""
    normalized = _UNSAFE.sub("_", llm_model.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"Niepoprawna nazwa modelu LLM: {llm_model!r}")
    return normalized


def llm_variant(llm_model_id: str) -> str:
    return LLM_PREFIX + llm_model_id


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def prepared(self) -> Path:
        return self.root / "prepared"

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    @property
    def results_valid(self) -> Path:
        return self.root / "results_valid"

    @property
    def comparisons(self) -> Path:
        return self.root / "results" / "comparisons"

    def item_texts(self, variant: str) -> Path:
        return self.root / "item_texts" / variant

    def models(self, variant: str) -> Path:
        return self.root / "models" / variant

    def results(self, variant: str, mechanism: str) -> Path:
        return self.root / "results" / variant / mechanism


PATHS = Paths(DATA_DIR)
