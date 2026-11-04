"""Atomowe zapisy i fingerprinty plików (cache używany tylko przy zgodnych fingerprintach)."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp")


def write_json(payload: Any, path: Path) -> None:
    temporary = _temporary(path)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    temporary = _temporary(path)
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary(path)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary(path)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_npy(array: np.ndarray, path: Path) -> None:
    temporary = _temporary(path)
    with temporary.open("wb") as file:
        np.save(file, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_pickle(obj: Any, path: Path) -> None:
    temporary = _temporary(path)
    with temporary.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)
