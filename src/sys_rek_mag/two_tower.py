"""Sieć dwuwieżowa (PyTorch, CPU) na zamrożonych embeddingach produktów; osobny model na każdy wariant."""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from . import config
from .config import PATHS, SEED, Paths
from .recommend import mean_profiles, ranking_metrics, top_k
from .storage import file_hash, read_json, write_json

HIDDEN = 256
OUT = 128
TEMPERATURE = 0.05
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 50
PATIENCE = 5
HYPERPARAMETERS = {"hidden": HIDDEN, "out": OUT, "temperature": TEMPERATURE, "lr": LR, "batch": BATCH,
                   "max_epochs": MAX_EPOCHS, "patience": PATIENCE}


def _tower(d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, OUT))


class TwoTower(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.user_tower = _tower(d)
        self.item_tower = _tower(d)

    def forward(self, user_inputs: torch.Tensor, item_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.user_tower(user_inputs), dim=1), F.normalize(self.item_tower(item_inputs), dim=1)


def _matrix(rows: pd.Series, cols: pd.Series, shape: tuple[int, int]) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)


def training_data(paths: Paths, variant: str) -> dict[str, Any]:
    """Macierze użytkownik × produkt (kolumny w porządku items.parquet) dla treningu i walidacji."""
    asins = pd.read_parquet(paths.prepared / "items.parquet")["parent_asin"]
    frame = pd.read_parquet(paths.prepared / "interactions.parquet")
    frame = frame[frame["parent_asin"].isin(asins)]
    user_codes, user_ids = pd.factorize(frame["user_id"], sort=True)
    item_codes = pd.Index(asins).get_indexer(frame["parent_asin"])
    shape = (len(user_ids), len(asins))
    positive = frame["rating"].ge(config.POSITIVE_RATING).to_numpy()
    train = frame["split"].eq("train").to_numpy()
    valid = frame["split"].eq("valid").to_numpy()
    history = _matrix(user_codes[positive & train], item_codes[positive & train], shape)
    seen = _matrix(user_codes[train], item_codes[train], shape)
    relevant = _matrix(user_codes[positive & valid], item_codes[positive & valid], shape)
    relevant = (relevant - relevant.multiply(seen)).tocsr()  # cel walidacyjny tylko poza interakcjami z treningu
    relevant.eliminate_zeros()
    seen, relevant = seen.astype(bool), relevant.astype(bool)
    has_history = np.asarray(history.sum(axis=1)).ravel() > 0
    has_target = np.asarray(relevant.sum(axis=1)).ravel() > 0
    return {"embeddings": np.load(paths.models(variant) / "embeddings.npy"), "train_history": history,
            "train_seen": seen, "valid_relevant": relevant, "valid_users": np.flatnonzero(has_history & has_target)}


def _examples(history: sp.csr_matrix, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dla użytkownika z ≥ 2 pozytywami: każdy pozytyw jest celem, wejście = średnia pozostałych."""
    counts = np.diff(history.indptr)
    sums = np.asarray(history @ embeddings, dtype=np.float64)
    users = np.repeat(np.arange(history.shape[0]), counts)
    targets = history.indices
    keep = counts[users] >= 2
    users, targets = users[keep], targets[keep]
    inputs = (sums[users] - embeddings[targets]) / (counts[users] - 1)[:, None]
    return inputs.astype(np.float32), targets


def two_tower_scores(model: TwoTower, history: sp.csr_matrix, embeddings: np.ndarray) -> np.ndarray:
    return _scores(model, mean_profiles(history, embeddings), embeddings)


def two_tower_query(model: TwoTower, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return _scores(model, np.atleast_2d(query_embedding), embeddings)


def _scores(model: TwoTower, user_inputs: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        users, items = model(torch.as_tensor(user_inputs, dtype=torch.float32),
                             torch.as_tensor(embeddings, dtype=torch.float32))
    return (users @ items.T).double().numpy()


def _validation_ndcg(model: TwoTower, data: dict[str, Any]) -> float:
    users = data["valid_users"]
    scores = two_tower_scores(model, data["train_history"][users], data["embeddings"])
    top, top_valid = top_k(scores, ~data["train_seen"][users].toarray(), 10)
    return float(np.nanmean(ranking_metrics(top, top_valid, data["valid_relevant"][users].toarray(), ks=(10,))[
        "ndcg_at_10"]))


def _fit(data: dict[str, Any], seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    inputs, targets = _examples(data["train_history"], data["embeddings"])
    if not len(targets) or not len(data["valid_users"]):
        raise ValueError("Za mało danych: brak przykładów treningowych albo użytkowników walidacyjnych")
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = TwoTower(data["embeddings"].shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        user_inputs = torch.from_numpy(inputs)
        item_inputs = torch.from_numpy(data["embeddings"][targets].astype(np.float32))
        curve: list[float] = []
        best, best_epoch, best_state = -np.inf, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in torch.randperm(len(targets), generator=generator).split(BATCH):
                users, items = model(user_inputs[batch], item_inputs[batch])
                loss = F.cross_entropy(users @ items.T / TEMPERATURE, torch.arange(len(batch)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            curve.append(_validation_ndcg(model, data))
            if curve[-1] > best:
                best, best_epoch, best_state = curve[-1], epoch, copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= PATIENCE:
                break
    finally:
        torch.use_deterministic_algorithms(deterministic)
    return best_state, {"validation_ndcg_at_10": curve, "best_epoch": best_epoch, "best_ndcg_at_10": best,
                        "train_examples": len(targets), "valid_users": len(data["valid_users"])}


def train_variant(paths: Paths, variant: str, seed: int = SEED) -> dict[str, Any]:
    out = paths.models(variant)
    manifest_path, weights_path = out / "two_tower_manifest.json", out / "two_tower.pt"
    fingerprint = {"embeddings": file_hash(out / "embeddings.npy"),
                   "interactions": file_hash(paths.prepared / "interactions.parquet"),
                   "seed": seed, "hyperparameters": HYPERPARAMETERS}
    if manifest_path.is_file() and weights_path.is_file() and read_json(manifest_path).get("fingerprint") == fingerprint:
        return read_json(manifest_path)
    print(f"Trenuję sieć dwuwieżową dla wariantu {variant}")
    state, info = _fit(training_data(paths, variant), seed)
    temporary = weights_path.with_name(f".{weights_path.name}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, weights_path)
    manifest = {"fingerprint": fingerprint, "hyperparameters": HYPERPARAMETERS, "seed": seed, **info}
    write_json(manifest, manifest_path)
    print(f"{variant}: najlepsza epoka {info['best_epoch']}, NDCG@10 walidacja {info['best_ndcg_at_10']:.4f}")
    return manifest


def load_two_tower(paths: Paths, variant: str) -> TwoTower:
    state = torch.load(paths.models(variant) / "two_tower.pt", weights_only=True)
    model = TwoTower(state["user_tower.0.weight"].shape[1])
    model.load_state_dict(state)
    return model.eval()


def main() -> None:
    argparse.ArgumentParser(description="Trening sieci dwuwieżowej dla każdego wariantu z embeddingami").parse_args()
    for path in sorted((PATHS.root / "models").glob("*/embeddings.npy")):
        train_variant(PATHS, path.parent.name)


if __name__ == "__main__":
    main()
