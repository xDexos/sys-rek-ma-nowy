"""Etap LLM: opis i cechy produktu generowane przez lokalny LLM wyłącznie z opinii klientów."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import config
from .config import PATHS, Paths
from .storage import read_json, read_jsonl, text_hash, write_json, write_jsonl
from .texts import llm_input_text

SCHEMA_VERSION = 1
MAX_RETRIES = 3
MAX_TOKENS = 1000
CHECKPOINT_EVERY = 10

SYSTEM_PROMPT = """You are a product analyst. You read ONLY customer reviews of one automotive product and write a description of it for a recommender system.
First work out what the product is from the reviews. Then describe what it is and what it is used for, its strengths and weaknesses according to the customers, and who it is for.
Use only information that follows from the reviews; do not guess and do not add facts from outside them.
Do not mention star ratings or popularity (no "best-selling", "highly rated", review counts) in the description.
Return one JSON object with exactly these fields:
- "description": 2-4 sentences, at most 80 words.
- "features": 5-12 objects {"aspect": ..., "value": ...}, short lowercase labels (e.g. aspect "installation", value "easy").
Write everything in English."""


class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspect: str = Field(min_length=1)
    value: str = Field(min_length=1)

    @field_validator("aspect", "value", mode="before")
    @classmethod
    def normalize(cls, value: Any) -> Any:
        return value.lower().strip() if isinstance(value, str) else value


class ItemDescription(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    description: str
    features: list[Feature] = Field(min_length=5, max_length=12)

    @field_validator("description")
    @classmethod
    def word_count(cls, value: str) -> str:
        if not 20 <= len(value.split()) <= 80:
            raise ValueError("description musi mieć 20-80 słów")
        return value


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "item_description", "strict": True, "schema": ItemDescription.model_json_schema()},
}
PROMPT_HASH = text_hash("\n".join([SYSTEM_PROMPT, json.dumps(RESPONSE_FORMAT, sort_keys=True), str(SCHEMA_VERSION)]))


def make_client(base_url: str = config.LLM_BASE_URL, api_key: str = config.LLM_API_KEY):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)


def generate_one(client, record: dict[str, Any], llm_model: str, seed: int = config.SEED) -> tuple[dict, int]:
    """Jeden opis z walidacją; przy błędzie ponowienie z komunikatem walidacji (max MAX_RETRIES prób)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": llm_input_text(record)}]
    error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=llm_model, messages=messages, temperature=0, seed=seed,
            max_tokens=MAX_TOKENS, response_format=RESPONSE_FORMAT,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        try:
            if choice.finish_reason == "length":
                raise ValueError("odpowiedź ucięta limitem tokenów; zwróć krótszy JSON")
            return ItemDescription.model_validate_json(raw).model_dump(), attempt
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:1000]
            messages = [*messages[:2], {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"The JSON is invalid: {error}\nReturn only the corrected JSON object."}]
    raise RuntimeError(f"Brak poprawnego opisu po {MAX_RETRIES} próbach: {error}")


def _get_json(url: str, payload: dict | None = None) -> dict | None:
    try:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except Exception:  # serwer nie ma tego endpointu -> to inny typ serwera
        return None


def server_info(base_url: str, llm_model: str) -> dict[str, Any]:
    """Typ i wersja serwera oraz kwantyzacja, o ile serwer je podaje (bez uruchamiania modelu)."""
    root = base_url.rstrip("/").removesuffix("/v1")
    if version := _get_json(f"{root}/api/version"):
        show = _get_json(f"{root}/api/show", {"model": llm_model}) or {}
        details = show.get("details", {})
        return {"server": "ollama", "version": version.get("version"),
                "quantization": details.get("quantization_level"), "parameter_size": details.get("parameter_size")}
    if version := _get_json(f"{root}/version"):
        return {"server": "vllm", "version": version.get("version"), "quantization": None}
    if model := _get_json(f"{root}/api/v0/models/{llm_model}"):
        return {"server": "lmstudio", "version": None, "quantization": model.get("quantization")}
    return {"server": "unknown", "version": None, "quantization": None}


def _reusable(record: dict | None, source_hash: str, llm_model: str) -> bool:
    return bool(record) and (record["source_hash"], record["prompt_hash"], record["llm_model"]) == (
        source_hash, PROMPT_HASH, llm_model)


def generate(
    llm_model: str,
    paths: Paths = PATHS,
    client=None,
    limit: int | None = None,
    seed: int = config.SEED,
    quantization: str = "",
) -> dict[str, Any]:
    item_ids = set(pd.read_parquet(paths.prepared / "items.parquet", columns=["parent_asin"])["parent_asin"])
    inputs = read_jsonl(paths.prepared / "llm_inputs.jsonl")
    # kolejność losowa (stała), żeby pilotaż --limit nie brał tylko pierwszych w kolejności parent_asin
    order = np.random.default_rng(seed).permutation(len(inputs))
    selected = [inputs[i] for i in order]
    selected = selected if limit is None else selected[:limit]
    out = paths.item_texts(config.llm_variant(config.model_id(llm_model)))
    generations_path, errors_path = out / "generations.jsonl", out / "errors.jsonl"
    records = {r["parent_asin"]: r for r in read_jsonl(generations_path) if r["parent_asin"] in item_ids}
    client = client or make_client()
    failures, seconds = [], []
    hashes = {r["parent_asin"]: text_hash(llm_input_text(r)) for r in inputs}
    todo = [r for r in selected if not _reusable(records.get(r["parent_asin"]), hashes[r["parent_asin"]], llm_model)]
    print(f"Model {llm_model}: do wygenerowania {len(todo)} z {len(selected)} produktów")
    for number, record in enumerate(todo, start=1):
        asin = record["parent_asin"]
        started = time.perf_counter()
        try:
            output, attempts = generate_one(client, record, llm_model, seed)
        except Exception as exc:  # błąd jednego produktu nie przerywa przebiegu; braki uzupełni ponowne uruchomienie
            failures.append({"parent_asin": asin, "error": str(exc)})
            print(f"  BŁĄD {asin}: {exc}")
            continue
        seconds.append(time.perf_counter() - started)
        records[asin] = {
            "parent_asin": asin, "llm_model": llm_model, "source_hash": text_hash(llm_input_text(record)),
            "prompt_hash": PROMPT_HASH, "attempts": attempts, "seconds": round(seconds[-1], 2), "output": output,
        }
        print(f"  [{number}/{len(todo)}] {asin} — {seconds[-1]:.1f} s, próby: {attempts}")
        if number % CHECKPOINT_EVERY == 0:
            write_jsonl([records[k] for k in sorted(records)], generations_path)
    write_jsonl([records[k] for k in sorted(records)], generations_path)
    if failures:
        write_jsonl(failures, errors_path)
    elif errors_path.exists():
        errors_path.unlink()
    valid = {a for a, rec in records.items() if a in hashes and _reusable(rec, hashes[a], llm_model)}
    mean_seconds = float(np.mean(seconds)) if seconds else None
    manifest = {
        "llm_model": llm_model, "model_id": config.model_id(llm_model), "base_url": str(client.base_url),
        "server": server_info(str(client.base_url), llm_model), "quantization_declared": quantization,
        "hardware": config.LLM_HARDWARE, "system_prompt": SYSTEM_PROMPT, "response_format": RESPONSE_FORMAT,
        "prompt_hash": PROMPT_HASH, "temperature": 0, "seed": seed, "max_tokens": MAX_TOKENS,
        "items": len(item_ids), "valid_generations": len(valid), "failed_generations": len(failures),
        "complete": valid == item_ids, "mean_seconds_last_run": mean_seconds,
    }
    write_json(manifest, out / "manifest.json")
    if mean_seconds is not None:
        remaining = len(item_ids) - len(valid)
        print(f"Średnio {mean_seconds:.1f} s/produkt; pozostało {remaining} ≈ {remaining * mean_seconds / 3600:.1f} h")
    if failures:
        raise SystemExit(f"Nieudane produkty: {len(failures)} (zob. {errors_path}); ponowne uruchomienie uzupełni braki")
    return manifest


def current_generations(paths: Paths, variant: str) -> dict[str, dict[str, Any]]:
    """parent_asin -> wynik (description, features) tylko dla generacji ważnych dla bieżącej próbki.

    Ważna = produkt jest w items.parquet, a skrót aktualnego wejścia z llm_inputs.jsonl, PROMPT_HASH i model
    zgadzają się z rekordem. Wariant jest kompletny, gdy wynik obejmuje wszystkie produkty próbki.
    """
    manifest = paths.item_texts(variant) / "manifest.json"
    if not manifest.is_file():
        return {}
    llm_model = read_json(manifest).get("llm_model")
    item_ids = set(pd.read_parquet(paths.prepared / "items.parquet", columns=["parent_asin"])["parent_asin"])
    hashes = {r["parent_asin"]: text_hash(llm_input_text(r)) for r in read_jsonl(paths.prepared / "llm_inputs.jsonl")
              if r["parent_asin"] in item_ids}
    records = read_jsonl(paths.item_texts(variant) / "generations.jsonl")
    return {r["parent_asin"]: r["output"] for r in records
            if r["parent_asin"] in hashes and _reusable(r, hashes[r["parent_asin"]], llm_model)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuj opisy i cechy produktów lokalnym LLM (UWAGA: uruchamia LLM)")
    parser.add_argument("--model", required=True, help="nazwa modelu na serwerze, np. gemma4:e2b")
    parser.add_argument("--base-url", default=config.LLM_BASE_URL)
    parser.add_argument("--limit", type=int, help="tylko N produktów (pilotaż)")
    parser.add_argument("--quantization", default="", help="kwantyzacja, gdy serwer jej nie podaje (np. FP8 w vLLM)")
    args = parser.parse_args()
    generate(args.model, client=make_client(args.base_url), limit=args.limit, quantization=args.quantization)


if __name__ == "__main__":
    main()
