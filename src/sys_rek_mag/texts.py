"""Stałe szablony tekstów: wejście dla LLM oraz warianty no_llm i llm_<model>."""

from __future__ import annotations


def no_llm_text(item: dict) -> str:
    """Tytuł, cechy i opis sklepu; puste części pomijane."""
    parts = [item["title"], *item["features"], *item["description"]]
    return ". ".join(p.strip() for p in parts if p and p.strip())


def llm_input_text(record: dict) -> str:
    """Opinie klientów (tytuły opinii, nigdy tytuł produktu) jako wejście dla LLM."""
    return "\n".join(
        f"Review {n} ({r['rating']:g}/5 stars): {r['title']}. {r['text']}" for n, r in enumerate(record["reviews"], 1)
    )


def llm_text(generated: dict) -> str:
    """Tekst wariantu LLM: wyłącznie wygenerowany opis i cechy."""
    features = "; ".join(f"{f['aspect']}: {f['value']}" for f in generated["features"])
    return f"{generated['description']} Features: {features}"
