"""Dashboard Streamlit: wyniki (odkrywane ze skanu folderów), koszt LLM, mój profil, słownik. Nie wywołuje LLM."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from . import config
from .config import PATHS, Paths
from .encode import encode, load_encoder, load_variant
from .evaluate import PRIMARY_METRIC
from .glossary import GLOSSARY
from .llm_texts import current_generations
from .recommend import query_scores
from .storage import read_json, read_jsonl
from .two_tower import load_two_tower, two_tower_query

MECHANISM_LABELS = {"tfidf": "TF-IDF", "dense": "Embeddingi (MiniLM)", "two_tower": "Sieć dwuwieżowa",
                    config.POPULARITY: "Popularność"}
# Paleta kategoryczna; przydział wariant -> slot w variant_colors
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PROFILE_MECHANISMS = ("tfidf", "dense", "two_tower")


def label(variant: str) -> str:
    if variant == config.NO_LLM:
        return "bez LLM"
    if variant == config.BASELINE:
        return "popularność"
    return f"z LLM ({variant.removeprefix(config.LLM_PREFIX)})"


def variant_colors(variants) -> dict[str, str]:
    """Stały kolor wariantu: no_llm slot 0, popularność slot 1, llm_* od slotu 2 alfabetycznie (cyklicznie)."""
    llm = sorted(v for v in variants if v.startswith(config.LLM_PREFIX))
    slots = {config.NO_LLM: 0, config.BASELINE: 1, **{v: 2 + i for i, v in enumerate(llm)}}
    return {v: SLOT_COLORS[i % len(SLOT_COLORS)] for v, i in slots.items()}


def mechanism_options(found: pd.DataFrame) -> list[str]:
    """Mechanizmy ze skanu (bez bazowej popularności): znane w stałej kolejności, nieznane alfabetycznie."""
    scanned = set(found.loc[found["variant"] != config.BASELINE, "mechanism"])
    return [m for m in PROFILE_MECHANISMS if m in scanned] + sorted(scanned - set(PROFILE_MECHANISMS))


def discover_results(paths: Paths) -> pd.DataFrame:
    """Jeden wiersz na results/<wariant>/<mechanizm>/metrics.json; nic na sztywno.

    Gdy jest manifest porównań, tylko warianty z ostatniego evaluate (stare foldery wyników są pomijane).
    """
    rows = [{"variant": f.parent.parent.name, "mechanism": f.parent.name, **read_json(f)}
            for f in sorted(paths.results_root.glob("*/*/metrics.json"))]
    manifest = paths.comparisons / "manifest.json"
    if manifest.is_file():
        keep = {*read_json(manifest)["variants"], config.BASELINE}
        rows = [r for r in rows if r["variant"] in keep]
    return pd.DataFrame(rows)


def llm_cost(paths: Paths) -> pd.DataFrame:
    """Czasy generowania opisów (generations.jsonl) i sprzęt (manifest) dla każdego wariantu llm_*."""
    rows = []
    for manifest in sorted(paths.root.glob(f"item_texts/{config.LLM_PREFIX}*/manifest.json")):
        variant, info = manifest.parent.name, read_json(manifest)
        seconds = [r["seconds"] for r in read_jsonl(manifest.parent / "generations.jsonl") if "seconds" in r]
        rows.append({"wariant": variant, "opisów": len(seconds),
                     "średni czas [s]": float(np.mean(seconds)) if seconds else np.nan,
                     "mediana czasu [s]": float(np.median(seconds)) if seconds else np.nan,
                     "sprzęt": info.get("hardware") or "nie podano", "serwer": str(info.get("server", ""))})
    return pd.DataFrame(rows)


def custom_top_n(paths: Paths, variant: str, mechanism: str, text: str, n: int, encoder) -> pd.DataFrame:
    """Top-N produktów dla profilu tekstowego; pusty tekst nie woła enkodera. Tytuł tylko do wyświetlenia."""
    columns = ["parent_asin", "title", "score"]
    if not text.strip():
        return pd.DataFrame(columns=columns)
    data = load_variant(paths, variant)
    items = pd.read_parquet(paths.prepared / "items.parquet", columns=["parent_asin", "title"])
    if data["embeddings"].shape[0] != len(items) or data["tfidf"].shape[0] != len(items):
        raise ValueError(f"Modele wariantu {variant} nie odpowiadają items.parquet; uruchom sysrekmag-encode")
    if mechanism == "tfidf":
        scores = query_scores(data["vectorizer"].transform([text]), data["tfidf"])[0]
    else:
        vector = encode(encoder, [text])
        if mechanism == "dense":
            scores = query_scores(vector, data["embeddings"])[0]
        else:
            scores = two_tower_query(load_two_tower(paths, variant), vector, data["embeddings"])[0]
    order = np.argsort(-scores, kind="stable")[:n]  # remis: niższy parent_asin (items posortowane)
    return items.iloc[order].assign(score=scores[order]).reset_index(drop=True)


@st.cache_resource
def _encoder():
    return load_encoder()


def _results_tab(paths: Paths, found: pd.DataFrame, mechanism: str) -> None:
    table = found[found["mechanism"].isin([mechanism, config.POPULARITY])].copy()
    table["wariant"] = table["variant"].map(label)
    table["mechanizm"] = table["mechanism"].map(lambda m: MECHANISM_LABELS.get(m, m))
    st.subheader("Metryki")
    metrics = [c for c in table.columns if c not in ("variant", "mechanism", "wariant", "mechanizm")]
    st.dataframe(table[["wariant", "mechanizm", *metrics]].set_index("wariant"))
    st.subheader("NDCG@10")
    palette = variant_colors(found["variant"])
    domain = list(table["wariant"])
    colors = [palette[v] for v in table["variant"]]
    st.altair_chart(alt.Chart(table).mark_bar().encode(
        x=alt.X("wariant:N", sort=None, title=None), y=alt.Y(f"{PRIMARY_METRIC}:Q", title="NDCG@10"),
        color=alt.Color("wariant:N", scale=alt.Scale(domain=domain, range=colors), legend=None)),
        use_container_width=True)
    path = paths.comparisons / "comparisons.csv"
    if not path.is_file():
        return
    cmp = pd.read_csv(path)
    cmp = cmp[(cmp["mechanism"] == mechanism) & (cmp["metric"] == PRIMARY_METRIC)]
    if cmp.empty:
        return
    st.subheader("Różnica NDCG@10 względem bez LLM (przedział ufności)")
    cmp = cmp.assign(wariant=cmp["variant"].map(label))
    base = alt.Chart(cmp).encode(y=alt.Y("wariant:N", title=None))
    st.altair_chart(base.mark_rule().encode(x=alt.X("ci_low:Q", title="Δ NDCG@10"), x2="ci_high:Q")
                    + base.mark_point(filled=True, size=80).encode(x="delta:Q")
                    + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="z:Q"),
                    use_container_width=True)
    st.dataframe(cmp.drop(columns=["wariant"]).reset_index(drop=True))


def _profile_tab(paths: Paths, llm: str | None, mechanism: str, n: int) -> None:
    if llm is None:
        st.info("Brak wariantu LLM w wynikach.")
        return
    if mechanism not in PROFILE_MECHANISMS:
        st.info("Dla popularności nie ma profilu tekstowego; wybierz TF-IDF, embeddingi albo sieć dwuwieżową.")
        return
    text = st.text_area("Twój profil", placeholder="Opisz po angielsku, jakie produkty samochodowe lubisz, "
                        "np. bright LED bulbs for headlights", height=120)
    if not text.strip():
        return
    generated = current_generations(paths, llm)
    for column, variant in zip(st.columns(2), (config.NO_LLM, llm), strict=True):
        top = custom_top_n(paths, variant, mechanism, text, n, None if mechanism == "tfidf" else _encoder())
        column.markdown(f"**{label(variant)}**")
        column.dataframe(top.rename(columns={"title": "tytuł"}).set_index("parent_asin"))
        for row in top.itertuples():
            with column.expander(f"Opis LLM: {row.title}"):
                item = generated.get(row.parent_asin)
                st.write(item["description"] if item else "Brak opisu.")


def main() -> None:
    st.set_page_config(page_title="Profile LLM w rekomendacjach", layout="wide")
    st.title("Profile produktów z LLM w rekomendacjach (Amazon Automotive)")
    found = discover_results(PATHS)
    if found.empty:
        st.warning("Brak wyników w data/results/. Uruchom kolejno: sysrekmag-prepare, sysrekmag-llm-texts, "
                   "sysrekmag-encode, sysrekmag-train, sysrekmag-evaluate.")
        return
    llm_variants = sorted(v for v in found["variant"].unique() if v.startswith(config.LLM_PREFIX))
    mechanisms = mechanism_options(found)
    with st.sidebar:
        llm = st.selectbox("Model LLM", llm_variants, format_func=label) if llm_variants else None
        mechanism = st.selectbox("Mechanizm", mechanisms, format_func=lambda m: MECHANISM_LABELS.get(m, m))
        n = st.radio("Liczba rekomendacji N", config.TOP_KS, horizontal=True)
    results, cost, profile, glossary = st.tabs(["Wyniki", "Koszt LLM", "Mój profil", "Słownik"])
    with results:
        _results_tab(PATHS, found, mechanism)
    with cost:
        table = llm_cost(PATHS)
        st.dataframe(table.set_index("wariant")) if not table.empty else st.info("Brak wygenerowanych opisów.")
    with profile:
        _profile_tab(PATHS, llm, mechanism, n)
    with glossary:
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}**: {text}")
