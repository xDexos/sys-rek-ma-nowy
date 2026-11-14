"""Dashboard Streamlit: wyniki dla wybranego modelu LLM i mechanizmu, mój profil, koszt LLM, słownik. Nie wywołuje LLM.

Panel boczny: okres oceny, model LLM, mechanizm. Strona główna pokazuje wszystko dla tego wyboru;
wykresy budują czyste funkcje z charts.py, tabele czyste funkcje poniżej (testowalne bez Streamlita).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import charts, config
from .charts import metric_label, variant_colors  # noqa: F401  (variant_colors: część API dashboardu)
from .config import PATHS, Paths
from .encode import encode, load_encoder, load_variant
from .evaluate import METRICS, PRIMARY_METRIC
from .glossary import GLOSSARY
from .llm_texts import current_generations
from .recommend import query_scores
from .storage import read_json, read_jsonl
from .two_tower import load_two_tower, two_tower_query

# Wersja panelu widoczna w panelu bocznym; każda zmiana dashboardu podnosi ją o 0.1 (1.0 -> 1.1 -> ...)
DASHBOARD_VERSION = "1.0"

MECHANISM_LABELS = {"tfidf": "TF-IDF", "dense": "Embeddingi (MiniLM)", "two_tower": "Sieć dwuwieżowa",
                    config.POPULARITY: "Popularność"}
MECHANISM_HELP = {
    "tfidf": "dopasowanie słów: poleca produkty, których tekst zawiera te same charakterystyczne słowa "
             "co produkty kupione wcześniej przez klienta.",
    "dense": "dopasowanie znaczenia: model MiniLM zamienia tekst na wektor, więc „LED bulb” i „LED lamp” "
             "są blisko siebie, choć mają różne słowa.",
    "two_tower": "mała sieć neuronowa uczona na historii zakupów; dostaje wektory tekstów z MiniLM.",
}
PROFILE_MECHANISMS = ("tfidf", "dense", "two_tower")
SPLIT_LABELS = {"test": "Test (wynik końcowy)", "valid": "Walidacja (strojenie)"}
SPLIT_HELP = ("Na jakich zakupach sprawdzamy rekomendacje. Modele uczą się na starszych opiniach, a oceniamy je "
              "na późniejszych, których nie widziały. **Walidacja**: środkowy okres, służy do poprawiania promptu. "
              "**Test**: ostatni okres, liczony raz dla wersji ostatecznej; to wynik do pracy.")
METRIC_HELP = {
    "NDCG": "główna metryka: trafienia na pierwszych K miejscach, wyżej = więcej punktów (0–1)",
    "Precyzja": "jaka część K poleconych produktów spodobała się klientowi",
    "Czułość": "jaka część produktów, które klient polubił, znalazła się wśród K poleconych",
    "HitRate": "odsetek klientów z co najmniej jednym trafieniem wśród K poleconych",
    "MRR": "średnia z 1 / pozycja pierwszego trafienia (trafienie na 4. miejscu = 0,25)",
}
CONCLUSION_ANSWER = {"poprawa": "Tak, istotna poprawa", "pogorszenie": "Nie, istotne pogorszenie",
                     "brak istotnej różnicy": "Brak istotnej różnicy"}


def label(variant: str) -> str:
    if variant == config.NO_LLM:
        return "Tekst sklepu (bez LLM)"
    if variant == config.BASELINE:
        return "Najpopularniejsze produkty"
    return f"Opis z LLM ({variant.removeprefix(config.LLM_PREFIX)})"


def intro(llm: str | None) -> str:
    """Ramka „Co tu porównujemy” nad wynikami."""
    llm_line = (f"- **{label(llm)}**: wyłącznie opis i cechy napisane przez model językowy na podstawie opinii "
                "klientów (model nie widzi tytułu ani opisu sklepu).\n" if llm else "")
    return ("**Co tu porównujemy.** Ten sam algorytm rekomendacji dostaje o produktach różny tekst:\n\n"
            "- **Tekst sklepu (bez LLM)**: tytuł, cechy i opis ze sklepu Amazon.\n" + llm_line +
            "- **Najpopularniejsze produkty**: punkt odniesienia bez tekstu; każdy klient dostaje te same, "
            "najczęściej kupowane produkty. Dobry algorytm powinien go pokonać.\n\n"
            "Pytanie badawcze: czy opis z LLM daje lepsze rekomendacje niż tekst sklepu?")


def pl_number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def pl_percent(value: float) -> str:
    return f"{value * 100:+.1f}%".replace(".", ",")  # zwykły "-": po nim st.metric wybiera kolor i strzałkę


def mechanism_options(found: pd.DataFrame) -> list[str]:
    """Mechanizmy ze skanu (bez bazowej popularności): znane w stałej kolejności, nieznane alfabetycznie."""
    scanned = set(found.loc[found["variant"] != config.BASELINE, "mechanism"])
    return [m for m in PROFILE_MECHANISMS if m in scanned] + sorted(scanned - set(PROFILE_MECHANISMS))


def comparisons_path(paths: Paths, split: str) -> Path:
    return (paths.results_valid if split == "valid" else paths.comparisons) / "comparisons.csv"


def discover_results(paths: Paths, split: str = "test") -> pd.DataFrame:
    """Test: jeden wiersz na results/<wariant>/<mechanizm>/metrics.json; nic na sztywno.

    Gdy jest manifest porównań, tylko warianty z ostatniego evaluate (stare foldery wyników są pomijane).
    Walidacja: results_valid/metrics.csv (nadpisywany w całości przez każde evaluate --split valid).
    """
    if split == "valid":
        path = paths.results_valid / "metrics.csv"
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()
    rows = [{"variant": f.parent.parent.name, "mechanism": f.parent.name, **read_json(f)}
            for f in sorted(paths.results_root.glob("*/*/metrics.json"))]
    manifest = paths.comparisons / "manifest.json"
    if manifest.is_file():
        keep = {*read_json(manifest)["variants"], config.BASELINE}
        rows = [r for r in rows if r["variant"] in keep]
    return pd.DataFrame(rows)


def load_comparisons(paths: Paths, split: str) -> pd.DataFrame:
    path = comparisons_path(paths, split)
    return pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["variant", "mechanism", "metric"])


def selected_rows(found: pd.DataFrame, llm: str | None, mechanism: str) -> pd.DataFrame:
    """Wiersze do wykresów: bez LLM i wybrany LLM dla mechanizmu, potem popularność (punkt odniesienia)."""
    picks = [(config.NO_LLM, mechanism), (llm, mechanism), (config.BASELINE, config.POPULARITY)]
    parts = [found[(found["variant"] == v) & (found["mechanism"] == m)] for v, m in picks if v]
    rows = pd.concat(parts, ignore_index=True)
    return rows.assign(wariant=rows["variant"].map(label))


def mechanism_rows(found: pd.DataFrame, llm: str | None) -> pd.DataFrame:
    """Wszystkie mechanizmy (bez popularności) dla bez LLM i wybranego LLM, w stałej kolejności."""
    variants = [v for v in (config.NO_LLM, llm) if v]
    order = {m: i for i, m in enumerate(mechanism_options(found))}
    rows = found[found["variant"].isin(variants) & found["mechanism"].isin(order)]
    rows = rows.assign(_v=rows["variant"].map(variants.index), _m=rows["mechanism"].map(order))
    rows = rows.sort_values(["_m", "_v"]).drop(columns=["_v", "_m"]).reset_index(drop=True)
    return rows.assign(wariant=rows["variant"].map(label))


def results_table(rows: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """Tabela do pracy: metryka × wariant, zmiana względem bez LLM i przedział ufności (liczby, bez formatowania)."""
    table = rows.set_index("wariant")[[m for m in METRICS if m in rows]].T
    table.columns.name = None
    table = table.assign(**{"Zmiana [%]": np.nan, "Przedział od [%]": np.nan, "Przedział do [%]": np.nan,
                            "Poziom ufności [%]": np.nan, "Wniosek": ""})
    for row in cmp.itertuples():
        if row.metric in table.index:
            table.loc[row.metric, ["Zmiana [%]", "Przedział od [%]", "Przedział do [%]", "Poziom ufności [%]"]] = [
                row.delta_relative * 100, row.ci_low / row.baseline_mean * 100,
                row.ci_high / row.baseline_mean * 100, row.confidence * 100]
            table.loc[row.metric, "Wniosek"] = row.conclusion
    table.index = table.index.map(metric_label)
    table.index.name = "Metryka"
    return table.dropna(axis=1, how="all").loc[:, lambda t: (t != "").any()]


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


def _summary(rows: pd.DataFrame, cmp: pd.DataFrame, llm: str | None) -> None:
    value = rows.set_index("variant")[PRIMARY_METRIC]
    primary = cmp[cmp["metric"] == PRIMARY_METRIC]
    st.markdown(f"**Wynik w skrócie.** {metric_label(PRIMARY_METRIC)} to jakość pierwszych 10 rekomendacji "
                "(0–1, więcej = lepiej; wartości rzędu 0,02 są typowe przy 1500 produktach).")
    tiles = st.columns(4)
    tiles[0].metric(label(config.NO_LLM), pl_number(value[config.NO_LLM]), border=True,
                    help=f"{metric_label(PRIMARY_METRIC)} dla tekstu sklepu.")
    if llm in value:
        delta = pl_percent(primary["delta_relative"].iloc[0]) if not primary.empty else None
        tiles[1].metric(label(llm), pl_number(value[llm]), delta, border=True,
                        help=f"{metric_label(PRIMARY_METRIC)} dla opisu z LLM; pod spodem zmiana względem tekstu sklepu.")
    if config.BASELINE in value:
        tiles[2].metric(label(config.BASELINE), pl_number(value[config.BASELINE]), border=True,
                        help="Punkt odniesienia: wszyscy dostają te same, najczęściej kupowane produkty.")
    if not primary.empty:
        row = primary.iloc[0]
        tiles[3].metric("Czy LLM pomaga?", CONCLUSION_ANSWER.get(row["conclusion"], row["conclusion"]),
                        f"przedział ufności różnicy: {pl_number(row['ci_low'])} … {pl_number(row['ci_high'])}",
                        delta_color="off", delta_arrow="off", border=True,
                        help="Sparowany bootstrap, poziom 98,33%. Poprawa tylko wtedy, gdy cały przedział "
                             "różnicy jest powyżej zera; jeśli obejmuje zero, różnica może być przypadkowa.")


def _results_tab(paths: Paths, found: pd.DataFrame, llm: str | None, mechanism: str, split: str) -> None:
    rows = selected_rows(found, llm, mechanism)
    all_cmp = load_comparisons(paths, split)
    cmp = all_cmp[(all_cmp["variant"] == llm) & (all_cmp["mechanism"] == mechanism)]
    users = int(cmp["users"].iloc[0]) if "users" in cmp and not cmp.empty else None
    st.info(intro(llm))
    st.subheader(f"Mechanizm: {MECHANISM_LABELS.get(mechanism, mechanism)}")
    st.caption(f"{MECHANISM_LABELS.get(mechanism, mechanism)}: {MECHANISM_HELP.get(mechanism, '')} "
               f"Dane do oceny: {SPLIT_LABELS[split].lower()}"
               + (f", {users:,} klientów.".replace(",", " ") if users else "."))
    _summary(rows, cmp, llm)

    st.markdown("#### Wszystkie metryki")
    st.markdown("Każdy panel to jedna miara jakości listy K poleconych produktów; słupki to warianty tekstu. "
                "Wyższy słupek = lepsze rekomendacje.")
    k = st.segmented_control("Długość listy rekomendacji K", config.TOP_KS, default=max(config.TOP_KS),
                             format_func=lambda v: f"K = {v}") or max(config.TOP_KS)
    metrics = [m for m in METRICS if m.endswith(f"_at_{k}")]
    st.altair_chart(charts.metric_bars(rows, metrics), width="content", theme=None)
    st.caption("Średnia po klientach. Każdy panel ma własną skalę osi pionowej, więc porównuj słupki w panelu, "
               "nie między panelami.")
    with st.expander("Co oznaczają metryki"):
        st.markdown("\n".join(f"- **{name}@K**: {text}" for name, text in METRIC_HELP.items()))

    if not cmp.empty:
        st.markdown("#### O ile opis z LLM zmienia wynik")
        st.markdown(f"Zmiana każdej metryki po zastąpieniu tekstu sklepu opisem z LLM, w procentach "
                    f"(mechanizm: {MECHANISM_LABELS.get(mechanism, mechanism)}). Na prawo od zera = LLM lepszy, "
                    "na lewo = gorszy. Kolor mówi, czy zmiana jest istotna statystycznie.")
        st.altair_chart(charts.delta_forest(cmp), width="content", theme=None)
        st.caption("Kropka: zmiana średniej. Kreska: przedział ufności (zakres, w którym prawdopodobnie leży "
                   "prawdziwa zmiana). Jeśli kreska przecina zero, różnica może być przypadkowa. Metryka główna "
                   "na poziomie 98,33% (poprawka Bonferroniego na 3 porównania), pozostałe 95%.")

    st.markdown(f"#### Wszystkie mechanizmy naraz ({metric_label(PRIMARY_METRIC)})")
    st.markdown("Ten sam wynik dla każdego algorytmu rekomendacji, niezależnie od wyboru w panelu bocznym. "
                "Pokazuje, przy którym algorytmie opis z LLM wypada najlepiej, a przy którym najgorzej.")
    popularity = found[found["variant"] == config.BASELINE][PRIMARY_METRIC]
    st.altair_chart(charts.mechanism_bars(mechanism_rows(found, llm), PRIMARY_METRIC, MECHANISM_LABELS,
                                          float(popularity.iloc[0]) if not popularity.empty else None),
                    width="content", theme=None)
    st.caption("Linia przerywana: najpopularniejsze produkty (punkt odniesienia bez tekstu). "
               "Słupek poniżej linii oznacza, że algorytm przegrywa z polecaniem wszystkim tego samego.")

    st.markdown("#### Tabela wyników")
    st.markdown("Te same liczby co na wykresach, do przepisania do pracy. Zmiana i przedział w % względem tekstu sklepu.")
    table = results_table(rows, cmp)
    numeric = [c for c in table.columns if c != "Wniosek"]
    fmt = {c: (lambda v: "" if pd.isna(v) else pl_number(v, 1)) if "[%]" in c else
           (lambda v: "" if pd.isna(v) else pl_number(v)) for c in numeric}
    st.dataframe(table.style.format(fmt), width="stretch")
    st.download_button("Pobierz tabelę (CSV)", table.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"wyniki_{split}_{llm or config.NO_LLM}_{mechanism}.csv", mime="text/csv")


def _profile_tab(paths: Paths, llm: str | None, mechanism: str) -> None:
    if llm is None:
        st.info("Brak wariantu LLM w wynikach.")
        return
    if mechanism not in PROFILE_MECHANISMS:
        st.info("Dla popularności nie ma profilu tekstowego; wybierz TF-IDF, embeddingi albo sieć dwuwieżową.")
        return
    st.caption(f"Mechanizm: {MECHANISM_LABELS.get(mechanism, mechanism)}. To demonstracja działania, nie metryka.")
    n = st.radio("Liczba rekomendacji N", config.TOP_KS, horizontal=True)
    text = st.text_area("Twój profil", placeholder="Opisz po angielsku, jakie produkty samochodowe lubisz, "
                        "np. bright LED bulbs for headlights", height=120)
    if not text.strip():
        return
    generated = current_generations(paths, llm)
    for column, variant in zip(st.columns(2), (config.NO_LLM, llm), strict=True):
        top = custom_top_n(paths, variant, mechanism, text, n, None if mechanism == "tfidf" else _encoder())
        column.markdown(f"**{label(variant)}**")
        column.dataframe(top.rename(columns={"title": "tytuł", "score": "wynik"}).set_index("parent_asin"))
        for row in top.itertuples():
            with column.expander(f"Opis LLM: {row.title}"):
                item = generated.get(row.parent_asin)
                st.write(item["description"] if item else "Brak opisu.")


def main() -> None:
    st.set_page_config(page_title="LLM w rekomendacjach", layout="wide")
    splits = list(SPLIT_LABELS)
    # domyślnie pierwszy okres, który ma wyniki (np. sama walidacja w trakcie strojenia promptu)
    default = next((i for i, s in enumerate(splits) if not discover_results(PATHS, s).empty), 0)
    with st.sidebar:
        st.header("Wybór")
        st.caption(f"Wersja panelu {DASHBOARD_VERSION}")
        split = st.selectbox("Dane do oceny", splits, index=default, format_func=SPLIT_LABELS.get, help=SPLIT_HELP)
    found = discover_results(PATHS, split)
    st.title("Opisy produktów z LLM a jakość rekomendacji")
    st.caption("Amazon Reviews 2023, kategoria Automotive. Porównanie tekstu sklepu (bez LLM) "
               "z opisem i cechami wygenerowanymi przez lokalny LLM wyłącznie z opinii klientów.")
    if found.empty:
        command = "sysrekmag-evaluate --split valid" if split == "valid" else "sysrekmag-evaluate"
        st.warning(f"Brak wyników ({SPLIT_LABELS[split]}) w {PATHS.root}. Uruchom kolejno: sysrekmag-prepare, "
                   f"sysrekmag-llm-texts, sysrekmag-encode, sysrekmag-train, {command}.")
        return
    llm_variants = sorted(v for v in found["variant"].unique() if v.startswith(config.LLM_PREFIX))
    with st.sidebar:
        llm = st.selectbox("Model LLM", llm_variants, format_func=lambda v: v.removeprefix(config.LLM_PREFIX),
                           help="Model językowy, który napisał opisy produktów z opinii klientów.") if llm_variants else None
        mechanism = st.selectbox("Mechanizm rekomendacyjny", mechanism_options(found),
                                 format_func=lambda m: MECHANISM_LABELS.get(m, m),
                                 help="Algorytm, który na podstawie tekstu o produktach układa listę poleceń. "
                                      + " ".join(f"**{MECHANISM_LABELS[m]}**: {t}" for m, t in MECHANISM_HELP.items()))
        st.divider()
        st.caption(f"Dane: `{PATHS.root}`. Tryb jasny/ciemny: menu ⋮ → Settings → Theme.")
    results, profile, cost, glossary = st.tabs(["Wyniki", "Mój profil", "Koszt LLM", "Słownik"])
    with results:
        _results_tab(PATHS, found, llm, mechanism, split)
    with profile:
        _profile_tab(PATHS, llm, mechanism)
    with cost:
        table = llm_cost(PATHS)
        st.dataframe(table.set_index("wariant")) if not table.empty else st.info("Brak wygenerowanych opisów.")
    with glossary:
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}**: {text}")
