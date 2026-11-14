"""Wykresy dashboardu (Altair) w stylu publikacyjnym: czyste funkcje DataFrame -> wykres, do zrzutu do pracy.

Styl: białe tło niezależnie od motywu strony, czcionka szeryfowa, osie w kształcie „L”, przedziały ufności
z poprzeczkami. Słupki w odcieniach szarości z czarnym obrysem, punkt odniesienia biały; wniosek kodowany
kształtem i wypełnieniem znacznika. Wszystko czytelne w druku czarno-białym.
Liczby z przecinkiem dziesiętnym (locale pl). Dashboard rysuje je z theme=None, żeby motyw Streamlita
nie nadpisywał stylu.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from . import config

FONT = "Times New Roman, TeX Gyre Termes, Liberation Serif, serif"
INK = "#000000"
# Odcienie szarości z czarnym obrysem (styl prac dyplomowych, czytelny w druku czarno-białym):
# tekst sklepu ciemny, warianty LLM jaśniejsze, popularność (punkt odniesienia) biała
SLOT_COLORS = ["#404040", "#a6a6a6", "#d4d4d4", "#737373", "#ececec"]
REFERENCE_COLOR = "#ffffff"
CONCLUSIONS = {  # wniosek -> (wypełnienie, kształt znacznika); kształt niesie znaczenie, wypełnienie je wzmacnia
    "poprawa": ("#000000", "triangle-up"),
    "brak istotnej różnicy": ("#a6a6a6", "circle"),
    "pogorszenie": ("#ffffff", "triangle-down"),
}
METRIC_NAMES = {"ndcg": "NDCG", "precision": "Precyzja", "recall": "Czułość", "hit_rate": "HitRate", "mrr": "MRR"}
PL_LOCALE = {"number": {"decimal": ",", "thousands": " ", "grouping": [3], "currency": ["", ""]}}


def metric_label(metric: str) -> str:
    """ndcg_at_10 -> NDCG@10."""
    name, k = metric.rsplit("_at_", 1)
    return f"{METRIC_NAMES.get(name, name)}@{k}"


def variant_colors(variants) -> dict[str, str]:
    """Stały kolor wariantu: no_llm slot 0, llm_* od slotu 1 alfabetycznie (cyklicznie), popularność szara."""
    llm = sorted({v for v in variants if v.startswith(config.LLM_PREFIX)})  # zbiór: wiersze się powtarzają
    colors = {config.NO_LLM: SLOT_COLORS[0], config.BASELINE: REFERENCE_COLOR}
    return colors | {v: SLOT_COLORS[1 + i % (len(SLOT_COLORS) - 1)] for i, v in enumerate(llm)}


def _finish(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    text = {"labelFont": FONT, "titleFont": FONT, "labelColor": INK, "titleColor": INK}
    return (chart.configure(locale=PL_LOCALE, font=FONT, background="white", padding=10)
            .configure_view(stroke=None)  # osie w kształcie „L”: lewa i dolna, bez ramki nakładającej się na siatkę
            .configure_axis(**text, labelFontSize=13, titleFontSize=14, titleFontWeight="normal",
                            domain=True, domainColor=INK, domainWidth=1, tickColor=INK, ticks=True, tickSize=5,
                            grid=True, gridColor="#d9d9d9", gridDash=[2, 3])
            .configure_axisX(grid=False)
            .configure_header(**text, labelFontSize=14, titleFontSize=14)
            .configure_legend(**text, labelFontSize=13, titleFontSize=13, titleFontWeight="normal",
                              orient="bottom", direction="horizontal", strokeColor=INK, padding=8,
                              symbolStrokeColor=INK, symbolStrokeWidth=0.8)
            .configure_text(font=FONT, color=INK))


def _variant_scale(rows: pd.DataFrame) -> alt.Scale:
    palette = variant_colors(rows["variant"])
    order = list(dict.fromkeys(rows["variant"]))
    return alt.Scale(domain=list(dict.fromkeys(rows["wariant"])), range=[palette[v] for v in order])


def metric_bars(rows: pd.DataFrame, metrics: list[str]) -> alt.TopLevelMixin:
    """Małe wykresy: jeden panel na metrykę (własna oś Y), słupek na wariant.

    rows: kolumny variant, wariant (etykieta) i metryki; kolejność wierszy = kolejność słupków.
    """
    long = rows.melt(id_vars=["variant", "wariant"], value_vars=metrics, var_name="metric", value_name="value")
    long["metryka"] = long["metric"].map(metric_label)
    order = list(dict.fromkeys(rows["wariant"]))
    base = alt.Chart(long).encode(
        x=alt.X("wariant:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format=".3f", tickCount=5),
                scale=alt.Scale(nice=True, zero=True)),
    )
    bars = base.mark_bar(stroke=INK, strokeWidth=0.8, width={"band": 0.62}).encode(
        color=alt.Color("wariant:N", scale=_variant_scale(rows), sort=order, title=None),
        tooltip=[alt.Tooltip("wariant:N", title="Wariant"), alt.Tooltip("metryka:N", title="Metryka"),
                 alt.Tooltip("value:Q", title="Wartość", format=".4f")],
    )
    labels = base.mark_text(dy=-7, fontSize=11).encode(text=alt.Text("value:Q", format=".4f"))
    return _finish((bars + labels).properties(width=150, height=230).facet(
        column=alt.Column("metryka:N", sort=[metric_label(m) for m in metrics], title=None,
                          header=alt.Header(labelOrient="top", labelPadding=6))
    ).resolve_scale(y="independent").configure_facet(spacing=18))


def delta_forest(cmp: pd.DataFrame) -> alt.TopLevelMixin:
    """Względna różnica (wariant LLM − bez LLM) / bez LLM z przedziałem ufności, jedna linia na metrykę.

    cmp: wiersze comparisons.csv jednego wariantu i mechanizmu.
    """
    data = cmp.assign(
        metryka=cmp["metric"].map(metric_label) + cmp["role"].map({"primary": " (główna)"}).fillna(""),
        delta_pct=cmp["delta_relative"] * 100,
        low_pct=cmp["ci_low"] / cmp["baseline_mean"] * 100,
        high_pct=cmp["ci_high"] / cmp["baseline_mean"] * 100,
        poziom=(cmp["confidence"] * 100).round(2).map(lambda c: f"{c:g}%".replace(".", ",")),
    )
    order = list(data["metryka"])
    y = alt.Y("metryka:N", sort=order, title=None, axis=alt.Axis(grid=True, gridDash=[2, 3]))
    domain = list(CONCLUSIONS)
    color = alt.Color("conclusion:N", title="Wniosek:", scale=alt.Scale(
        domain=domain, range=[c for c, _ in CONCLUSIONS.values()]))
    shape = alt.Shape("conclusion:N", title="Wniosek:", scale=alt.Scale(
        domain=domain, range=[s for _, s in CONCLUSIONS.values()]))
    tooltip = [alt.Tooltip("metryka:N", title="Metryka"), alt.Tooltip("delta_pct:Q", title="Zmiana [%]", format=".1f"),
               alt.Tooltip("low_pct:Q", title="Przedział od [%]", format=".1f"),
               alt.Tooltip("high_pct:Q", title="Przedział do [%]", format=".1f"),
               alt.Tooltip("poziom:N", title="Poziom ufności"), alt.Tooltip("conclusion:N", title="Wniosek")]
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK, strokeDash=[5, 4], strokeWidth=1).encode(x="z:Q")
    whiskers = alt.Chart(data).mark_errorbar(ticks=alt.MarkConfig(size=9, color=INK), color=INK,
                                             thickness=1.2).encode(
        y=y, x=alt.X("low_pct:Q", title="Zmiana względem tekstu sklepu [%]"), x2="high_pct:Q", tooltip=tooltip)
    points = alt.Chart(data).mark_point(filled=True, size=110, opacity=1, stroke=INK, strokeWidth=0.8).encode(
        y=y, x="delta_pct:Q", color=color, shape=shape, tooltip=tooltip)
    return _finish((zero + whiskers + points).properties(width=640, height=30 * len(data) + 10))


def mechanism_bars(rows: pd.DataFrame, metric: str, mechanism_labels: dict[str, str],
                   baseline: float | None) -> alt.TopLevelMixin:
    """Wybrana metryka dla każdego mechanizmu: słupki wariantów obok siebie, popularność jako linia odniesienia."""
    data = rows.assign(mechanizm=rows["mechanism"].map(lambda m: mechanism_labels.get(m, m)))
    order = list(dict.fromkeys(data["wariant"]))
    mech_order = list(dict.fromkeys(data["mechanizm"]))
    bars = alt.Chart(data).mark_bar(stroke=INK, strokeWidth=0.8).encode(
        x=alt.X("mechanizm:N", sort=mech_order, title="Mechanizm rekomendacyjny",
                axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.3)),
        xOffset=alt.XOffset("wariant:N", sort=order),
        y=alt.Y(f"{metric}:Q", title=metric_label(metric), axis=alt.Axis(format=".3f"),
                scale=alt.Scale(nice=True, zero=True)),
        color=alt.Color("wariant:N", scale=_variant_scale(rows), sort=order, title=None),
        tooltip=[alt.Tooltip("mechanizm:N", title="Mechanizm"), alt.Tooltip("wariant:N", title="Wariant"),
                 alt.Tooltip(f"{metric}:Q", title=metric_label(metric), format=".4f")],
    )
    labels = alt.Chart(data).mark_text(dy=-7, fontSize=11).encode(
        x=alt.X("mechanizm:N", sort=mech_order), xOffset=alt.XOffset("wariant:N", sort=order),
        y=f"{metric}:Q", text=alt.Text(f"{metric}:Q", format=".4f"))
    layers = [bars, labels]
    if baseline is not None:
        ref = pd.DataFrame({"y": [baseline], "opis": [f"najpopularniejsze produkty: {baseline:.4f}".replace(".", ",")]})
        # opis linii w legendzie (przerywana kreska), żeby nie nachodził na etykiety słupków
        layers.append(alt.Chart(ref).mark_rule(color=INK, strokeWidth=1.2).encode(
            y="y:Q", strokeDash=alt.StrokeDash("opis:N", title=None, scale=alt.Scale(range=[[6, 4]]),
                                              legend=alt.Legend(symbolType="stroke", symbolSize=500,
                                                                symbolStrokeWidth=1.2, labelLimit=0))))
    return _finish(alt.layer(*layers).properties(width=640, height=320).resolve_legend(strokeDash="independent"))
