"""Słownik pojęć dashboardu: proste wyjaśnienia z przykładami z kategorii Automotive."""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "wariant": "Jaki tekst o produkcie dajemy algorytmowi. „Bez LLM” to tytuł, cechy i opis ze sklepu. „Z LLM” to wyłącznie opis i lista cech napisane przez model językowy na podstawie opinii klientów.",
    "mechanizm": "Sposób układania rankingu: TF-IDF, embeddingi albo sieć dwuwieżowa. Każdy działa na tekście wybranego wariantu.",
    "TF-IDF": "Zliczanie słów: słowo, które jest w tekście produktu często, a w całym katalogu rzadko, waży najwięcej. Np. „halogen” wskaże żarówki, a „the” nic nie znaczy.",
    "embedding": "Zamiana tekstu na listę liczb tak, by podobne znaczeniowo teksty dawały podobne liczby. „Bright LED bulb” i „white LED lamp” będą blisko siebie, choć mają różne słowa.",
    "sieć dwuwieżowa": "Mała sieć neuronowa z dwiema „wieżami”: jedna zamienia historię użytkownika, druga produkt, na wspólne wektory. Im bardziej wektory są zgodne, tym wyżej produkt w rankingu.",
    "popularność": "Punkt odniesienia bez tekstu: wszystkim polecamy te same produkty, najczęściej kupowane w okresie treningowym, np. popularną żarówkę czy wycieraczki.",
    "NDCG@k": "Główna metryka (0–1, więcej znaczy lepiej). Nagradza trafienia w pierwszych k pozycjach i bardziej te wyżej: trafiona wycieraczka na miejscu 1 jest warta więcej niż na miejscu 9.",
    "precyzja@k": "Jaka część k poleconych produktów faktycznie spodobała się użytkownikowi. 2 trafienia w 10 polecanych daje 0,2.",
    "czułość@k": "Jaka część produktów, które użytkownik polubił w okresie testowym, trafiła do pierwszych k. Z 4 polubionych 2 w top 10 daje 0,5.",
    "HitRate@k": "Odsetek użytkowników, którzy dostali co najmniej jedno trafienie w pierwszych k pozycjach.",
    "MRR": "Średnia z 1/pozycja pierwszego trafienia. Pierwsze trafienie na miejscu 4 daje 0,25.",
    "podział czasowy": "Uczymy się na opiniach sprzed pierwszej granicy czasu, a oceniamy na opiniach z końca okresu (test), czyli jak w prawdziwym sklepie: znamy przeszłość, zgadujemy przyszłość.",
    "bootstrap": "Sprawdzenie, czy różnica nie jest przypadkiem: wielokrotnie losujemy ze zwracaniem grupę użytkowników i liczymy różnicę wyników. Rozrzut tych wyników daje przedział ufności.",
    "przedział ufności": "Zakres, w którym prawdopodobnie leży prawdziwa różnica. Jeśli cały przedział jest powyżej zera, wariant z LLM jest wyraźnie lepszy; jeśli obejmuje zero, nie umiemy tego stwierdzić.",
    "Bonferroni": "Poprawka na wiele porównań naraz: im więcej hipotez testujemy, tym łatwiej o fałszywy alarm, więc dla porównań głównych przedział jest szerszy (98,33% zamiast 95%).",
}
