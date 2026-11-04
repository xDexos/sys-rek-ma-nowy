# Protokół eksperymentu

Rola dokumentu: jedyne, ustalone z góry źródło reguł badania. Zatwierdzić i zacommitować **przed** pierwszą ewaluacją na prawdziwych danych. Zmiana protokołu oznacza aktualizację tego pliku (z wpisem w „Rejestrze zmian”) i przeliczenie wszystkich wyników. Wartości liczbowe pochodzą z `src/sys_rek_mag/config.py` (i stałych modułów wskazanych niżej).

## Pytanie badawcze

Czy opis i cechy produktu, które lokalny LLM wygenerował **wyłącznie z opinii klientów**, dają lepsze rekomendacje niż oryginalny tekst ze sklepu (tytuł, cechy, opis)? Sprawdzamy to osobno dla trzech mechanizmów rekomendacji, na prawdziwych użytkownikach i z podziałem czasowym.

## Dane i próbka produktów

Amazon Reviews 2023, kategoria Automotive (pliki, adresy i rozmiary: `docs/DATA.md`).

- **Kandydaci:** `parent_asin` z treningowej części 5-core (`Automotive.train.csv.gz`). Pliki `valid`/`test` 5-core nie są czytane.
- **Warunki:** metadane z niepustym tytułem oraz ≥ `MIN_ITEM_TRAIN_REVIEWS = 5` opinii z niepustym tekstem przed `T1` (liczone w pełnym pliku opinii).
- **Wybór:** top `N_ITEMS` produktów według liczby tych opinii; remis rozstrzyga niższy `parent_asin`. Wybór zależy wyłącznie od danych sprzed `T1`.
- **N_ITEMS = 1500** to wartość wstępna. Ostateczną wartość ustala pilotaż (sekcja „Pilotaż i reguła ustalenia N”).
- **Źródło interakcji:** historie, cele, maskowanie, popularność i trening sieci pochodzą z **pełnego pliku opinii** `Automotive.jsonl.gz`, zawężonego do produktów z próbki. Plik 5-core służy tylko do wyboru kandydatów.
- **Jedna para użytkownik–produkt liczy się raz**, według najwcześniejszej opinii. Przy równych znacznikach czasu zostaje pierwsza opinia w kolejności pliku.

## Podział czasowy

Cięcia z gotowego podziału zbioru (ms od 1970-01-01 UTC):

| Okres | Zakres | Do czego służy |
|---|---|---|
| trening | `timestamp < T1`, `T1 = 1628643414042` (2021-08-11 00:56 UTC) | historia użytkowników, wejście LLM, wybór produktów, popularność, trening sieci dwuwieżowej |
| walidacja | `T1 ≤ timestamp < T2` | tylko wczesne zatrzymanie sieci dwuwieżowej |
| test | `timestamp ≥ T2`, `T2 = 1658002729837` (2022-07-16 20:18 UTC) | wyłącznie końcowe metryki |

Ocena pozytywna: `rating ≥ POSITIVE_RATING = 4`.

## Użytkownicy ewaluacyjni i maskowanie

- Użytkownik ewaluacyjny ma ≥ 1 pozytywną ocenę produktu z próbki przed `T1` i ≥ 1 pozytywną ocenę produktu z próbki w teście.
- Cele testowe to jego pozytywne oceny z testu. Ponieważ para liczy się raz (najwcześniejsza opinia), cel testowy zawsze dotyczy produktu bez wcześniejszej interakcji tego użytkownika.
- Profil użytkownika powstaje wyłącznie z produktów ocenionych pozytywnie przed `T1`, w tym samym wariancie tekstu.
- Każdy produkt, z którym użytkownik miał interakcję przed `T2` (dowolna ocena, także w walidacji), jest usuwany z kandydatów i celów.
- Remisy w rankingu rozstrzyga niższy `parent_asin` (kolumny są zawsze w porządku rosnącego `parent_asin`, sortowanie stabilne).
- Nie ma twardego minimum. Cel to ≥ 1000 użytkowników ewaluacyjnych; faktyczną liczbę raportujemy (`data/prepared/manifest.json`).

## Reguła podkategorii

Jeśli `sysrekmag-prepare` da mniej niż `MIN_EVAL_USERS_WARN = 500` użytkowników ewaluacyjnych, wypisze ostrzeżenie i 10 podkategorii z największą sumą opinii treningowych (tylko dane sprzed `T1`). Autor może wtedy **jeden raz**, przed obliczeniem jakiejkolwiek metryki, uruchomić `sysrekmag-prepare --subcategory <nazwa>`. Decyzję (także decyzję „bez zmian”) wpisuje do rejestru zmian.

## Warianty tekstu produktu

Teksty są po angielsku (opinie i enkoder są angielskie). Szablony: `src/sys_rek_mag/texts.py`.

| Wariant | Tekst |
|---|---|
| `no_llm` | tytuł, cechy (`features`) i opis (`description`) ze sklepu, połączone `". "`; puste części pomijane |
| `llm_<model_id>` | **wyłącznie** wynik LLM: `<description> Features: aspekt: wartość; aspekt: wartość; …` |

`model_id` to nazwa modelu z serwera po oczyszczeniu (znaki spoza `A-Za-z0-9._-` → `_`), np. `gemma4:e2b` → wariant `llm_gemma4_e2b`.

## Generowanie tekstów LLM

Kod: `src/sys_rek_mag/llm_texts.py`.

- **Wejście:** do `MAX_REVIEWS_PER_ITEM = 20` opinii treningowych z niepustym tekstem, w kolejności: najwięcej `helpful_vote`, potem nowsze, potem `user_id`. Każda opinia ma ocenę w gwiazdkach i tytuł opinii; tekst obcięty do `REVIEW_MAX_CHARS = 800` znaków.
- **LLM widzi tylko opinie.** Nie widzi tytułu produktu, marki ani opisu ze sklepu.
- **Wyjście (JSON, walidacja pydantic):** `description` (prompt: „2-4 sentences, at most 80 words”, walidator: 20–80 słów) i `features` (5–12 par `aspect: value`, małe litery). Prompt zabrania wspominania ocen i popularności.
- `response_format: json_schema`, temperatura 0, seed 2026, `max_tokens = 1000`, maks. 3 próby z komunikatem błędu walidacji.
- **Ważność generacji:** generacja jest ważna tylko przy zgodnej trójce (`source_hash` = skrót tekstu wejścia, `PROMPT_HASH` = skrót promptu i schematu, nazwa modelu). Wariant jest **kompletny** tylko wtedy, gdy każdy produkt próbki ma ważną generację względem bieżących wejść (`encode` sprawdza to sam przy każdym uruchomieniu; `complete` w manifeście jest tylko informacją). `sysrekmag-encode` pomija warianty niekompletne, a `train` i `evaluate` ich nieaktualne embeddingi.
- Manifest zapisuje prompt, schemat, `prompt_hash`, serwer, wersję, kwantyzację, sprzęt (`LLM_HARDWARE`) i czasy; każda generacja ma swój czas w `generations.jsonl`.

## Mechanizmy rekomendacji

Każdy mechanizm działa osobno dla każdego wariantu. Profil prawdziwego użytkownika to złożenie wektorów produktów, które ocenił pozytywnie przed `T1`.

1. **TF-IDF** (`tfidf`): `TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")` dopasowany do tekstów produktów danego wariantu. Wynik = cosinus profilu (suma wektorów historii; cosinus nie zależy od skali, więc to samo co średnia) z produktem.
2. **Embeddingi** (`dense`): `sentence-transformers/all-MiniLM-L6-v2` na CPU, `max_seq_length = 512`, wektory znormalizowane L2. Wynik = cosinus średniej wektorów historii z produktem.
3. **Sieć dwuwieżowa** (`two_tower`, PyTorch, CPU): obie wieże to MLP `384 → 256 → ReLU → 128` z normalizacją L2 na wyjściu. Wieża produktu dostaje zamrożony embedding MiniLM produktu, wieża użytkownika średnią embeddingów jego historii. Osobny model na każdy wariant.

**Punkt odniesienia — popularność:** `log1p(liczba interakcji treningowych produktu, dowolna ocena)`, ten sam ranking dla wszystkich (z maskowaniem). To nie jest jeden z trzech mechanizmów i nie zależy od tekstu.

## Hiperparametry sieci dwuwieżowej

Stałe w `src/sys_rek_mag/two_tower.py`, dobierane tylko na walidacji:

| Parametr | Wartość |
|---|---|
| warstwa ukryta / wyjście | 256 / 128 |
| temperatura softmaksu | 0,05 |
| optymalizator, lr | Adam, 0,001 |
| batch | 256 |
| epoki maks. / cierpliwość | 50 / 5 |
| seed | 2026 (`torch.manual_seed`, `torch.Generator`), algorytmy deterministyczne |

- **Przykłady treningowe:** dla każdego użytkownika z ≥ 2 pozytywami przed `T1` każdy pozytyw jest celem, a wejściem jest średnia embeddingów pozostałych pozytywów.
- **Strata:** entropia krzyżowa z negatywami w batchu (pozostałe produkty z tego samego batcha).
- **Wczesne zatrzymanie:** po każdej epoce NDCG@10 na walidacji (użytkownicy z historią i pozytywem w walidacji na produkcie spoza treningu; maskowane produkty z treningu). Zostają wagi z najlepszej epoki.
- Odcisk treningu: skrót `embeddings.npy`, skrót `interactions.parquet`, seed i hiperparametry. `sysrekmag-evaluate` odmawia użycia sieci z innym odciskiem.

## Metryki

Binarna istotność, `k ∈ TOP_KS = (5, 10)`, makro po użytkownikach ewaluacyjnych (każdy ma ≥ 1 cel testowy).

- **Główna:** NDCG@10.
- **Pozostałe:** Precision@5/10 (trafienia / k), Recall@5/10 (trafienia / liczba celów), HitRate@5/10, **MRR@10** (1 / pozycja pierwszego trafienia w top 10, 0 gdy brak).

## Porównania i poziomy ufności

- Różnica = `llm_<model_id>` − `no_llm` dla tego samego mechanizmu i metryki, sparowana po `user_id`.
- Sparowany bootstrap percentylowy po użytkownikach, `BOOTSTRAP_RESAMPLES = 10 000` replik, ziarno `[2026, crc32(etykieta porównania)]` (wynik nie zależy od kolejności porównań).
- **Rodzina główna (Bonferroni w obrębie jednego modelu LLM):** `PRIMARY_COMPARISONS_PER_MODEL = 3` porównania (NDCG@10, `llm` vs `no_llm`, po jednym na mechanizm), `FAMILY_ALPHA = 0,05`, więc przedziały **98,33%** (1 − 0,05/3).
- **Porównania pomocnicze:** pozostałe metryki, przedziały `SUPPORTING_CONFIDENCE = 95%`.
- Dodanie kolejnego modelu nie zmienia wyników poprzednich (każdy model ma własną rodzinę). Porównania między modelami LLM są opisowe.
- Wniosek: „poprawa”, gdy cały przedział > 0; „pogorszenie”, gdy cały przedział < 0; w pozostałych przypadkach „brak istotnej różnicy”.
- Wszystkie warianty i popularność trafiają do tabeli opisowej (`data/results/<wariant>/<mechanizm>/metrics.json`).

## Pilotaż i reguła ustalenia N

- Polecenie: `uv run sysrekmag-llm-texts --model gemma4:e2b --limit 20`. **Uruchamia LLM — tylko za zgodą autora.** Produkty pilotażu to pierwsze 20 ze stałej losowej permutacji (seed 2026).
- Sprawdzamy: czas na produkt, poprawność JSON (liczba prób), czy LLM bez tytułu rozpoznaje, czym jest produkt, oraz czy wejście mieści się w kontekście modelu.
- Jeśli LLM bez tytułu nie potrafi ustalić, czym jest produkt, wracamy do decyzji „tylko opinie” z przykładami z pilotażu, zanim ruszy pełne generowanie.
- **Reguła N:** `N_ITEMS` takie, żeby `N_ITEMS × średni czas na produkt z pilotażu` ≤ ok. 4 h; cel 1000–1500. Wynik pilotażu trafia do `docs/MODELS.md`, a wartość N do `config.py` i rejestru zmian.
- Protokół z ostatecznym N commitujemy przed pierwszą ewaluacją na prawdziwych danych.

## Ograniczenia

- **5-core na pełnym zakresie czasu.** Kandydaci pochodzą z treningowej części 5-core, ale sam filtr 5-core twórcy zbioru policzyli na całym okresie (także po `T1`). Do próbki trafiają produkty najczęściej opiniowane przed `T1`, czyli popularne, które przeszłyby filtr i bez danych późniejszych, więc efekt jest pomijalny.
- **Metadane to stan z 2023 r., bez dat.** Tekst `no_llm` może zawierać informacje dopisane po `T1`.
- **Reguła podkategorii używa danych testowych.** Liczba użytkowników ewaluacyjnych, która uruchamia ostrzeżenie, zależy od ocen z testu. Dlatego decyzja zapada raz, przed jakąkolwiek metryką, i jest zapisana w rejestrze zmian.
- **Fałszywe negatywy w batchu.** Ten sam produkt może trafić do batcha dwa razy; wtedy jego druga kopia jest traktowana jako negatyw.
- **Profil autora w dashboardzie to demonstracja jakościowa.** Metryki dotyczą tylko prawdziwych użytkowników, których profil powstaje z produktów (ocena ≥ 4 przed `T1`), a nie z wolnego tekstu.
- **Jedna próbka produktów.** Bootstrap uwzględnia tylko losowanie użytkowników.
- **Kwantyzacja.** `gemma4:e2b` działa w Q4_K_M; wynik dotyczy tej konfiguracji.

## Rejestr zmian

- 2026-11-04: pierwsza wersja protokołu, spisana przed pobraniem danych, pilotażem i jakąkolwiek ewaluacją. `N_ITEMS = 1500` jest wartością wstępną do potwierdzenia po pilotażu.
