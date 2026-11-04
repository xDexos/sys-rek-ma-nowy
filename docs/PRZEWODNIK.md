# Przewodnik po projekcie

Dla kogo: dla studenta, który chce zrozumieć, co robi ten projekt, i samodzielnie go uruchomić. Reguły badania (wiążące) są w `docs/EXPERIMENT_PROTOCOL.md`. Ten przewodnik tłumaczy je prostymi słowami. Aktualizujemy go razem z kodem.

## 1. Projekt w 5 zdaniach

1. Bierzemy produkty samochodowe z Amazona (kategoria Automotive) i opinie klientów o nich.
2. Lokalny model językowy (LLM, `gemma4:e2b`) czyta **tylko opinie** i pisze z nich krótki opis oraz listę cech produktu.
3. Każdy produkt ma więc dwa teksty: oryginalny ze sklepu (`no_llm`) i napisany przez LLM (`llm_gemma4_e2b`).
4. Trzy mechanizmy rekomendacji (TF-IDF, embeddingi, sieć dwuwieżowa) polecają produkty prawdziwym użytkownikom raz na jednym, raz na drugim tekście.
5. Sprawdzamy na późniejszych zakupach tych użytkowników, który tekst daje trafniejsze polecenia, i czy różnica nie jest przypadkiem.

## 2. Jak to działa na przykładzie

> **Uwaga: cały ten rozdział jest zmyślonym przykładem.** Produkt, opinie, teksty i rankingi wymyśliliśmy, żeby pokazać mechanikę. To nie są dane z Amazona ani wyniki projektu.

**Produkt:** żarówka LED H11 do reflektorów.

**Tekst ze sklepu (wariant `no_llm`):** tytuł, cechy i opis połączone kropkami:

```
H11 LED Headlight Bulb 6000K Cool White. 12000 lumens per pair. Plug and play installation. Built-in cooling fan. 2-year warranty
```

**Trzy opinie klientów** (to widzi LLM; nie widzi tytułu ani opisu sklepu):

```
Review 1 (5/5 stars): Night and day. Much brighter than my old halogens, took 10 minutes to put in.
Review 2 (4/5 stars): Good but loud. Bright light, but the fan hums and one side flickered until I added a decoder.
Review 3 (2/5 stars): Glare. Fits my car fine, but oncoming drivers flash me. Beam pattern is scattered.
```

**Co pisze LLM** (JSON, potem zamieniany na tekst wariantu `llm_gemma4_e2b`):

```
An LED replacement bulb for car headlights. Customers say it is much brighter than halogen bulbs and quick to
install, but some report fan noise, flickering that needs a decoder, and glare for oncoming drivers. Suits
drivers who want brighter night visibility. Features: brightness: high; installation: easy; noise: fan hum;
flicker: sometimes; beam pattern: scattered
```

Zwróć uwagę: w tekście LLM nie ma „H11” ani nazwy marki, bo LLM ich nie widział. Jest za to to, co mówią klienci (hałas, migotanie, oślepianie).

**Profil użytkownika.** Prawdziwy użytkownik przed `T1` ocenił na 4–5 gwiazdek np. żarówki LED H7 i dekoder CANbus. Jego profil to średnia wektorów tych dwóch produktów. W dashboardzie możesz zamiast tego wpisać własny tekst, np. `bright LED bulbs for headlights, easy install`.

**Co mogłyby polecić warianty (przykład):**

| Wariant | Co trafia wysoko i dlaczego |
|---|---|
| `no_llm` + TF-IDF | inne produkty ze słowami „LED”, „headlight”, „H11” w opisie sklepu, np. kolejne zestawy żarówek |
| `llm_gemma4_e2b` + embeddingi | produkty, które klienci opisują podobnie: jasne, łatwe w montażu, czasem migoczą, np. żarówki do świateł przeciwmgielnych i dekodery |
| popularność | to samo dla wszystkich: najczęściej oceniane produkty, np. wycieraczki |

Później sprawdzamy, co użytkownik naprawdę kupił i ocenił pozytywnie w okresie testowym. Jeśli dekoder z listy `llm_gemma4_e2b` był wśród jego zakupów, ten wariant dostaje punkt.

## 3. Schemat przepływu

```
data/raw/                        (sysrekmag-download, ok. 3,7 GB)
  Automotive.jsonl.gz            opinie
  meta_Automotive.jsonl.gz       metadane (tytuł, cechy, opis)
  Automotive.train.csv.gz        5-core: lista kandydatów
        │
        ▼  sysrekmag-prepare
data/prepared/                   items, interactions, users, llm_inputs, manifest
        │
        ├──────────────────────────────┐
        ▼  sysrekmag-llm-texts         │  (UWAGA: uruchamia LLM)
data/item_texts/llm_<model_id>/        │  generations.jsonl, manifest.json
        │                              │
        ▼  sysrekmag-encode  ◄─────────┘  (no_llm bierze tekst sklepu z items.parquet)
data/item_texts/<wariant>/texts.parquet
data/models/<wariant>/           embeddings.npy, tfidf.npz, tfidf_vectorizer.pkl, manifest.json
        │
        ▼  sysrekmag-train
data/models/<wariant>/           two_tower.pt, two_tower_manifest.json
        │
        ▼  sysrekmag-evaluate
data/results/<wariant>/<mechanizm>/   per_user.parquet, metrics.json, manifest.json
data/results/comparisons/             comparisons.csv, manifest.json
        │
        ▼  uv run streamlit run dashboard.py
przeglądarka: wyniki, koszt LLM, mój profil, słownik
```

## 4. Słowniczek

Te same pojęcia co w zakładce „Słownik” dashboardu (`src/sys_rek_mag/glossary.py`), plus kilka technicznych.

- **Wariant:** jaki tekst o produkcie dajemy algorytmowi. `no_llm` = tytuł, cechy i opis ze sklepu. `llm_<model_id>` = wyłącznie opis i cechy napisane przez LLM z opinii.
- **Mechanizm:** sposób układania rankingu: TF-IDF, embeddingi albo sieć dwuwieżowa. Każdy działa na tekście wybranego wariantu.
- **TF-IDF:** liczenie słów. Słowo częste w tekście produktu, a rzadkie w całym katalogu, waży najwięcej. „halogen” wskaże żarówki, „the” nic nie znaczy.
- **Embedding:** zamiana tekstu na listę liczb (tu 384), tak by teksty o podobnym znaczeniu dawały podobne liczby. „Bright LED bulb” i „white LED lamp” są blisko, choć mają inne słowa.
- **Sieć dwuwieżowa:** mała sieć neuronowa z dwiema „wieżami”. Jedna zamienia historię użytkownika, druga produkt, na wektory w tej samej przestrzeni. Im bardziej wektory są zgodne, tym wyżej produkt. Uczy się na tym, co użytkownicy naprawdę kupowali.
- **Popularność:** punkt odniesienia bez tekstu. Wszystkim polecamy to samo: produkty z największą liczbą ocen przed `T1`, np. popularne wycieraczki.
- **Podział czasowy:** uczymy się na opiniach sprzed `T1` (2021-08-11), dobieramy ustawienia sieci na okresie `T1`–`T2`, oceniamy na opiniach od `T2` (2022-07-16). Jak w sklepie: znamy przeszłość, zgadujemy przyszłość.
- **Maskowanie:** produktów, które użytkownik już ocenił przed `T2`, nie polecamy mu i nie liczymy jako trafienia. Nie ma sensu polecać komuś wycieraczek, które już kupił.
- **NDCG@k:** główna metryka (0–1, więcej = lepiej). Nagradza trafienia w pierwszych k pozycjach, mocniej te wyżej: trafiona wycieraczka na miejscu 1 jest warta więcej niż na miejscu 9.
- **Precyzja@k:** jaka część k poleconych produktów spodobała się użytkownikowi. 2 trafienia w 10 = 0,2.
- **Czułość@k (recall):** jaka część produktów polubionych w teście trafiła do pierwszych k. Z 4 polubionych 2 w top 10 = 0,5.
- **HitRate@k:** odsetek użytkowników z co najmniej jednym trafieniem w pierwszych k.
- **MRR (tu MRR@10):** średnia z 1/pozycja pierwszego trafienia. Pierwsze trafienie na miejscu 4 = 0,25; brak trafienia w top 10 = 0.
- **Bootstrap:** sprawdzenie, czy różnica nie jest przypadkiem. 10 000 razy losujemy ze zwracaniem grupę użytkowników i liczymy średnią różnicę. Rozrzut tych średnich daje przedział ufności.
- **Przedział ufności:** zakres, w którym prawdopodobnie leży prawdziwa różnica. Cały powyżej zera = LLM wyraźnie lepszy; obejmuje zero = nie umiemy tego stwierdzić.
- **Bonferroni:** poprawka na wiele porównań naraz. Na jeden model LLM mamy 3 porównania główne, więc przedział jest szerszy: 98,33% zamiast 95%.
- **Cache i odcisk (fingerprint):** odcisk to skrót (hash) wszystkiego, od czego zależy wynik, np. tekstów produktów i nazwy enkodera. Etap zapisuje odcisk w `manifest.json`. Przy kolejnym uruchomieniu liczy odcisk jeszcze raz: ten sam = bierze gotowy wynik (cache), inny = liczy od nowa.
- **Manifest:** plik `manifest.json`, który każdy etap zapisuje obok wyników: z jakich ustawień i danych powstały.
- **Zapis atomowy:** plik najpierw powstaje pod nazwą tymczasową (`.<nazwa>.tmp`), a dopiero gotowy dostaje właściwą nazwę. Przerwany program nie zostawi połowy pliku.

## 5. Mapa folderów i plików

### Repozytorium

| Ścieżka | Co to jest |
|---|---|
| `README.md`, `CLAUDE.md` | krótki opis i zasady dla asystenta AI |
| `docs/` | ten przewodnik, protokół, dane, modele, licencje; `docs/superpowers/` = projekt i plan budowy |
| `.env.example` | wzór pliku `.env` (adres serwera LLM, sprzęt, enkoder) |
| `dashboard.py` | punkt startowy dashboardu |
| `src/sys_rek_mag/config.py` | **jedno źródło** ścieżek i stałych protokołu (`T1`, `T2`, `N_ITEMS`, seed…) |
| `src/sys_rek_mag/storage.py` | zapisy atomowe, skróty plików i tekstów |
| `src/sys_rek_mag/download.py` | etap `download` |
| `src/sys_rek_mag/dataset.py` | etap `prepare`: próbka, interakcje, użytkownicy, wejścia LLM |
| `src/sys_rek_mag/texts.py` | stałe szablony tekstów: `no_llm`, wejście LLM, `llm_*` |
| `src/sys_rek_mag/llm_texts.py` | etap `llm-texts`: prompt, walidacja JSON, cache generacji |
| `src/sys_rek_mag/encode.py` | etap `encode`: embeddingi MiniLM i TF-IDF |
| `src/sys_rek_mag/recommend.py` | ranking i metryki jako czyste funkcje na macierzach |
| `src/sys_rek_mag/two_tower.py` | etap `train`: sieć dwuwieżowa |
| `src/sys_rek_mag/evaluate.py` | etap `evaluate`: metryki per użytkownik i bootstrap |
| `src/sys_rek_mag/dashboard.py`, `glossary.py` | dashboard Streamlit i słownik pojęć |
| `tests/` | testy na sztucznych mini-danych, z fałszywym LLM i fałszywym enkoderem |

### Folder `data/` (nie trafia do gita)

Przykład dla jednego modelu (`gemma4:e2b`):

```
data/
  raw/
    Automotive.jsonl.gz, meta_Automotive.jsonl.gz
    Automotive.train.csv.gz, Automotive.valid.csv.gz, Automotive.test.csv.gz
  prepared/
    items.parquet          próbka produktów (parent_asin, title, features, description, categories, train_reviews)
    interactions.parquet   user_id, parent_asin, rating, timestamp, split (train/valid/test)
    users.parquet          user_id, train_positives, eval
    llm_inputs.jsonl       do 20 opinii na produkt (wejście LLM)
    manifest.json          parametry, liczby, top podkategorie, odciski plików źródłowych
  item_texts/
    no_llm/texts.parquet
    llm_gemma4_e2b/
      generations.jsonl    jedna linia na produkt: wynik LLM, czas, liczba prób, skróty
      errors.jsonl         tylko gdy jakieś produkty się nie udały
      manifest.json        prompt, serwer (z context_length), kwantyzacja, sprzęt, prompt_tokens_max/median, complete
      texts.parquet        gotowe teksty wariantu (pisze encode)
  models/
    no_llm/, llm_gemma4_e2b/
      embeddings.npy, tfidf.npz, tfidf_vectorizer.pkl, manifest.json      (encode)
      two_tower.pt, two_tower_manifest.json                               (train)
  results/
    no_llm/{tfidf,dense,two_tower}/           per_user.parquet, metrics.json, manifest.json
    llm_gemma4_e2b/{tfidf,dense,two_tower}/   jw.
    baseline/popularity/                      jw. (punkt odniesienia)
    comparisons/                              comparisons.csv, manifest.json
```

## 6. Instalacja

```bash
cd sys-rek-mag-nowy
uv venv
uv sync                 # instaluje zależności (torch z indeksu CPU)
cp .env.example .env    # uzupełnij LLM_HARDWARE, np. "RX 6600 8 GB, 15 GB RAM"
uv run pytest           # testy: bez prawdziwych danych, bez LLM
```

Wszystko uruchamiamy przez `uv run`. Do etapu LLM potrzebny jest działający serwer Ollama z modelem (`ollama pull gemma4:e2b`).

## 7. Etapy krok po kroku

Czasy to szacunki, nie pomiary (do uzupełnienia po pierwszym przebiegu).

| Polecenie | Co robi | Czyta | Zapisuje | Ile trwa | Kiedy uruchomić ponownie |
|---|---|---|---|---|---|
| `uv run sysrekmag-download` | pobiera 5 plików Amazon, istniejące pomija | internet | `data/raw/` | ok. 3,7 GB, zależy od łącza | gdy brakuje pliku (uszkodzony usuń ręcznie) |
| `uv run sysrekmag-prepare` | wybiera produkty, buduje interakcje, użytkowników i wejścia LLM | `data/raw/` | `data/prepared/` | kilkanaście minut (2× czyta 2,29 GB opinii) | po zmianie stałych próbki w `config.py` lub decyzji o `--subcategory` |
| `uv run sysrekmag-llm-texts --model gemma4:e2b` | **uruchamia LLM**: opis i cechy z opinii | `prepared/items.parquet`, `llm_inputs.jsonl` | `item_texts/llm_<model_id>/` | pełna próbka ≤ ok. 4 h | po zmianie promptu, po ponownym `prepare`, gdy wariant jest niekompletny |
| `uv run sysrekmag-encode` | teksty wariantów, embeddingi MiniLM, TF-IDF | `prepared/items.parquet`, kompletne `item_texts/llm_*` | `item_texts/<wariant>/texts.parquet`, `models/<wariant>/` | minuty | zawsze po zmianie tekstów; sam pomija niezmienione warianty |
| `uv run sysrekmag-train` | sieć dwuwieżowa dla każdego aktualnego wariantu (nieaktualne foldery `models/` pomija) | `models/<wariant>/embeddings.npy`, `prepared/` | `models/<wariant>/two_tower*` | minuty na wariant | po `encode`; sam pomija aktualne modele |
| `uv run sysrekmag-evaluate` | metryki i bootstrap | `prepared/`, `models/` | `data/results/` | minuty | po każdej zmianie wyżej; liczy zawsze od nowa |
| `uv run streamlit run dashboard.py` | przeglądanie wyników | `data/results/`, `models/`, `item_texts/` | nic | — | kiedy chcesz |

**`prepare` — opcje.** `--items N` zmienia liczbę produktów (domyślnie `N_ITEMS` z `config.py`), `--subcategory <nazwa>` ogranicza próbkę do jednej podkategorii. Na końcu wypisuje: liczbę produktów, interakcji w każdym okresie, użytkowników i użytkowników ewaluacyjnych. Jeśli ewaluacyjnych jest mniej niż 500, wypisze `UWAGA: tylko … użytkowników ewaluacyjnych (< 500)` i listę podkategorii. Decyzję o `--subcategory` podejmujesz **raz, przed jakąkolwiek metryką**, i wpisujesz do rejestru zmian protokołu.

**`llm-texts` — opcje.** `--model` (wymagane, nazwa jak na serwerze), `--limit N` (tylko N produktów, pilotaż), `--base-url` (inny serwer), `--quantization` (gdy serwer jej nie podaje, np. `FP8`). Co 10 produktów zapisuje postęp, więc przerwanie (Ctrl+C) traci tylko opisy od ostatniego zapisu. Ponowne uruchomienie generuje tylko brakujące.

**`encode`.** Bez LLM, tylko enkoder MiniLM na CPU. Przy pierwszym uruchomieniu pobiera MiniLM (ok. 90 MB) z Hugging Face. Wypisuje medianę i maksimum długości tekstów w tokenach oraz liczbę tekstów uciętych powyżej 512 tokenów.

**`train`.** Wypisuje najlepszą epokę i NDCG@10 na walidacji dla każdego wariantu.

**`evaluate`.** Wypisuje liczbę ocenionych użytkowników i tabelę 3 porównań głównych. **Na prawdziwych danych uruchamiaj dopiero po zacommitowaniu protokołu.**

## 8. Co się przelicza samo (cache i odciski)

| Etap | Bierze gotowy wynik, gdy… | W przeciwnym razie |
|---|---|---|
| `download` | plik już istnieje (bez sprawdzania zawartości) | pobiera |
| `prepare` | nigdy | liczy od nowa |
| `llm-texts` | produkt ma generację z tym samym skrótem wejścia, tym samym `PROMPT_HASH` i tym samym modelem | generuje ten produkt |
| `encode` | odcisk (nazwa enkodera, 512, lista `parent_asin`, wszystkie teksty) zgadza się z manifestem i pliki istnieją | koduje wariant od nowa |
| `train` | odcisk (skrót `embeddings.npy`, skrót `interactions.parquet`, `POSITIVE_RATING`, seed, hiperparametry) zgadza się z manifestem | trenuje od nowa |
| `evaluate` | nigdy; ocenia tylko warianty bieżące (`no_llm` + kompletne `llm_*`), nadmiarowe foldery `models/` pomija z ostrzeżeniem `UWAGA: pomijam …`; **odmawia** pracy na nieaktualnych embeddingach lub sieci bieżącego wariantu | liczy od nowa |

Przykłady:

- **Zmieniasz prompt** w `llm_texts.py` → zmienia się `PROMPT_HASH` → wszystkie stare generacje są nieważne → `llm-texts` generuje wszystko od nowa (LLM, ok. 4 h). Potem `encode`, `train`, `evaluate` przeliczają się same, bo zmieniły się teksty. Jeśli po zmianie promptu zrobisz tylko pilotaż (`--limit`), wariant jest niekompletny: `encode` go pomija, a `train` i `evaluate` pomijają jego stare `models/llm_*` (ostrzeżenie `UWAGA: pomijam …`), zamiast liczyć na tekstach ze starego promptu.
- **Zmieniasz `N_ITEMS`** → `prepare` od nowa → uruchom `llm-texts` (wygeneruje tylko nowe produkty; stare z tym samym wejściem zostają) → `encode` → `train` → `evaluate`.
- **Zmieniasz tylko hiperparametr sieci** w `two_tower.py` → `train` sam to wykryje (hiperparametry są w odcisku); `encode` nic nie robi.

Kompletność wariantu LLM `encode` liczy sam przy każdym uruchomieniu: generacja jest ważna, gdy skrót **bieżącego** wejścia z `llm_inputs.jsonl`, `PROMPT_HASH` i model zgadzają się z rekordem. Flaga `complete` w manifeście LLM to tylko informacja z ostatniego przebiegu `llm-texts`. Po ponownym `prepare`, które zmieniło wejścia LLM (np. `REVIEW_MAX_CHARS`), stare generacje przestają się liczyć i wariant znika z `encode`/`train`/`evaluate`, dopóki nie uruchomisz `llm-texts` (za zgodą).

Uwaga, czego odciski **nie widzą**:

- Zmiana kodu, który nie jest częścią odcisku (np. ustawień `TfidfVectorizer` w `encode.py` albo architektury sieci), nie uruchomi przeliczenia. Wtedy usuń `data/models/<wariant>/manifest.json` (encode) albo `two_tower_manifest.json` (train).

## 9. Pilotaż

**Uruchamia LLM — tylko świadomie, za zgodą autora.**

```bash
uv run sysrekmag-llm-texts --model gemma4:e2b --limit 20
```

- Bierze 20 produktów ze stałej losowej kolejności (seed 2026), żeby nie były to tylko produkty o najniższych `parent_asin`.
- Po każdym produkcie wypisuje czas i liczbę prób, na końcu średni czas i szacunek dla reszty, np. `Średnio 9.0 s/produkt; pozostało 1480 ≈ 3.7 h`.
- Przeczytaj opisy w `data/item_texts/llm_gemma4_e2b/generations.jsonl` i porównaj z tytułami w `items.parquet`. Pytanie: czy LLM bez tytułu wie, czym jest produkt?
- Ustal `N_ITEMS` tak, żeby całość zmieściła się w ok. 4 h, wpisz do `config.py`, wynik do `docs/MODELS.md`, zmianę do rejestru zmian protokołu. Potem `prepare` i pełne `llm-texts` (wygenerowane w pilotażu opisy zostaną użyte ponownie).

## 10. Jak dodać kolejny model LLM

1. Uruchom serwer z modelem (Ollama: `ollama pull <nazwa>`; albo vLLM / LM Studio z API zgodnym z OpenAI).
2. Dla porządku dopisz nazwę do `LLM_MODELS` w `.env` (kod wykrywa warianty ze skanu folderów `item_texts/llm_*`, więc to tylko zapis; dashboard pokazuje tylko warianty z ostatniego `evaluate`).
3. Pilotaż, potem pełny przebieg (**LLM, za zgodą**):
   ```bash
   uv run sysrekmag-llm-texts --model <nazwa> --limit 20
   uv run sysrekmag-llm-texts --model <nazwa> [--base-url http://host:8000/v1] [--quantization FP8]
   ```
   Wynik ląduje w `data/item_texts/llm_<model_id>/` (np. `qwen3:8b` → `llm_qwen3_8b`).
4. `uv run sysrekmag-encode`, `uv run sysrekmag-train`, `uv run sysrekmag-evaluate`. Stare warianty biorą wynik z cache.
5. Nowy model ma własne 3 porównania główne (98,33%). Wyniki poprzednich modeli się nie zmieniają. Porównanie modeli między sobą jest opisowe.
6. Dopisz model, sprzęt i czasy do `docs/MODELS.md`.

## 11. Jak czytać wyniki

### Dashboard

`uv run streamlit run dashboard.py`. Dashboard **nie wywołuje LLM**; w zakładce „Mój profil” używa tylko enkodera MiniLM.

- **Panel boczny:** „Model LLM” (który wariant `llm_*` pokazać w „Mój profil”), „Mechanizm”, „Liczba rekomendacji N” (5 albo 10).
- **Wyniki:** tabela metryk wszystkich wariantów dla wybranego mechanizmu (plus popularność), wykres NDCG@10 i wykres „Różnica NDCG@10 względem bez LLM (przedział ufności)”: kropka to średnia różnica, kreska to przedział, przerywana linia to zero. Kreska w całości na prawo od zera = poprawa.
- **Koszt LLM:** liczba opisów, średni i medianowy czas generowania, sprzęt, serwer.
- **Mój profil:** wpisz po angielsku, czego szukasz (np. `bright LED bulbs for headlights`). Zobaczysz top N obok siebie: „bez LLM” i „z LLM”, z opisem LLM do rozwinięcia. To demonstracja, nie metryka.
- **Słownik:** pojęcia z rozdziału 4.

### `data/results/comparisons/comparisons.csv`

Jeden wiersz = jedno porównanie `llm_<model_id>` vs `no_llm` (na model: 3 mechanizmy × 9 metryk = 27 wierszy).

| Kolumna | Znaczenie |
|---|---|
| `variant`, `baseline` | porównywany wariant LLM i punkt odniesienia (`no_llm`) |
| `mechanism`, `metric` | np. `dense`, `ndcg_at_10` |
| `role` | `primary` (NDCG@10, rodzina Bonferroniego) albo `supporting` |
| `confidence` | 0.9833 dla głównych, 0.95 dla pomocniczych |
| `users` | liczba sparowanych użytkowników |
| `baseline_mean`, `variant_mean` | średnia metryki bez LLM i z LLM |
| `delta` | `variant_mean − baseline_mean` |
| `delta_relative` | `delta / baseline_mean` (0.10 = o 10% lepiej) |
| `ci_low`, `ci_high` | przedział ufności różnicy z bootstrapu |
| `conclusion` | wniosek (niżej) |

- **„poprawa”**: `ci_low > 0`, cały przedział powyżej zera.
- **„pogorszenie”**: `ci_high < 0`, cały przedział poniżej zera.
- **„brak istotnej różnicy”**: przedział obejmuje zero. To nie znaczy „warianty są równe”, tylko „nie umiemy rozstrzygnąć”.

Najpierw patrz na 3 wiersze `primary`. Pozostałe pomagają zrozumieć, skąd różnica (np. LLM poprawia HitRate, ale nie pozycję trafień). Popularność nie jest w tym pliku; jej metryki są w `data/results/baseline/popularity/metrics.json` i w tabeli dashboardu.

## 12. Najczęstsze problemy

**Ollama nie działa** (albo zły adres). Przy każdym produkcie zobaczysz `  BŁĄD <parent_asin>: Connection error.`, a na końcu `Nieudane produkty: N (zob. …/errors.jsonl); ponowne uruchomienie uzupełni braki`. Przerwij Ctrl+C, uruchom serwer (`ollama serve`), sprawdź `LLM_BASE_URL` w `.env` i uruchom ponownie. Gotowe opisy nie przepadną.

**Model nie jest pobrany.** Błąd 404 od serwera w komunikacie `BŁĄD …`. Uruchom `ollama pull gemma4:e2b`; nazwa w `--model` musi być identyczna z nazwą na serwerze.

**LLM zwraca zły JSON.** Program sam ponawia (maks. 3 próby, z treścią błędu). Gdy się nie uda: `Brak poprawnego opisu po 3 próbach: …` dla tego produktu; reszta idzie dalej. Częsty powód: `description musi mieć 20-80 słów` albo `odpowiedź ucięta limitem tokenów; zwróć krótszy JSON`.

**Niekompletny wariant.** `encode` pomija wariant LLM, jeśli choć jeden produkt próbki nie ma generacji ważnej dla bieżącego wejścia, promptu i modelu (podpowiedź: `valid_generations` vs `items` w `data/item_texts/llm_<model_id>/manifest.json`; po zmianie promptu lub ponownym `prepare` liczy się stan bieżący, nie ta flaga). Objaw: brak nowego `data/models/llm_*`, ostrzeżenie `UWAGA: pomijam llm_…` w `train`/`evaluate` i brak wariantu LLM w dashboardzie. Uruchom ponownie `llm-texts` (za zgodą): wygeneruje tylko brakujące.

**Brak miejsca na dysku.** Python zgłosi `OSError: [Errno 28] No space left on device`. Pobieranie usuwa wtedy niedokończony plik `.part`, a zapisy atomowe nie zostawią połowy pliku z wynikami. Zwolnij miejsce (potrzeba ok. 3,7 GB na `data/raw/` plus zapas) i uruchom etap ponownie.

**Brak pliku z danymi.** `prepare` bez pobranych plików kończy się `FileNotFoundError` z nazwą pliku w `data/raw/`. Uruchom `sysrekmag-download`.

**Za mało użytkowników.** `prepare` wypisuje `UWAGA: tylko … użytkowników ewaluacyjnych (< 500)`. Zobacz „Reguła podkategorii” w protokole. `Brak produktów spełniających warunki próbki` oznacza, że filtr (np. zła nazwa w `--subcategory`) odrzucił wszystko.

**Nieaktualne wyniki poprzednich etapów** (komunikaty z `evaluate`):

| Komunikat | Co zrobić |
|---|---|
| `Embeddingi wariantu <wariant> nie odpowiadają aktualnej próbce; uruchom sysrekmag-encode` | `encode`, potem `train` |
| `Embeddingi wariantu <wariant> są nieaktualne albo ich brak; uruchom sysrekmag-encode` (w `train` albo `evaluate`) | `encode`, potem `train` |
| `UWAGA: pomijam llm_…: teksty wariantu nie są kompletne …` | nic, jeśli to celowe (np. pilotaż nowego promptu); inaczej `llm-texts` (za zgodą), `encode`, `train` |
| `Brak sieci dwuwieżowej dla wariantu <wariant>; uruchom sysrekmag-train` | `train` |
| `Sieć dwuwieżowa wariantu <wariant> jest nieaktualna; uruchom sysrekmag-train` | `train` |
| `Użytkownik ewaluacyjny bez pozytywu w treningu; uruchom ponownie sysrekmag-prepare` | `prepare` (pliki w `prepared/` są niespójne) |

**`train` się nie udaje:** `Za mało danych: brak przykładów treningowych albo użytkowników walidacyjnych`. Za mała próbka (np. po bardzo wąskiej podkategorii).

**`encode` kończy się `KeyError` z `parent_asin`.** Próbka zmieniła się po ostatnim `llm-texts` (zob. rozdz. 8). Uruchom `llm-texts` dla tego modelu (za zgodą).

**Dashboard pokazuje `Brak wyników w data/results/`.** Nie było jeszcze `evaluate`. „Brak wariantu LLM w wynikach.” w zakładce „Mój profil” znaczy, że oceniono tylko `no_llm`.
