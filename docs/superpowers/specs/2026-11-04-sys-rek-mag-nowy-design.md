# sys-rek-mag-nowy — dokument projektowy

Data: 2026-11-04. Status: do akceptacji autora.

## 1. Temat i cel

**Temat (oficjalny):** „Zastosowanie dużych modeli językowych do generowania opisów i cech produktów na potrzeby systemów rekomendacyjnych” / “Application of large language models for generating product descriptions and features for recommendation systems”.

**Cel:** zbadać, czy lokalny LLM, czytając **opinie klientów** o produkcie, potrafi wygenerować opis i cechy produktu, które dają **lepsze rekomendacje** niż oryginalny opis ze sklepu. Wyniki eksperymentu są materiałem do pracy magisterskiej. Nie chodzi o prezentację, chodzi o działający, uczciwy eksperyment.

Koncepcja uzgodniona z promotorem: zbiór produktów + opinie → LLM generuje opisy i cechy → profil nowego użytkownika jako **tekst** („lubię to, nie lubię tamtego”) → systemy rekomendacyjne polecają produkty.

Projekt sys-rek-mag-3 (gry Steam, profile gier) jest porzucony jako koncepcja. Nowy projekt powstaje od zera w `/mnt/kingstone-240gb-m2/DEV/sys-rek-mag-nowy` jako osobne repozytorium git. Z sys-rek-mag-3 przenosimy tylko sprawdzone rozwiązania techniczne (rozdz. 9). sys-rek-mag-3 zostaje nietknięty.

## 2. Cele, nie-cele, anty-cele

**Cele**
- Działający potok: dane → LLM → reprezentacje tekstowe → 3 mechanizmy rekomendacji → metryki.
- Porównanie wariantu **bez LLM** z wariantem **z LLM** dla każdego mechanizmu, na prawdziwych użytkownikach, z podziałem czasowym.
- Na start jeden model (`gemma4:e2b`), architektura gotowa na kolejne modele bez przebudowy.
- Dynamiczny dashboard (Streamlit): wybór modelu, mechanizmu i wariantu, potem metryki, wykresy i opcjonalnie top 5/10 dla profilu tekstowego autora.
- Przewodnik dla studenta prostym językiem (rozdz. 11).

**Nie-cele**
- Wiele kategorii naraz (tylko Automotive).
- Mechanizmy, które nie przyjmują profilu tekstowego (np. czysta faktoryzacja macierzy) jako jeden z trzech głównych.
- Większe modele i mocniejszy sprzęt na tym etapie (dojdą później).
- Scraping, płatne API, publikowanie surowych danych.

**Anty-cele (porażka, nawet jeśli działa)**
- **Mieszanie danych:** wariant „z LLM” zawiera choćby fragment oryginalnego opisu ze sklepu. Promotor wprost tego zabronił („zaśmiecanie”).
- **Błędne koło w ocenie:** prawdę o tym, co się komu podoba, ustalamy sami na podstawie cech z LLM.
- **Przeciek z testu:** dane z okresu testowego wpływają na wybór produktów, użytkowników, opinie wejściowe LLM lub trening.
- Uruchomienie LLM bez zgody autora.

## 3. Dane

**Zbiór:** Amazon Reviews 2023 (McAuley Lab, UC San Diego), kategoria **Automotive** (części, akcesoria, chemia samochodowa). Strona: https://amazon-reviews-2023.github.io/ . Cytowanie: Hou i in., 2024, arXiv:2403.03952. Twórcy nie podają formalnej licencji, proszą o cytowanie. Odnotować w `docs/LICENSES.md` i potwierdzić z promotorem. Surowych danych nie publikujemy.

**Pliki:**
- `meta_Automotive.jsonl.gz`: metadane produktów (`parent_asin`, `title`, `features`, `description`, `categories`, `details`, `store`, …).
- `Automotive.jsonl.gz`: opinie (`user_id`, `parent_asin`, `rating`, `title`, `text`, `timestamp`, `helpful_vote`, `verified_purchase`). Duży plik (20 mln opinii), czytany strumieniowo i filtrowany do próbki.
- 5-core z podziałem czasowym (`benchmark/5core/timestamp_w_his/Automotive.{train,valid,test}.csv.gz`): oceny. 5-core: 632 tys. użytkowników, 267 tys. produktów.

Dokładne URL-e i rozmiary ustala builder, a zapisuje w `docs/DATA.md`. Pobieranie nigdy nie nadpisuje istniejących plików.

**Podział czasowy:** cięcia z gotowego podziału zbioru: `t1 = 1628643414042` ms (2021-08-11), `t2 = 1658002729837` ms (2022-07-16).
- **trening** `< t1`: historia użytkowników, opinie wejściowe dla LLM, trening sieci dwuwieżowej,
- **walidacja** `[t1, t2)`: tylko dobór hiperparametrów / wczesne zatrzymanie sieci dwuwieżowej,
- **test** `≥ t2`: wyłącznie końcowe metryki.

**Próbka produktów (budżet LLM: kilka godzin, maks. ~4 h na `gemma4:e2b`):**
- Kandydaci: produkty z 5-core, które mają metadane z niepustym tytułem i ≥ 5 opinii z niepustym tekstem przed `t1`.
- Wybór: top N według liczby opinii treningowych (remisy według niższego `parent_asin`). Wybór zależy wyłącznie od danych sprzed `t1`.
- N ustala pilotaż (rozdz. 6). Cel to 1000–1500. N jest wiążące, bo wynika z budżetu LLM.

**Źródło interakcji (moja decyzja):** wszystkie interakcje (historie, cele, maskowanie, popularność, trening sieci) pochodzą z **pełnego pliku opinii** `Automotive.jsonl.gz`, zawężonego do produktów z próbki. Daje to więcej użytkowników niż 5-core. Para użytkownik–produkt liczy się raz, według najwcześniejszej opinii. Plik 5-core służy tylko do wyboru kandydatów na produkty. Cięcia `t1`, `t2` bierzemy z gotowego podziału zbioru.

**Użytkownicy ewaluacyjni:** ≥ 1 pozytywna ocena (`rating ≥ 4`) produktu z próbki przed `t1` oraz ≥ 1 pozytywna ocena produktu z próbki w teście (`≥ t2`), dla produktu nieocenionego przed `t2`. Cele testowe to te pozytywne oceny. Profil użytkownika powstaje wyłącznie z interakcji `< t1`. Każdy produkt oceniony przed `t2` (dowolna ocena, również w okresie walidacji) jest maskowany z kandydatów i celów. Remisy w rankingu rozstrzyga niższy `parent_asin`.

*Moja decyzja:* nie ma twardego minimum. Celem jest ≥ 1000 użytkowników ewaluacyjnych, a faktyczną liczbę raportujemy. Jeśli wyjdzie wyraźnie mniej (< 500), `prepare` sprawdza na danych sprzed `t1` wariant z próbką ograniczoną do jednej podkategorii. Wybieramy go tylko wtedy, gdy daje więcej użytkowników, i wpisujemy do rejestru zmian protokołu przed jakąkolwiek ewaluacją.

## 4. Reprezentacje produktów (warianty)

Teksty są w języku angielskim, bo opinie i encoder są angielskie.

- `no_llm`: tytuł + `features` + `description` ze sklepu (stały szablon w `texts.py`).
- `llm_<model_id>`: **wyłącznie** wynik LLM, czyli opis + cechy. Bez tytułu, bez marki, bez czegokolwiek ze sklepu. `model_id` to oczyszczona nazwa modelu z serwera, np. `llm_gemma4-e2b`.

**Wejście LLM:** do K = 20 opinii treningowych produktu, wybranych deterministycznie (najdłuższy tekst, potem nowsze, potem `user_id`; bez `helpful_vote`, bo głosy „pomocne” zbierane są także po T1), z oceną w gwiazdkach. Długość obcinana do budżetu tokenów ustalonego w pilotażu. Do LLM nie trafia ani tytuł, ani opis ze sklepu, bo wynik ma pochodzić tylko z opinii. *Decyzja autora:* tylko opinie, bez tytułu. Pilotaż sprawdza, czy LLM bez tytułu potrafi ustalić, czym jest produkt. Jeśli nie, wracamy do tej decyzji z przykładami z pilotażu, zanim ruszy pełne generowanie.

**Wyjście LLM (JSON, walidowany pydantic):** `description` (2–4 zdania: czym jest produkt, mocne i słabe strony wg klientów) + `features` (lista 5–12 krótkich par aspekt: wartość, np. `brightness: high`, `durability: mixed`). Ponawianie przy błędnym JSON, cache po odcisku (tekst wejścia + hash promptu + model), zapis czasu każdej generacji, serwera, kwantyzacji i sprzętu w manifeście.

**Model:** `gemma4:e2b`, Ollama (Q4_K_M), RX 6600 8 GB, API zgodne z OpenAI. Kolejne modele później, na mocniejszym sprzęcie.

## 5. Mechanizmy rekomendacji

Każdy mechanizm przyjmuje **profil tekstowy** i działa osobno dla każdego wariantu. Profil prawdziwego użytkownika to złożenie reprezentacji produktów ocenionych pozytywnie przed `t1`, w tym samym wariancie. Profil autora w dashboardzie to wpisany tekst (po angielsku).

1. **TF-IDF:** dopasowanie słów (scikit-learn). Profil użytkownika to średnia wektorów TF-IDF jego produktów. Tekst autora jest wektoryzowany tym samym słownikiem.
2. **Embeddingi zdaniowe:** `sentence-transformers/all-MiniLM-L6-v2` (CPU), cosinus. Profil użytkownika to średnia wektorów produktów, a tekst autora jest kodowany bezpośrednio.
3. **Sieć dwuwieżowa (PyTorch, CPU):** wieża produktu to MLP na zamrożonym embeddingu MiniLM produktu, wieża użytkownika to MLP na średniej embeddingów jego produktów (tekst autora trafia jako embedding MiniLM). Trening na interakcjach treningowych z negatywami w batchu, wczesne zatrzymanie na walidacji, seed 2026. Osobny model na każdy wariant.

**Tło:** popularność (liczba ocen przed `t1`). To punkt odniesienia, a nie jeden z trzech mechanizmów.

*Znane ograniczenie (do rozdziału o ograniczeniach):* profil prawdziwego użytkownika powstaje z produktów, a profil autora z wolnego tekstu. Metryki dotyczą tylko tego pierwszego. Top-N dla autora to demonstracja jakościowa.

## 6. Pilotaż

Wymaga zgody autora (wywołuje LLM). Około 20 produktów: sprawdzić czas na produkt, poprawność JSON, sensowność opisów bez tytułu, budżet tokenów. Z czasu wyznaczyć N, tak żeby całość zmieściła się w ~4 h. Wynik pilotażu trafia do `docs/MODELS.md`.

## 7. Ewaluacja

- **Metryka główna:** NDCG@10, makro po użytkownikach ewaluacyjnych.
- **Pozostałe:** Precision@5/10, Recall@5/10, HitRate@5/10, MRR.
- **Porównania główne:** `llm_<model>` vs `no_llm` dla każdego z 3 mechanizmów (3 na model). Sparowany bootstrap po użytkownikach, seed 2026, korekta Bonferroniego **w obrębie jednego modelu** (3 porównania, poziom ufności 98,33%). Dodanie kolejnego modelu nie zmienia wyników poprzednich. Porównania między modelami LLM są opisowe.
- Wszystkie warianty i popularność w tabeli opisowej.
- Protokół (`docs/EXPERIMENT_PROTOCOL.md`) zatwierdzić i zacommitować **przed** pierwszą ewaluacją na prawdziwych danych. Zmiana protokołu oznacza aktualizację dokumentu z wpisem w rejestrze zmian i przeliczenie wszystkiego.
- Losowość tylko przez jawny `numpy.random.Generator` / `torch.Generator` z seedem 2026.

## 8. Struktura projektu i danych

```
sys-rek-mag-nowy/
  README.md, CLAUDE.md, pyproject.toml, uv.lock, .env.example, .gitignore
  docs/  PRZEWODNIK.md, EXPERIMENT_PROTOCOL.md, DATA.md, MODELS.md, LICENSES.md
  src/sys_rek_mag/  config, storage, download, dataset, texts, llm_texts, encode,
                    recommend (TF-IDF, dense, popularność), two_tower, evaluate,
                    dashboard, glossary
  dashboard.py
  tests/
data/                       (gitignored)
  raw/                      pobrane pliki Amazon
  prepared/                 próbka, użytkownicy, podział, manifest
  item_texts/<wariant>/     teksty produktów (+ dla LLM: surowe JSON, czasy, manifest)
  models/<wariant>/         embeddingi, macierz TF-IDF, wagi sieci dwuwieżowej
  results/<wariant>/<mechanizm>/  metryki, wyniki per użytkownik, top-N, manifest
  results/comparisons/      bootstrap
```

Nazwy modułów są orientacyjne. Builder może je dopasować, ale zachowuje zasadę: etapy CLI niezależne, komunikacja przez pliki, każdy etap pisze `manifest.json`, cache tylko przy zgodnych odciskach, zapisy atomowe, jedno źródło ścieżek (`config.py`, `Paths(root)`).

**Etapy CLI** (nazwy `sysrekmag-*`): `download` → `prepare` → `llm-texts --model <nazwa> [--limit N]` (**LLM, tylko za zgodą**) → `encode` → `train` (sieć dwuwieżowa) → `evaluate`. Dashboard: `uv run streamlit run dashboard.py`.

**Dashboard:** odkrywa warianty i mechanizmy, skanując `data/results/`, bez listy na sztywno (gdy jest `results/comparisons/manifest.json`, tylko warianty z ostatniego `evaluate`). Pokazuje: wybór modelu LLM / mechanizmu / wariantu; tabelę metryk; wykresy porównania wariantów z przedziałami ufności; czasy generowania LLM; pole na profil tekstowy autora z top 5/10 dla wybranej kombinacji, obok siebie bez LLM i z LLM (nie wywołuje LLM, tylko encoder); słownik pojęć.

## 9. Co przenosimy z sys-rek-mag-3

Przepisane pod nowe dane, bez logiki gier, bez warm/cold:
- `storage.py`: zapisy atomowe (prawie bez zmian),
- `config.py`: wzorzec `Paths(root)`, `.env`, stałe protokołu,
- klient LLM z `profiles.py`: OpenAI API do Ollamy, pydantic, ponawianie, cache po odcisku, pomiar czasu,
- `embeddings.py`: MiniLM z cache'em,
- `recommend.py`: wektorowe metryki i ranking na macierzach użytkownik × kandydat, maskowanie, remisy,
- `evaluate.py`: przetwarzanie wsadowe, sparowany bootstrap, porównania jako dane,
- `dashboard.py`, `glossary.py`: wzorzec Streamlit i słownika,
- testy: syntetyczne mini-dane w `tmp_path`, fałszywy klient LLM, fałszywy encoder, bez prawdziwych danych i bez LLM,
- `pyproject.toml`: ten sam stos (Python ≥ 3.12, uv, torch z indeksu CPU, sentence-transformers, pandas, pyarrow, pydantic, openai, streamlit) + scikit-learn.

Nie przenosimy: danych i logiki Steam, starego protokołu i dokumentów, audytu gier (ewentualny audyt opisów LLM to późniejsza decyzja).

## 10. Ograniczenia techniczne

- Najpierw `uv venv`, potem instalacja i uruchamianie. Wszystko przez `uv run`.
- Kod i identyfikatory po angielsku, dokumenty i komunikaty CLI po polsku.
- LLM uruchamiany tylko za wyraźną zgodą autora.
- Sprzęt: RX 6600 8 GB (Ollama), reszta na CPU.

## 11. Przewodnik dla studenta (`docs/PRZEWODNIK.md`)

Prosty język, dla osoby, która chce się z niego uczyć:
- co robi projekt w 5 zdaniach i schemat przepływu danych,
- słowniczek (embedding, TF-IDF, sieć dwuwieżowa, NDCG, precyzja, czułość, podział czasowy, bootstrap, cache/odcisk) z przykładami z Automotive,
- mapa folderów i plików: co jest czym, który moduł za co odpowiada,
- krok po kroku: instalacja (`uv`), każdy etap (co robi, co czyta, co zapisuje, ile trwa, kiedy go uruchomić ponownie), dashboard,
- jak dodać kolejny model LLM,
- jak czytać wyniki i najczęstsze problemy.

Przewodnik aktualizujemy razem z kodem.

## 12. Pozostawione builderowi

Dokładne URL-e, prompt LLM (po angielsku, stały, hashowany), szablony tekstów, hiperparametry sieci dwuwieżowej (dobór tylko na walidacji), układ dashboardu, wykresy, nazwy plików wyników, budżet tokenów (z pilotażu).
