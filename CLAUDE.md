# CLAUDE.md

Projekt badawczy (praca magisterska): "Zastosowanie dużych modeli językowych do generowania opisów i cech produktów na potrzeby systemów rekomendacyjnych". Czy opis i cechy produktu wygenerowane przez lokalny LLM wyłącznie z opinii klientów dają lepsze rekomendacje niż tekst sklepu (Amazon Reviews 2023, Automotive)? Odpowiadaj po polsku. Kod i identyfikatory po angielsku; docstringi, komunikaty CLI i dokumentacja po polsku.

## Polecenia

Python ≥3.12, `uv`, torch z indeksu CPU. Testy: `uv run pytest` (sztuczne mini-dane w `tmp_path`, fałszywy klient LLM i enkoder; bez prawdziwych danych i LLM).

```bash
uv run sysrekmag-download     # download.py  – pliki Amazon do data/raw/, nigdy nie nadpisuje
uv run sysrekmag-prepare [--items N] [--subcategory S]   # dataset.py – próbka, interakcje, użytkownicy, wejścia LLM
uv run sysrekmag-llm-texts --model <nazwa> [--limit N] [--base-url URL] [--quantization Q]   # llm_texts.py – WYWOŁUJE LLM
uv run sysrekmag-encode       # encode.py    – no_llm + kompletne llm_<model_id>: MiniLM, TF-IDF
uv run sysrekmag-train        # two_tower.py – sieć dwuwieżowa na wariant
uv run sysrekmag-evaluate     # evaluate.py  – metryki per użytkownik, bootstrap -> data/results/
uv run streamlit run dashboard.py
```

**Nigdy nie uruchamiaj `sysrekmag-llm-texts` ani nie wołaj LLM bez wyraźnej zgody użytkownika. Nie pobieraj prawdziwych danych bez zgody. Nie uruchamiaj `sysrekmag-evaluate` na prawdziwych danych przed zacommitowaniem protokołu.**

## Architektura

- Niezależne etapy CLI komunikują się przez pliki w `data/` (gitignored); każdy pisze `manifest.json`. Cache tylko przy zgodnych odciskach (LLM: skrót wejścia + `PROMPT_HASH` + model; encode: enkoder + max długość + teksty; train: skróty embeddingów i interakcji + seed + hiperparametry). Kompletność wariantu LLM liczona z bieżących wejść (`current_generations`), nie z flagi `complete`; `train`/`evaluate` biorą warianty z `current_variants` (nieaktualne `models/<v>` pomijają z ostrzeżeniem), a `evaluate` odmawia pracy na nieaktualnych embeddingach/sieci bieżącego wariantu.
- `config.py` to jedno źródło ścieżek i stałych protokołu, wczytuje `.env`. `Paths(root)`; testy podają `Paths(tmp_path / "data")`.
- `texts.py`: stałe szablony. `recommend.py`: czyste funkcje rankingu i metryk na macierzach użytkownik × produkt. Porównania to dane (`comparison_specs`).
- Warianty: `no_llm`, `llm_<model_id>` (`model_id("gemma4:e2b") == "gemma4_e2b"`). Mechanizmy: `tfidf`, `dense`, `two_tower`; punkt odniesienia `baseline/popularity`.

## Inwarianty

- Protokół: `docs/EXPERIMENT_PROTOCOL.md`. Zmiana = aktualizacja dokumentu (wpis w rejestrze zmian) i przeliczenie wszystkiego. Przewodnik `docs/PRZEWODNIK.md` aktualizujemy razem z kodem.
- Podział czasowy (ms): train `< T1`, valid `[T1, T2)`, test `≥ T2`; pozytyw = rating ≥ 4. `SEED = 2026`, losowość tylko przez jawne `Generator`/`SeedSequence`.
- Wariant `llm_<model_id>` zawiera wyłącznie tekst z LLM; LLM nie widzi tytułu ani opisu sklepu.
- Każdy produkt z interakcją przed `T2` jest maskowany z kandydatów i celów. Remisy: niższy `parent_asin`; wiersze obiektów zawsze posortowane rosnąco po `parent_asin`.
- Wszystkie zapisy atomowe (`storage.py`). Metryka główna: makro NDCG@10; 3 porównania główne na model LLM przy 98,33% (Bonferroni).
