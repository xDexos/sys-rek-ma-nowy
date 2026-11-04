# CLAUDE.md

Projekt badawczy (praca magisterska): "Zastosowanie dużych modeli językowych w procesie profilowania obiektów dla systemów rekomendacyjnych". Czy teksty obiektów generowane przez lokalne LLM poprawiają rekomendacje (Amazon Reviews 2023, Automotive; warm i cold-start)? Odpowiadaj po polsku. Kod i identyfikatory po angielsku; docstringi, komunikaty CLI i dokumentacja po polsku.

## Polecenia

Python ≥3.12, `uv`, torch z indeksu CPU. Testy: `uv run pytest` (bez prawdziwych danych, LLM i enkodera).
Etapy: `sysrekmag-download`, `-prepare`, `-llm-texts`, `-encode`, `-train`, `-evaluate`.

**Nigdy nie uruchamiaj `sysrekmag-llm-texts` ani nie wołaj LLM bez wyraźnej zgody użytkownika. Nie pobieraj prawdziwych danych bez zgody.**

## Inwarianty

- Specyfikacja: `docs/superpowers/specs/2026-11-04-sys-rek-mag-nowy-design.md`.
- Podział czasowy (ms): train `< T1`, valid `[T1, T2)`, test `≥ T2`; pozytyw = rating ≥ 4. `SEED = 2026`, losowość tylko przez jawne `Generator`/`SeedSequence`.
- Wariant `llm_<model_id>` zawiera wyłącznie tekst z LLM; LLM nie widzi tytułu ani opisu sklepu.
- Remisy: niższy `parent_asin`; wiersze obiektów zawsze posortowane rosnąco po `parent_asin`.
- Wszystkie zapisy atomowe (`storage.py`); pobieranie nie nadpisuje plików; każdy etap pisze `manifest.json`; cache tylko przy zgodnych fingerprintach.
