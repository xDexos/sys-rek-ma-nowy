# Modele

Rola dokumentu: jakie modele generują teksty i kodują je, na jakim sprzęcie, oraz wynik pilotażu.

## Model LLM (na start)

| Pole | Wartość |
|---|---|
| model | `gemma4:e2b` (wariant `llm_gemma4_e2b`) |
| serwer | Ollama, API zgodne z OpenAI (`LLM_BASE_URL`, domyślnie `http://localhost:11434/v1`) |
| kwantyzacja | Q4_K_M (biblioteka Ollamy) |
| sprzęt | AMD RX 6600 8 GB (komputer autora) |
| ustawienia | temperatura 0, seed 2026, `max_tokens` 1000, maks. 3 próby |

Serwer, wersję i kwantyzację manifest odczytuje sam z Ollamy (`/api/version`, `/api/show`). Sprzęt wpisz w `.env` jako `LLM_HARDWARE` (np. `RX 6600 8 GB, 15 GB RAM`), bo serwer go nie podaje.

## Enkoder

`sentence-transformers/all-MiniLM-L6-v2` (384 wymiary), CPU, `max_seq_length = 512`, wektory znormalizowane L2. Ten sam dla wszystkich wariantów. Manifest `data/models/<wariant>/manifest.json` podaje medianę i maksimum długości tekstów w tokenach oraz liczbę tekstów uciętych.

## Pilotaż

**Do wypełnienia po pilotażu.** Polecenie (uruchamia LLM, tylko za zgodą autora):

```bash
uv run sysrekmag-llm-texts --model gemma4:e2b --limit 20
```

| Pole | Wynik |
|---|---|
| data, wersja Ollamy | — |
| średni / mediana / maks. czas na produkt | — |
| poprawne za 1. próbą (z 20) | — |
| czy LLM bez tytułu rozpoznaje, czym jest produkt (ocena autora) | — |
| czy wejście mieści się w kontekście modelu (zob. niżej) | — |
| wyznaczone `N_ITEMS` (≤ ~4 h) i szacowany czas pełnego przebiegu | — |

Na co uważać:

- Wejście ma do 20 opinii po maks. 800 znaków, czyli do kilku tysięcy tokenów. Ollama ma ograniczony domyślny kontekst i może po cichu uciąć początek promptu. Przed pilotażem sprawdź ustawienie kontekstu (np. zmienna `OLLAMA_CONTEXT_LENGTH` przy starcie serwera) i zapisz je tutaj.
- Orientacyjnie: w poprzednim projekcie (sys-rek-mag-3, inne zadanie, krótsze wejście) `gemma4:e2b` na tym samym sprzęcie potrzebował ok. 9 s na odpowiedź. To nie jest pomiar dla tego projektu.

## Kolejne modele

- Później, na mocniejszym sprzęcie (np. vLLM). Ten sam prompt, ten sam schemat, te same produkty.
- Każdy model to osobny wariant `llm_<model_id>` i osobna rodzina 3 porównań głównych; dodanie modelu nie zmienia wyników poprzednich.
- Dla serwera, który nie podaje kwantyzacji, przekaż ją ręcznie: `--quantization FP8`; adres serwera: `--base-url`.
- Instrukcja krok po kroku: `docs/PRZEWODNIK.md`, „Jak dodać kolejny model LLM”.
