# Modele

Rola dokumentu: jakie modele generują teksty i kodują je, na jakim sprzęcie, oraz wynik pilotażu.

## Model LLM (na start)

| Pole | Wartość |
|---|---|
| model | `gemma4:e2b` (wariant `llm_gemma4_e2b`) |
| serwer | Ollama, API zgodne z OpenAI (`LLM_BASE_URL`, domyślnie `http://localhost:11434/v1`) |
| kwantyzacja | Q4_K_M (biblioteka Ollamy) |
| sprzęt | AMD RX 6600 8 GB (komputer autora) |
| ustawienia | temperatura 0, seed 2026, `max_tokens` 1000, tryb myślenia wyłączony (`reasoning_effort: none`), maks. 3 próby |

Serwer, wersję i kwantyzację manifest odczytuje sam z Ollamy (`/api/version`, `/api/show`). Sprzęt wpisz w `.env` jako `LLM_HARDWARE` (np. `RX 6600 8 GB, 15 GB RAM`), bo serwer go nie podaje.

## Enkoder

`sentence-transformers/all-MiniLM-L6-v2` (384 wymiary), CPU, `max_seq_length = 512`, wektory znormalizowane L2. Ten sam dla wszystkich wariantów. Manifest `data/models/<wariant>/manifest.json` podaje medianę i maksimum długości tekstów w tokenach oraz liczbę tekstów uciętych.

## Pilotaż

Polecenie (uruchamia LLM, tylko za zgodą autora):

```bash
uv run sysrekmag-llm-texts --model gemma4:e2b --limit 20
```

Wynik z 2026-11-05 (Ollama 0.34.3, `OLLAMA_CONTEXT_LENGTH=8192`). Dwa przebiegi na tych samych produktach:

| Pole | Z trybem myślenia (20 produktów) | Bez trybu myślenia (10 produktów, ustawienie ostateczne) |
|---|---|---|
| średni / mediana / maks. czas na produkt | 16,9 / 15,3 / 24,1 s | 5,3 / 5,4 / 5,6 s |
| poprawne za 1. próbą | 17 z 20 | 10 z 10 |
| tokeny odpowiedzi (sonda na 1 produkcie) | 913 (ok. 2/3 to myślenie) | 294 |
| czy LLM bez tytułu rozpoznaje, czym jest produkt | tak, 20 z 20 (wstępna ocena; do potwierdzenia przez autora) | tak, 10 z 10 |
| `prompt_tokens` min / mediana / maks. | 3305 / 4292 / 4653 (mieści się w 8192; przy domyślnym 4096 większość byłaby ucięta) | jak obok (to samo wejście) |
| wyznaczone `N_ITEMS` i czas pełnego przebiegu | — | **1500**, ok. 2,2 h (1500 × 5,3 s) |

Drobne usterki widoczne w pilotażu: niejednolite nazwy cech (`mounting_hardware` i `mounting hardware`), czasem powtórzony aspekt, pojedyncza nieścisłość (rozgałęźnik na 3 gniazda opisany jako „na dwa”, zgodnie z częścią opinii o starszym wariancie). Nie blokują pełnego przebiegu.

Na co uważać:

- Wejście ma do 20 opinii po maks. 800 znaków, czyli do kilku tysięcy tokenów. Ollama ma ograniczony domyślny kontekst i może po cichu uciąć początek promptu. Przed pilotażem sprawdź ustawienie kontekstu (np. zmienna `OLLAMA_CONTEXT_LENGTH` przy starcie serwera) i zapisz je tutaj.
  - Stan 2026-11-04: Ollama 0.34.3, `OLLAMA_CONTEXT_LENGTH` nieustawione (domyślny kontekst serwera), model `gemma4:e2b` (5,1B, Q4_K_M, maks. kontekst 131072, tryb „thinking” domyślnie włączony). Wejścia z `prepare` mają medianę ok. 16,7 tys. znaków (≈ 4,2 tys. tokenów, szacunek znaki/4; maks. ≈ 4,3 tys.) plus prompt systemowy. To więcej niż typowy domyślny kontekst 4096, więc **przed pilotażem ustaw `OLLAMA_CONTEXT_LENGTH=8192`** i zrestartuj serwer; po pilotażu potwierdź liczbą `prompt_tokens_max`.
- Jak to odczytać po pilotażu: każda generacja w `generations.jsonl` ma `prompt_tokens` (liczba tokenów wejścia policzona przez serwer, z `usage`), a manifest `data/item_texts/llm_<model_id>/manifest.json` podaje `prompt_tokens_max` i `prompt_tokens_median`. Dla Ollamy `server.context_length` to maksymalny kontekst modelu z `/api/show`; faktycznie używany kontekst to ustawienie serwera (`OLLAMA_CONTEXT_LENGTH` / `num_ctx`), zwykle mniejsze. Jeśli `prompt_tokens_max` jest równe albo bliskie temu ustawieniu, wejście zostało ucięte: zwiększ kontekst serwera albo zmniejsz `REVIEW_MAX_CHARS` / `MAX_REVIEWS_PER_ITEM` (zmiana protokołu). `null` oznacza, że serwer nie podał `usage`.
- Orientacyjnie: w poprzednim projekcie (sys-rek-mag-3, inne zadanie, krótsze wejście) `gemma4:e2b` na tym samym sprzęcie potrzebował ok. 9 s na odpowiedź. To nie jest pomiar dla tego projektu.

## Kolejne modele

- Później, na mocniejszym sprzęcie (np. vLLM). Ten sam prompt, ten sam schemat, te same produkty.
- Każdy model to osobny wariant `llm_<model_id>` i osobna rodzina 3 porównań głównych; dodanie modelu nie zmienia wyników poprzednich.
- Dla serwera, który nie podaje kwantyzacji, przekaż ją ręcznie: `--quantization FP8`; adres serwera: `--base-url`.
- Instrukcja krok po kroku: `docs/PRZEWODNIK.md`, „Jak dodać kolejny model LLM”.
