# sys-rek-mag-nowy

**Temat (PL):** Zastosowanie dużych modeli językowych w procesie profilowania obiektów dla systemów rekomendacyjnych.
**Topic (EN):** Application of large language models in the process of object profiling for recommendation systems.

**Cel:** sprawdzić, czy opisy produktów (Amazon Reviews 2023, Automotive) generowane przez lokalne LLM poprawiają rekomendacje dla znanych (warm) i nowych (cold-start) obiektów.

## Polecenia

```bash
uv run sysrekmag-download    # pobranie danych (nigdy nie nadpisuje)
uv run sysrekmag-prepare     # przygotowanie zbioru
uv run sysrekmag-llm-texts --model <nazwa>   # WYWOŁUJE LLM
uv run sysrekmag-encode      # kodowanie tekstów
uv run sysrekmag-train       # model two-tower
uv run sysrekmag-evaluate    # metryki i bootstrap
uv run pytest                # testy
```

Pełny opis: Task 9.
