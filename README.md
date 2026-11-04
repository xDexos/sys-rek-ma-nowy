# sys-rek-mag-nowy

**Temat (PL):** Zastosowanie dużych modeli językowych do generowania opisów i cech produktów na potrzeby systemów rekomendacyjnych.
**Topic (EN):** Application of large language models for generating product descriptions and features for recommendation systems.

**Cel:** sprawdzić, czy opis i cechy produktu, które lokalny LLM (`gemma4:e2b`) wygenerował wyłącznie z opinii klientów, dają lepsze rekomendacje niż oryginalny tekst ze sklepu. Dane: Amazon Reviews 2023, kategoria Automotive (Hou i in., 2024, arXiv:2403.03952).

## Potok

Dane Amazon → próbka produktów i użytkowników z podziałem czasowym → LLM pisze opis i cechy z opinii → dwa warianty tekstu (`no_llm`, `llm_<model_id>`) → trzy mechanizmy (TF-IDF, embeddingi MiniLM, sieć dwuwieżowa) + popularność → metryki (NDCG@10 i inne) i sparowany bootstrap → dashboard.

## Polecenia

```bash
uv run sysrekmag-download                    # pobranie danych, ok. 3,7 GB (nigdy nie nadpisuje)
uv run sysrekmag-prepare                     # próbka, interakcje, użytkownicy, wejścia LLM
uv run sysrekmag-llm-texts --model <nazwa>   # WYWOŁUJE LLM (pilotaż: --limit 20)
uv run sysrekmag-encode                      # teksty wariantów, embeddingi, TF-IDF
uv run sysrekmag-train                       # sieć dwuwieżowa
uv run sysrekmag-evaluate                    # metryki i bootstrap
uv run streamlit run dashboard.py            # dashboard
uv run pytest                                # testy (bez danych i LLM)
```

## Dokumentacja

- [Przewodnik dla studenta](docs/PRZEWODNIK.md): jak to działa, krok po kroku, problemy.
- [Protokół eksperymentu](docs/EXPERIMENT_PROTOCOL.md), [dane](docs/DATA.md), [modele](docs/MODELS.md), [licencje](docs/LICENSES.md).
