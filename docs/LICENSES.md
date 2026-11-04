# Licencje i prawa autorskie

Rola dokumentu: licencje wszystkiego, czego projekt używa i co wytwarza. Stan na 2026-11-04. Pozycje **TODO** wymagają weryfikacji przez autora. To nie jest porada prawna; obowiązują regulacje uczelni i treść samych licencji.

## Podsumowanie

| Element | Licencja / prawa | Status |
|---|---|---|
| Kod projektu | nie ustalono | **TODO** (autor) |
| Amazon Reviews 2023 (Automotive) | brak formalnej licencji; twórcy proszą o cytowanie | **TODO**: potwierdzić z promotorem |
| Teksty opinii i opisy produktów | prawa autorów opinii / sprzedawców / Amazon | używane lokalnie, nie publikowane |
| `gemma4:e2b` (Gemma 4, Google) | Apache 2.0 | potwierdzone (`ollama show gemma4:e2b`, 2026-11-04) |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache 2.0 | potwierdzone (karta modelu w cache Hugging Face) |
| Ollama | MIT | narzędzie, nie jest dystrybuowane |
| Biblioteki Pythona | BSD / MIT / Apache 2.0 | potwierdzone (metadane pakietów) |
| Teksty LLM, embeddingi, wyniki | zob. „Dane wytworzone w projekcie” | nie są publikowane w repozytorium |

## Kod projektu

Kod w `src/`, `tests/` i `dashboard.py` napisał autor pracy z pomocą asystenta AI (Claude Code; commity mają adnotację `Co-Authored-By`). Repozytorium nie ma pliku `LICENSE`.

**TODO (autor):** wybrać licencję kodu i sprawdzić zasady uczelni dotyczące praw do kodu z pracy dyplomowej oraz deklarowania użycia narzędzi AI.

## Amazon Reviews 2023

- Twórcy (McAuley Lab, UC San Diego) nie podają formalnej licencji; proszą o cytowanie: Hou i in., 2024, arXiv:2403.03952 (pełny zapis: `docs/DATA.md`).
- **Surowych danych nie publikujemy.** `data/` jest w `.gitignore`; pliki leżą tylko lokalnie.
- Opinie napisali klienci Amazon, opisy produktów sprzedawcy. W pracy cytujemy tylko krótkie fragmenty jako przykłady, z podaniem źródła (prawo cytatu, art. 29 ustawy o prawie autorskim i prawach pokrewnych).
- `user_id` w zbiorze jest zanonimizowany; nie próbujemy go łączyć z kontami i nie publikujemy danych pojedynczych użytkowników.
- **TODO (autor):** potwierdzić z promotorem, że brak formalnej licencji wystarcza do użycia w pracy magisterskiej.

## Modele

| Model | Rola | Licencja |
|---|---|---|
| `gemma4:e2b` | opisy i cechy produktów | Apache 2.0* |
| `sentence-transformers/all-MiniLM-L6-v2` | enkoder tekstu | Apache 2.0 |

\* Gemma 4 jest na licencji Apache 2.0, co potwierdza `ollama show gemma4:e2b` (2026-11-04); Gemma 1–3 miały własne „Gemma Terms of Use”. Modele działają lokalnie; wag nie redystrybuujemy.

## Oprogramowanie

Wersje z `uv.lock`, licencje z metadanych zainstalowanych pakietów:

| Pakiet | Wersja | Licencja |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause (z komponentami 0BSD, MIT, Zlib, CC0-1.0) |
| pandas | 3.0.5 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| pyarrow | 25.0.1 | Apache-2.0 |
| pydantic | 2.13.5 | MIT |
| openai (klient API) | 3.14.1 | Apache-2.0 |
| sentence-transformers | 6.0.1 | Apache-2.0 |
| transformers | 5.22.1 | Apache-2.0 |
| torch (CPU) | 2.14.0 | BSD-3-Clause z komponentami Apache-2.0, BSD-2-Clause, BSL-1.0, MIT |
| streamlit | 1.64.0 | Apache-2.0 |
| altair | 6.3.0 | BSD-3-Clause |
| pytest (dev) | 9.2.0 | MIT |

Pełna lista zależności: `uv.lock`; licencję pakietu pokazuje `uv pip show <pakiet>`.

## Dane wytworzone w projekcie

- **Teksty LLM** (`data/item_texts/llm_*/`) powstają z opinii klientów, więc mogą je parafrazować. Nie są publikowane w repozytorium.
- **Embeddingi, macierze TF-IDF, wagi sieci, metryki** (`data/models/`, `data/results/`) to dane pochodne; publikujemy tylko zagregowane wyniki (tabele, wykresy) w pracy.

## Przy publikacji repozytorium

1. Dodać plik `LICENSE` z wybraną licencją kodu.
2. Upewnić się, że `data/` i `.env` nie trafiają do repozytorium (są w `.gitignore`).
3. W README podać źródło danych z cytowaniem i licencje modeli zgodnie z tym dokumentem.
