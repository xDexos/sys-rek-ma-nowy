# Wyniki

Rola dokumentu: wyniki kolejnych wersji eksperymentu, także archiwalnych. Liczby z plików w `data/` (poza gitem), przepisane tutaj, żeby przetrwały.

## Wersja 1 promptu — `gemma4:e2b` (archiwum, 2026-11-05)

Prompt v1: „opisz, czym jest produkt, jego mocne i słabe strony według klientów i dla kogo jest”. Tryb myślenia wyłączony. 1500 produktów, wszystkie za 1. próbą, średnio 5,1 s/produkt (ok. 2,1 h).

Pliki: `data/archive/2026-11-05_prompt_v1/` (`item_texts/`, `models/`, `results/` = test, `results_valid/` = walidacja).

### Test (18 751 klientów), NDCG@10

| Mechanizm | `no_llm` | `llm_gemma4_e2b` | Zmiana | Przedział 98,33% | Wniosek |
|---|---|---|---|---|---|
| TF-IDF | 0,0162 | 0,0123 | −24% | −0,0055 … −0,0023 | pogorszenie |
| MiniLM | 0,0149 | 0,0117 | −21% | −0,0047 … −0,0016 | pogorszenie |
| Two-tower | 0,0178 | 0,0163 | −8% | −0,0028 … −0,00005 | pogorszenie |
| popularność | 0,0180 | | | | |

### Walidacja (30 197 klientów), NDCG@10

| Mechanizm | `no_llm` | `llm_gemma4_e2b` | Zmiana | Wniosek |
|---|---|---|---|---|
| TF-IDF | 0,0188 | 0,0149 | −21% | pogorszenie |
| MiniLM | 0,0173 | 0,0139 | −20% | pogorszenie |
| Two-tower | 0,0224 | 0,0210 | −6% | pogorszenie |
| popularność | 0,0193 | | | |

Two-tower wybiera epokę na walidacji, więc jego wynik walidacyjny jest lekko zawyżony (tak samo dla obu wariantów).

### Diagnoza

- Cechy v1 są ogólne: aspekt `durability` w 709, `installation` w 708 z 1500 produktów; `installation=easy` 183 razy. Opisują, jak produkt się sprawuje, a mało, czym jest.
- Tekst `no_llm` zawiera markę, model, typ i dopasowanie do aut, czyli słowa najlepiej odróżniające produkty. Najwięcej traci TF-IDF (dopasowanie słów), najmniej two-tower (uczy się z historii).
- 209 z 1500 tekstów `no_llm` przekracza 512 tokenów MiniLM i jest uciętych; teksty LLM mieszczą się w całości.
- W odróżnieniu od większości literatury (KAR, LLMRec, ONCE) tekst LLM **zastępuje** opis sklepu zamiast go uzupełniać, a LLM nie dostaje tytułu.
