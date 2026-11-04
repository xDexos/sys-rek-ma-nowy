# Dane

Rola dokumentu: źródło, pliki, pola i wymagane miejsce na dysku. Licencja i prawa: `docs/LICENSES.md`.

## Źródło

Amazon Reviews 2023 (McAuley Lab, UC San Diego), kategoria **Automotive**. Strona: https://amazon-reviews-2023.github.io/ .

Cytowanie (wymagane przez twórców):

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952.

## Pliki

Pobiera je `uv run sysrekmag-download` do `data/raw/`. Istniejące pliki są pomijane (nigdy nie nadpisywane); plik w trakcie pobierania ma nazwę `.<nazwa>.part` i dopiero po końcu dostaje docelową nazwę. Adresy są w `config.AMAZON_FILES`; baza: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023`.

| Plik | Ścieżka pod bazą | Rozmiar | Użycie |
|---|---|---|---|
| `Automotive.jsonl.gz` | `raw/review_categories/` | 2,29 GB | opinie: interakcje, wejście LLM, wybór produktów |
| `meta_Automotive.jsonl.gz` | `raw/meta_categories/` | 1,27 GB | metadane: tytuł, cechy, opis (wariant `no_llm`) |
| `Automotive.train.csv.gz` | `benchmark/5core/timestamp_w_his/` | 0,11 GB | tylko lista kandydatów (`parent_asin`) |
| `Automotive.valid.csv.gz` | jw. | 0,03 GB | pobierany, **nieczytany** |
| `Automotive.test.csv.gz` | jw. | 0,03 GB | pobierany, **nieczytany** |

Rozmiary odczytane z nagłówków HTTP; razem ok. 3,7 GB (5-core ok. 0,16 GB). Pliki valid/test 5-core są pobierane dla kompletności, ale kod ich nie czyta, bo zależą od danych po `T1`.

## Używane pola

| Plik | Pola |
|---|---|
| opinie | `user_id`, `parent_asin`, `rating`, `title` (tytuł opinii), `text`, `timestamp` (ms), `helpful_vote` |
| metadane | `parent_asin`, `title`, `features`, `description`, `categories` |
| 5-core train | `parent_asin` |

Nieużywane: `verified_purchase`, `images`, `price`, `store`, `details` i pozostałe pola. Plik opinii jest czytany strumieniowo dwa razy (zliczenie opinii treningowych, potem interakcje i opinie wybranych produktów).

## Miejsce na dysku

- `data/raw/`: ok. 3,7 GB.
- `data/prepared/`, `data/item_texts/`, `data/models/`, `data/results/`: dane pochodne dla ok. 1500 produktów; rozmiar znany dopiero po pierwszym uruchomieniu (szacunkowo znacznie mniej niż `raw/`).
- Przy budowie projektu na dysku było 89 GB wolnego miejsca.
- Model MiniLM (ok. 90 MB) trafia do cache Hugging Face (`~/.cache/huggingface`), model LLM do katalogu Ollamy.

## Znane cechy danych

- Metadane to stan z 2023 r., bez znaczników czasu (zob. „Ograniczenia” w protokole).
- 5-core twórcy zbioru policzyli na pełnym zakresie czasu (zob. protokół).
- Jeden użytkownik może opiniować ten sam produkt kilka razy; liczymy tylko najwcześniejszą opinię.
