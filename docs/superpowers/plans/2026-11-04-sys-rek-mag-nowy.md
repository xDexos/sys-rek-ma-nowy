# sys-rek-mag-nowy — plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Potok badawczy: Amazon Reviews 2023 (Automotive) → opisy i cechy produktów z opinii (lokalny LLM) → 3 mechanizmy rekomendacji na profilu tekstowym → metryki i bootstrap → dynamiczny dashboard + przewodnik dla studenta.

**Architecture:** Niezależne etapy CLI komunikujące się plikami w `data/`. Każdy etap pisze `manifest.json`, a cache jest używany tylko przy zgodnych odciskach. Wzorce przeniesione z `/mnt/kingstone-240gb-m2/DEV/sys-rek-mag-3` (czytaj odpowiedni plik, gdy zadanie na niego wskazuje). Funkcje rankingu i metryk to czyste funkcje na macierzach użytkownik × produkt. Wiersze produktów zawsze są posortowane rosnąco po `parent_asin`, a wszystkie macierze mają ten sam porządek.

**Tech Stack:** Python ≥3.12, uv, PyTorch (CPU), sentence-transformers (all-MiniLM-L6-v2), scikit-learn (TF-IDF), pandas, pyarrow, scipy, pydantic, openai (klient Ollamy), Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-sys-rek-mag-nowy-design.md`. Przeczytaj ją przed każdym zadaniem.

## Global Constraints

- Katalog projektu: `/mnt/kingstone-240gb-m2/DEV/sys-rek-mag-nowy`, pakiet `src/sys_rek_mag`, skrypty `sysrekmag-*`.
- Najpierw `uv venv`, potem `uv sync`. Wszystko uruchamiamy przez `uv run`. Torch z indeksu CPU (jak w `sys-rek-mag-3/pyproject.toml`).
- Kod i identyfikatory po angielsku, docstringi, komunikaty CLI i dokumenty po polsku.
- **Nigdy nie uruchamiaj `sysrekmag-llm-texts` bez wyraźnej zgody autora.** Testy nie wywołują LLM ani prawdziwego encodera.
- Wszystkie zapisy atomowe (`storage.py`). Pobieranie nigdy nie nadpisuje istniejących plików.
- Losowość tylko przez jawne `numpy.random.Generator`/`SeedSequence` i `torch.Generator`, `SEED = 2026`.
- Cięcia czasu (ms): `T1 = 1628643414042` (koniec treningu), `T2 = 1658002729837` (początek testu). Trening `< T1`, walidacja `[T1, T2)`, test `≥ T2`.
- Pozytywna ocena: `rating ≥ 4`.
- Wariant `llm_<model_id>` zawiera **wyłącznie** tekst wygenerowany przez LLM. LLM nie widzi tytułu ani opisu ze sklepu.
- Remisy w rankingu rozstrzyga niższy `parent_asin` (stabilne sortowanie po kolumnach w porządku `parent_asin`).
- Nie uruchamiaj `sysrekmag-evaluate` na prawdziwych danych, zanim `docs/EXPERIMENT_PROTOCOL.md` nie jest zacommitowany.

## Review Focus

1. Opinia z pustym albo białym tekstem nie trafia do wejścia LLM i nie liczy się do progu ≥ 5 opinii treningowych (test w Task 3: `test_blank_reviews_ignored`).
2. Ta sama para użytkownik–produkt w kilku opiniach liczy się raz, według najwcześniejszej (Task 3: `test_duplicate_pair_keeps_earliest`).
3. Produkt oceniony przez użytkownika w okresie walidacji i ponownie pozytywnie w teście nie jest ani kandydatem, ani celem (Task 7: `test_valid_period_items_masked`).
4. Serwer LLM zwraca zły JSON, ucina odpowiedź albo jest niedostępny: ponawianie, błąd zapisany w `errors.jsonl`, wznowienie uzupełnia braki, a niekompletny wariant jest pomijany przez `encode` (Task 4: `test_invalid_json_retries_and_errors_resume`, Task 5: `test_incomplete_llm_variant_skipped`).
5. Pusty profil tekstowy w dashboardzie nie wywołuje rankingu i pokazuje komunikat (Task 8: `test_empty_custom_profile_returns_nothing`).

---

### Task 1: Szkielet projektu, `config.py`, `storage.py`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `CLAUDE.md`, `README.md` (krótki: temat, cel, polecenia; pełny opis w Task 9), `src/sys_rek_mag/__init__.py`, `src/sys_rek_mag/config.py`, `src/sys_rek_mag/storage.py`
- Test: `tests/test_config.py`, `tests/test_storage.py`

**Interfaces:**
- Produces:
  - `storage.py`: skopiuj `sys-rek-mag-3/src/sys_rek_mag/storage.py` i dodaj `write_pickle(obj, path)` oraz `read_pickle(path)` (atomowo, jak `write_parquet`). Zachowaj `write_json, read_json, write_jsonl, read_jsonl, write_parquet, write_csv, write_npy, text_hash, file_hash`.
  - `config.py`: wzorzec z sys-rek-mag-3 (`PROJECT_ROOT`, `_load_env_file`, `model_id`, `llm_variant`, `PATHS`). Stałe:
    `SEED=2026, T1=1628643414042, T2=1658002729837, POSITIVE_RATING=4.0, N_ITEMS=1500, MIN_ITEM_TRAIN_REVIEWS=5, MAX_REVIEWS_PER_ITEM=20, REVIEW_MAX_CHARS=800, MIN_EVAL_USERS_WARN=500, TOP_KS=(5, 10), BOOTSTRAP_RESAMPLES=10_000, FAMILY_ALPHA=0.05, PRIMARY_COMPARISONS_PER_MODEL=3, SUPPORTING_CONFIDENCE=0.95, NO_LLM="no_llm", LLM_PREFIX="llm_", MECHANISMS=("tfidf", "dense", "two_tower"), BASELINE="baseline", POPULARITY="popularity", EMBEDDING_MODEL (env, domyślnie "sentence-transformers/all-MiniLM-L6-v2"), MAX_SEQ_LENGTH=512, LLM_BASE_URL, LLM_API_KEY, LLM_MODELS (env, domyślnie "gemma4:e2b"), LLM_HARDWARE`.
    `AMAZON_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"` i `AMAZON_FILES: dict[str, str]` (nazwa lokalna → URL): `Automotive.jsonl.gz` ← `raw/review_categories/Automotive.jsonl.gz`, `meta_Automotive.jsonl.gz` ← `raw/meta_categories/meta_Automotive.jsonl.gz`, `Automotive.{train,valid,test}.csv.gz` ← `benchmark/5core/timestamp_w_his/…`.
  - `Paths(root)` (frozen dataclass): `raw`, `prepared`, `item_texts(variant) -> root/"item_texts"/variant`, `models(variant) -> root/"models"/variant`, `results(variant, mechanism) -> root/"results"/variant/mechanism`, `comparisons -> root/"results"/"comparisons"`, `results_root -> root/"results"`.

- [ ] **Step 1:** W katalogu projektu `uv venv`. Napisz `pyproject.toml` na wzór sys-rek-mag-3 (te same przypięte wersje, `name = "sys-rek-mag-nowy"`, nowy opis tematu), dodaj `scikit-learn` (najnowsza zgodna wersja, przypięta po `uv lock`) i skrypty: `sysrekmag-download = download:main`, `sysrekmag-prepare = dataset:main`, `sysrekmag-llm-texts = llm_texts:main`, `sysrekmag-encode = encode:main`, `sysrekmag-train = two_tower:main`, `sysrekmag-evaluate = evaluate:main`. `.gitignore` i `.env.example` jak w sys-rek-mag-3, bez `KAGGLE_ARCHIVE_DIR`, `LLM_MODELS=gemma4:e2b`. `uv sync`.
- [ ] **Step 2: Testy.** `test_model_id`: `model_id("gemma4:e2b") == "gemma4_e2b"`, `model_id("  ")` rzuca `ValueError`. `test_paths_layout`: `Paths(tmp_path).results("no_llm", "dense") == tmp_path/"results"/"no_llm"/"dense"`, `models("llm_x")`, `item_texts("no_llm")`. `test_write_pickle_roundtrip_atomic`: zapis i odczyt, brak pliku `.tmp` po zapisie. `test_write_json_roundtrip`.
- [ ] **Step 3:** `uv run pytest` → FAIL (brak modułów).
- [ ] **Step 4:** Zaimplementuj `config.py` i `storage.py`. `CLAUDE.md`: krótko, na wzór sys-rek-mag-3 (temat, polecenia, zakaz uruchamiania LLM bez zgody, inwarianty z Global Constraints, „odpowiadaj po polsku”).
- [ ] **Step 5:** `uv run pytest` → PASS.
- [ ] **Step 6: Commit** `git add -A && git commit -m "Szkielet projektu: config, storage, pyproject"`

### Task 2: `download.py`

**Files:** Create `src/sys_rek_mag/download.py`. Test: `tests/test_download.py`

**Interfaces:**
- Produces: `download(paths: Paths = PATHS, files: dict[str, str] = config.AMAZON_FILES, fetch: Callable[[str, Path], None] = _fetch) -> list[Path]` zwraca ścieżki w `paths.raw`. `_fetch(url, target)` pobiera strumieniowo (`urllib.request`, bloki 1 MB, pasek postępu w MB) do `target`.

- [ ] **Step 1: Testy.** `test_download_skips_existing`: istniejący plik nie jest nadpisany, a `fetch` nie jest wołany. `test_download_atomic_on_failure`: `fetch` rzuca wyjątek w połowie, więc nie ma pliku docelowego ani `.tmp` po błędzie, a wyjątek przechodzi dalej. `test_download_fetches_missing`: fałszywy `fetch` zapisuje bajty, plik istnieje.
- [ ] **Step 2:** `uv run pytest tests/test_download.py` → FAIL.
- [ ] **Step 3:** Implementacja: pobieranie do `.<nazwa>.part`, potem `os.replace`, a `.part` jest usuwany w `finally` przy błędzie. `main()` drukuje, co pobrano, a co pominięto.
- [ ] **Step 4:** PASS. **Step 5: Commit** „Pobieranie danych Amazon bez nadpisywania”.

### Task 3: `dataset.py` (etap prepare) i syntetyczne dane testowe

**Files:** Create `src/sys_rek_mag/dataset.py`, `tests/conftest.py`. Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `config`, `storage`.
- Produces (w `paths.prepared`):
  - `items.parquet`: `parent_asin: str, title: str, features: list[str], description: list[str], categories: list[str], train_reviews: int`, posortowane po `parent_asin`.
  - `interactions.parquet`: `user_id: str, parent_asin: str, rating: float, timestamp: int, split: str` (`train`/`valid`/`test`), tylko produkty z próbki, jedna para użytkownik–produkt (najwcześniejsza).
  - `llm_inputs.jsonl`: `{"parent_asin", "reviews": [{"rating", "title", "text"}]}`, posortowane po `parent_asin`.
  - `users.parquet`: `user_id, train_positives: int, eval: bool` (`eval` = ≥ 1 pozytyw w treningu i ≥ 1 pozytyw w teście dla produktu bez interakcji przed `T2`).
  - `manifest.json`: parametry, liczby (produkty, interakcje wg split, użytkownicy eval), odciski plików źródłowych (rozmiar + mtime, bez hashowania 2 GB), `subcategory`.
  - `prepare(paths=PATHS, n_items=config.N_ITEMS, subcategory: str | None = None) -> dict` (manifest).
  - Funkcje pomocnicze (testowane osobno): `select_items(candidates: set[str], meta: pd.DataFrame, train_counts: pd.Series, n_items: int, subcategory: str | None) -> pd.DataFrame`, `pick_reviews(reviews: pd.DataFrame) -> list[dict]`.

Algorytm (dwa przebiegi strumieniowe po `Automotive.jsonl.gz`, `gzip` + `json` linia po linii):
1. Kandydaci: `parent_asin` z sumy trzech plików 5-core, przecięci z metadanymi z niepustym `title` (meta czytana strumieniowo, zachowujemy tylko kandydatów i pola z kontraktu).
2. Przebieg 1: dla kandydatów zlicz opinie z `timestamp < T1` i niepustym `text.strip()`.
3. `select_items`: `train_reviews ≥ MIN_ITEM_TRAIN_REVIEWS`, opcjonalnie filtr `subcategory in categories`, sort `(-train_reviews, parent_asin)`, pierwsze `n_items`.
4. Przebieg 2: dla wybranych produktów zbierz interakcje `(user_id, parent_asin, rating, timestamp)`, zostaw najwcześniejszą parę i przypisz `split`. Zbierz też opinie treningowe z tekstem do `pick_reviews`: sort `(-helpful_vote, -timestamp, user_id)`, pierwsze `MAX_REVIEWS_PER_ITEM`, `text` obcięty do `REVIEW_MAX_CHARS`, `title` opinii (nie produktu!).
5. Gdy liczba użytkowników eval `< MIN_EVAL_USERS_WARN`: ostrzeżenie i wydruk 10 podkategorii z największą sumą `train_reviews` (decyzję o `--subcategory` podejmuje autor i wpisuje ją do protokołu).
6. CLI: `--items N`, `--subcategory NAME`.

- [ ] **Step 1: `conftest.py`.** Fixture `raw(paths)` zapisuje do `paths.raw` mini-pliki gz w formacie Amazon: ~30 produktów w 3 grupach tematycznych (np. `led bulb`, `wiper blade`, `car wax`; słowa grupy w tytule i w tekstach opinii), ~40 użytkowników z pozytywami treningowymi, walidacyjnymi i testowymi w swojej grupie. Przypadki brzegowe: produkt bez tytułu w meta, produkt z 4 opiniami treningowymi, opinia z tekstem `"   "`, zduplikowana para użytkownik–produkt (późniejsza z inną oceną), produkt spoza 5-core, opinie po `T2`. Stałe tych przypadków eksportuj z conftest (jak `EXPOSED_COLD_ID` w sys-rek-mag-3). Fixture `paths` = `Paths(tmp_path / "data")`. `PREPARE_SIZES = {"n_items": 20}`.
- [ ] **Step 2: Testy.** `test_select_items_order_and_threshold` (sort, próg, remis po `parent_asin`). `test_items_without_title_or_outside_5core_excluded`. `test_blank_reviews_ignored` (opinia z białym tekstem nie liczy się do progu i nie trafia do `llm_inputs`). `test_duplicate_pair_keeps_earliest`. `test_splits_by_cutoffs` (`timestamp == T1` → `valid`, `== T2` → `test`). `test_pick_reviews_order_and_truncation`. `test_llm_inputs_only_train_and_no_product_title` (żaden tekst w `llm_inputs` nie pochodzi z `timestamp ≥ T1`, a rekord nie ma pól `title`/`description` produktu). `test_eval_users_definition`. `test_selection_ignores_test_data` (dodanie opinii testowych nie zmienia `items.parquet`).
- [ ] **Step 3:** FAIL. **Step 4:** Implementacja. **Step 5:** PASS.
- [ ] **Step 6: Commit** „Etap prepare: próbka Automotive, interakcje, wejścia LLM”.

### Task 4: `texts.py` i `llm_texts.py` (etap LLM)

**Files:** Create `src/sys_rek_mag/texts.py`, `src/sys_rek_mag/llm_texts.py`. Test: `tests/test_texts.py`, `tests/test_llm_texts.py`

**Interfaces:**
- Consumes: `prepared/items.parquet`, `prepared/llm_inputs.jsonl`.
- Produces:
  - `texts.py`: `no_llm_text(item: dict) -> str` (tytuł, potem cechy, potem opis, oddzielone `". "`, puste pomijane), `llm_input_text(record: dict) -> str` (opinie jako `"Review N (R/5 stars): <title>. <text>"`), `llm_text(generated: dict) -> str` (`description` + `" Features: "` + `"; ".join("aspect: value")`).
  - `llm_texts.py`: pydantic `Feature(aspect: str, value: str)`, `ItemDescription(description: str, features: list[Feature])` z `extra="forbid"`, `description` 20–80 słów, `features` 5–12, etykiety małymi literami i przycięte. `SYSTEM_PROMPT` (EN): analityk produktów czytający **tylko opinie klientów** o jednym produkcie samochodowym; najpierw ustala, czym jest produkt; opisuje, czym jest i do czego służy, mocne i słabe strony według klientów, dla kogo; tylko informacje wynikające z opinii, bez zgadywania; bez liczby gwiazdek i bez popularności w opisie. `RESPONSE_FORMAT` (json_schema strict), `PROMPT_HASH`, `make_client`, `server_info` przeniesione z `sys-rek-mag-3/profiles.py`.
  - `generate_one(client, record: dict, llm_model: str, seed: int) -> tuple[dict, int]` (wynik, liczba prób), `generate(llm_model: str, paths=PATHS, client=None, limit: int | None = None, seed=SEED, quantization="") -> dict` (manifest).
  - Wyjście w `paths.item_texts(llm_variant(model_id(model)))`: `generations.jsonl` (`parent_asin, llm_model, source_hash, prompt_hash, attempts, seconds, output`), `errors.jsonl`, `manifest.json` (jak w sys-rek-mag-3 plus `complete: bool` = wszystkie produkty z `items.parquet`).
  - `load_generations(paths, variant) -> dict[str, dict]`.
- Logika ponawiania, checkpointów, cache po `(source_hash, PROMPT_HASH, llm_model)`, kolejności losowej z seedem dla `--limit` (pilotaż) i szacunku pozostałego czasu: przenieś z `generate_profiles` w sys-rek-mag-3. CLI: `--model` (wymagane), `--base-url`, `--limit`, `--quantization`, opis ostrzega, że uruchamia LLM.

- [ ] **Step 1: Testy.** `test_no_llm_text_skips_empty_parts`. `test_llm_text_contains_only_generated` (dla produktu, którego tytuł zawiera unikalne słowo `"ZZTITLE"`, `llm_text` z wygenerowanego wyniku go nie zawiera, a `llm_input_text` nie zawiera tytułu produktu). Fałszywy klient (`FakeClient` w `conftest.py`, zwraca odpowiedzi z kolejki i liczy wywołania): `test_generate_writes_and_caches` (drugie uruchomienie = 0 wywołań). `test_invalid_json_retries_and_errors_resume` (zły JSON 3× → rekord w `errors.jsonl`, `complete=False`, `SystemExit`; ponowne uruchomienie z poprawnym klientem uzupełnia braki i usuwa `errors.jsonl`). `test_truncated_response_retried` (`finish_reason="length"`). `test_feature_count_validated`.
- [ ] **Step 2:** FAIL. **Step 3:** Implementacja. **Step 4:** PASS.
- [ ] **Step 5: Commit** „Etap LLM: opisy i cechy produktów wyłącznie z opinii”.

### Task 5: `encode.py` (teksty wariantów, embeddingi, TF-IDF)

**Files:** Create `src/sys_rek_mag/encode.py`. Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `items.parquet`, `load_generations`, `texts.*`.
- Produces:
  - `complete_llm_variants(paths) -> list[str]` (warianty `llm_*` z `manifest.complete == True`).
  - `variant_texts(paths, items) -> dict[str, list[str]]` (`no_llm` + kompletne warianty LLM, w porządku `items`).
  - Dla każdego wariantu: `item_texts(v)/texts.parquet` (`parent_asin, text`), `models(v)/embeddings.npy` (float32, L2-norm, porządek `items`), `models(v)/tfidf.npz` (`scipy.sparse.save_npz`, przez plik tymczasowy i `os.replace`), `models(v)/tfidf_vectorizer.pkl`, `models(v)/manifest.json` (odcisk = hash enkodera + `MAX_SEQ_LENGTH` + `parent_asin` + teksty; statystyki tokenów jak w sys-rek-mag-3).
  - TF-IDF: `TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")`, fit tylko na tekstach danego wariantu.
  - `load_encoder`, `encode(encoder, texts) -> np.ndarray`: przenieś z `sys-rek-mag-3/embeddings.py`.
  - `load_variant(paths, variant) -> dict` z kluczami `embeddings, tfidf, vectorizer`.
  - `encode_all(paths=PATHS, encoder=None) -> dict`. Wariant z niezmienionym odciskiem jest pomijany.
- `FakeEncoder` w `conftest.py`: deterministyczny worek słów haszowany do 32 wymiarów, z L2-normalizacją. Ma `encode(texts, **kw)`, `tokenizer(texts) -> {"input_ids": [...]}` i `get_sentence_embedding_dimension()`.

- [ ] **Step 1: Testy.** `test_encode_all_writes_variants`. `test_cache_skips_unchanged` (drugie uruchomienie nie woła encodera). `test_incomplete_llm_variant_skipped`. `test_row_order_matches_items`.
- [ ] **Step 2:** FAIL. **Step 3:** Implementacja. **Step 4:** PASS. **Step 5: Commit** „Etap encode: teksty wariantów, embeddingi, TF-IDF”.

### Task 6: `recommend.py` i `two_tower.py`

**Files:** Create `src/sys_rek_mag/recommend.py`, `src/sys_rek_mag/two_tower.py`. Test: `tests/test_recommend.py`, `tests/test_two_tower.py`

**Interfaces:**
- `recommend.py` (czyste funkcje):
  - przenieś z sys-rek-mag-3: `normalize_rows`, `mean_profiles(history: csr, item_vectors) -> ndarray` (dla rzadkich `item_vectors` zwraca gęstą macierz), `top_k(scores, valid, k)`;
  - `ranking_metrics(top, top_valid, relevant, ks=TOP_KS) -> dict[str, ndarray]` z kluczami `ndcg_at_{k}, precision_at_{k}, recall_at_{k}, hit_rate_at_{k}` dla każdego k oraz `mrr_at_{max k}`. `top` ma długość `max(ks)`, wiersze bez celów dają NaN. Precision@k = trafienia / k;
  - `dense_scores(history: csr, embeddings) -> ndarray` = cosinus średniego profilu z produktami;
  - `tfidf_scores(history: csr, tfidf: csr) -> ndarray` = to samo na TF-IDF (profil L2-normalizowany);
  - `popularity_scores(n_users: int, counts: ndarray) -> ndarray` = `log1p(counts)` rozgłoszone na wiersze;
  - `query_scores(query_vector: ndarray, item_vectors) -> ndarray` (1 × N, dla profilu tekstowego autora).
- `two_tower.py`:
  - `class TwoTower(nn.Module)`: `user_tower`, `item_tower` = `Linear(d, 256) → ReLU → Linear(256, 128)`, `forward(user_inputs, item_inputs)` zwraca znormalizowane wektory. Stałe: `HIDDEN=256, OUT=128, TEMPERATURE=0.05, LR=1e-3, BATCH=256, MAX_EPOCHS=50, PATIENCE=5`;
  - przykłady treningowe: dla użytkownika z ≥ 2 pozytywami treningowymi każdy pozytyw jest celem, a wejście użytkownika to średnia embeddingów pozostałych pozytywów; strata to softmax z negatywami w batchu (`cross_entropy` na `u @ i.T / TEMPERATURE`);
  - wczesne zatrzymanie na walidacji: użytkownicy z ≥ 1 pozytywem treningowym i pozytywem walidacyjnym; profil z treningu; cele to pozytywy z `split=="valid"` dla produktów bez interakcji treningowej; maskowanie produktów z treningu; metryka NDCG@10 (z `recommend.ranking_metrics`);
  - `train_variant(paths, variant, seed=SEED) -> dict` zapisuje `models(variant)/two_tower.pt` (state_dict) i `two_tower_manifest.json` (hiperparametry, krzywa walidacji, najlepsza epoka, odcisk = hash `embeddings.npy` + `interactions.parquet`; cache przy zgodnym odcisku);
  - `two_tower_scores(model, history: csr, embeddings) -> ndarray` i `two_tower_query(model, query_embedding, embeddings) -> ndarray`;
  - `load_two_tower(paths, variant) -> TwoTower`;
  - `main()`: trenuje wszystkie warianty z `models/*/embeddings.npy`.
- Determinizm: `torch.manual_seed(seed)`, `torch.Generator().manual_seed(seed)` do tasowania, `torch.use_deterministic_algorithms(True)`.

- [ ] **Step 1: Testy recommend.** `test_ranking_metrics_known_values` (ręcznie policzony przykład: 2 użytkowników, 6 produktów, wszystkie metryki dla k=5 i 10). `test_ties_break_by_lower_index`. `test_rows_without_targets_nan`. `test_dense_and_tfidf_prefer_same_group` (na `FakeEncoder` / małym TF-IDF produkt z tej samej grupy wygrywa).
- [ ] **Step 2: Testy two_tower.** `test_training_deterministic` (dwa treningi z tym samym seedem dają identyczne wagi). `test_two_tower_learns_groups` (na danych z conftest NDCG@10 na walidacji po treningu > losowy ranking). `test_cache_reused`.
- [ ] **Step 3:** FAIL. **Step 4:** Implementacja. **Step 5:** PASS. **Step 6: Commit** „Mechanizmy: TF-IDF, embeddingi, sieć dwuwieżowa”.

### Task 7: `evaluate.py`

**Files:** Create `src/sys_rek_mag/evaluate.py`. Test: `tests/test_evaluate.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `prepared/*`, `encode.load_variant`, `two_tower.load_two_tower`, `recommend.*`.
- Produces:
  - `build_matrices(items, interactions, users) -> dict` z kluczami `eval_users: ndarray[str]`, `history: csr` (pozytywy `< T1`), `seen: csr` (dowolna interakcja `< T2`), `relevant: ndarray[bool]` (pozytywy `≥ T2` & ~seen), `train_counts: ndarray` (liczba interakcji `< T1` na produkt);
  - `per_user_metrics(paths, seed) -> pd.DataFrame`: kolumny `user_id, variant, mechanism, <metryki>`; wiersze dla każdego wariantu × `MECHANISMS` oraz `variant="baseline", mechanism="popularity"`; przetwarzanie w batchach 4000 użytkowników;
  - zapis do `results(variant, mechanism)`: `per_user.parquet`, `metrics.json` (średnie, `users`), `manifest.json` (seed, k, odciski wejść);
  - `comparison_specs(variants: list[str]) -> list[dict]`: dla każdego `llm_*` × mechanizm wszystkie metryki vs `no_llm`; `role="primary"` dla `ndcg_at_10`, ufność `1 - FAMILY_ALPHA / PRIMARY_COMPARISONS_PER_MODEL` (98,33%); pozostałe `supporting` przy 95%;
  - `compare(per_user, seed) -> pd.DataFrame` → `results/comparisons/comparisons.csv`; bootstrap i kolumny wyniku jak w `sys-rek-mag-3/evaluate.py` (`_bootstrap` z ziarnem zależnym od etykiety porównania);
  - `evaluate(paths=PATHS, seed=SEED) -> dict` + `main()` (drukuje porównania główne).

- [ ] **Step 1: Testy.** `test_valid_period_items_masked` (produkt z interakcją w `[T1, T2)` nie jest kandydatem ani celem). `test_only_eval_users_scored`. `test_comparison_specs_primary_family` (dla 1 modelu: 3 primary, ufność 0,98333…; dodanie drugiego modelu nie zmienia ufności). `test_results_layout` (pliki w `results/<wariant>/<mechanizm>/` i `results/baseline/popularity/`).
- [ ] **Step 2: `test_pipeline.py`.** Cały potok na danych z conftest: prepare → generate (`FakeClient`, opisy z prawdziwych słów grupy) → encode (`FakeEncoder`) → train → evaluate. Asercje: wszystkie warianty × mechanizmy mają wyniki, NDCG@10 dense > popularity dla `no_llm`, deterministyczny wynik przy dwóch uruchomieniach.
- [ ] **Step 3:** FAIL. **Step 4:** Implementacja. **Step 5:** PASS. **Step 6: Commit** „Ewaluacja: metryki, porównania, bootstrap”.

### Task 8: Dashboard i słownik pojęć

**Files:** Create `src/sys_rek_mag/dashboard.py`, `src/sys_rek_mag/glossary.py`, `dashboard.py` (5 linii jak w sys-rek-mag-3). Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `results/*/*/metrics.json`, `results/comparisons/comparisons.csv`, `item_texts/llm_*/manifest.json`, `generations.jsonl` (czasy), `encode.load_variant`, `two_tower.*`, `recommend.query_scores`.
- Produces:
  - `discover_results(paths) -> pd.DataFrame` z kolumnami `variant, mechanism, users, <metryki>`, zbudowane wyłącznie ze skanu folderów;
  - `custom_top_n(paths, variant, mechanism, text: str, n: int, encoder) -> pd.DataFrame` (`parent_asin, title, score`); pusty lub biały `text` → pusta ramka bez wywołania encodera. TF-IDF: `vectorizer.transform`, dense: encoder, two_tower: `two_tower_query`. Tytuł w tabeli wynikowej jest tylko do wyświetlenia;
  - UI: boczny panel z wyborem modelu LLM (warianty `llm_*` znalezione w wynikach), mechanizmu i liczby N (5/10); zakładki **Wyniki** (tabela wszystkich wariantów dla mechanizmu + popularność, wykres słupkowy NDCG@10 i różnice z przedziałami ufności z `comparisons.csv`), **Koszt LLM** (średni i medianowy czas, sprzęt z manifestu), **Mój profil** (pole tekstowe z podpowiedzią „Opisz po angielsku…”, top-N obok siebie: `no_llm` vs wybrany `llm_*`, plus dla każdego produktu wygenerowany opis LLM w rozwijanym polu), **Słownik**. Brak wyników → komunikat, który etap uruchomić. Kolory wariantów stałe (wzorzec `SLOT_COLORS` z sys-rek-mag-3).
  - `glossary.py`: `GLOSSARY: dict[str, str]` (TF-IDF, embedding, sieć dwuwieżowa, NDCG@k, precyzja@k, czułość@k, HitRate@k, MRR, podział czasowy, bootstrap, przedział ufności, Bonferroni, wariant, mechanizm, popularność), prosty język z przykładami z Automotive.

- [ ] **Step 1: Testy.** `test_discover_results_from_folders` (dodanie folderu `results/llm_other/dense/` pojawia się bez zmian w kodzie). `test_empty_custom_profile_returns_nothing` (encoder, który rzuca przy wywołaniu, nie jest wołany). `test_custom_top_n_relevant_group` (tekst „led bulb bright” daje produkty z grupy LED na górze dla dense i tfidf). Test UI przez `streamlit.testing.v1.AppTest` jak w `sys-rek-mag-3/tests/test_dashboard.py`: aplikacja startuje na danych z potoku testowego bez wyjątków.
- [ ] **Step 2:** FAIL. **Step 3:** Implementacja. **Step 4:** PASS. **Step 5: Commit** „Dashboard: dynamiczne wyniki, mój profil, słownik”.

### Task 9: Dokumentacja i przewodnik dla studenta

**Files:** Create `docs/PRZEWODNIK.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/DATA.md`, `docs/MODELS.md`, `docs/LICENSES.md`. Modify: `README.md`, `CLAUDE.md`.

- `EXPERIMENT_PROTOCOL.md`: pytanie badawcze, dane i próbka, podział czasowy, użytkownicy eval, maskowanie, warianty, 3 mechanizmy + popularność, hiperparametry, metryki, rodzina porównań i poziomy ufności, pilotaż i reguła ustalenia N, reguła podkategorii, „Rejestr zmian” (pierwszy wpis z datą). Wartości dokładnie z `config.py`.
- `DATA.md`: pliki, URL-e, rozmiary (2,29 GB opinie, 1,27 GB meta, 0,16 GB 5-core), pola, cytowanie Hou i in. 2024 (arXiv:2403.03952), wymagane miejsce na dysku.
- `MODELS.md`: `gemma4:e2b`, Ollama Q4_K_M, RX 6600 8 GB; sekcja „Pilotaż” do wypełnienia; plan kolejnych modeli.
- `LICENSES.md`: brak formalnej licencji Amazon Reviews 2023, wymóg cytowania, surowych danych nie publikujemy, do potwierdzenia z promotorem; licencje bibliotek i MiniLM (Apache-2.0); licencja Gemma.
- `PRZEWODNIK.md` (spec, rozdz. 11): projekt w 5 zdaniach; schemat przepływu (blok tekstowy ASCII: pliki → etapy → foldery); słowniczek z przykładami (te same pojęcia co `glossary.py`); mapa folderów i modułów; krok po kroku dla każdego etapu (co robi, co czyta, co zapisuje, ile trwa: download ~3,7 GB, prepare kilkanaście minut, LLM ~4 h, encode i train minuty, kiedy uruchomić ponownie, np. po zmianie promptu trzeba ponownie generować, a encode, train i evaluate przeliczają się same dzięki odciskom); jak uruchomić pilotaż (`--limit 20`, **tylko świadomie, bo uruchamia LLM**); jak dodać model; jak czytać wyniki i dashboard; najczęstsze problemy (Ollama nie działa, brak miejsca, niekompletny wariant). Prosty język, krótkie akapity, przykłady z Automotive.
- `README.md`: temat PL/EN, cel, krótko o potoku, polecenia, odnośnik do przewodnika.

- [ ] **Step 1:** Napisz dokumenty. Sprawdź, że każda liczba w protokole zgadza się z `config.py` (`grep` stałych).
- [ ] **Step 2:** `uv run pytest` → PASS (cały zestaw).
- [ ] **Step 3: Commit** „Dokumentacja, protokół eksperymentu i przewodnik dla studenta”.

---

## Po planie (poza zakresem zadań, każde za zgodą autora)

1. `uv run sysrekmag-download` (~3,7 GB), `uv run sysrekmag-prepare`: sprawdzić liczbę użytkowników eval.
2. Pilotaż `uv run sysrekmag-llm-texts --model gemma4:e2b --limit 20` (**LLM, tylko za zgodą**): ocena opisów bez tytułu z autorem, czas na produkt. Na tej podstawie ustalić `N_ITEMS` w protokole i zacommitować protokół.
3. Pełne generowanie, encode, train, evaluate, dashboard.
