from sys_rek_mag.texts import llm_input_text, llm_text, no_llm_text


def test_no_llm_text_skips_empty_parts():
    assert no_llm_text({"title": "T", "features": ["a", "b"], "description": []}) == "T. a. b"
    assert no_llm_text({"title": "T", "features": [], "description": ["d"]}) == "T. d"


def test_llm_input_text_format_and_no_product_title():
    record = {"parent_asin": "A", "reviews": [{"rating": 4.0, "title": "Great", "text": "Works."}]}
    assert llm_input_text(record) == "Review 1 (4/5 stars): Great. Works."


def test_llm_text_contains_only_generated():
    item = {"parent_asin": "A", "title": "ZZTITLE bulb", "features": ["ZZTITLE feature"], "description": ["ZZTITLE desc"]}
    record = {"parent_asin": "A", "reviews": [{"rating": 5.0, "title": "Nice", "text": "Bright."}]}
    generated = {"description": "A bulb.", "features": [{"aspect": "light", "value": "bright"}, {"aspect": "fit", "value": "easy"}]}
    assert "ZZTITLE" in no_llm_text(item)  # kontrola: wariant no_llm zawiera tytuł, więc wyciek byłby widoczny
    assert llm_text(generated) == "A bulb. Features: light: bright; fit: easy"
    assert "ZZTITLE" not in llm_text(generated)
    assert "ZZTITLE" not in llm_input_text(record)
