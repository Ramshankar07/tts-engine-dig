import json

from tts_bench.prompts import CATEGORIES, bucket_for, load_prompts, prompt_set_hash, validate_prompt_file


def test_shipped_prompt_set_is_valid(repo):
    path = repo / "prompts" / "p1_prompts.jsonl"
    assert validate_prompt_file(path) == []
    ps = load_prompts(path)
    assert 60 <= len(ps) <= 90
    assert {p.length_bucket for p in ps} == {"short", "medium", "long"}
    assert {p.category for p in ps} <= CATEGORIES
    assert len({p.id for p in ps}) == len(ps)
    # variety the plan asks for: numbers, names, punctuation, long paragraphs
    cats = {p.category for p in ps}
    assert {"numbers", "names", "punctuation", "long_paragraph"} <= cats
    assert len(prompt_set_hash(path)) == 64


def test_bucket_boundaries():
    assert bucket_for(" ".join(["w"] * 12)) == "short"
    assert bucket_for(" ".join(["w"] * 13)) == "medium"
    assert bucket_for(" ".join(["w"] * 40)) == "medium"
    assert bucket_for(" ".join(["w"] * 41)) == "long"


def test_validator_catches_problems(tmp_path):
    bad = [
        {"id": "a", "text": "Hello there.", "length_bucket": "short", "category": "plain"},
        {"id": "a", "text": "Another one.", "length_bucket": "short", "category": "plain"},  # dup id
        {"id": "b", "text": "Hello there.", "length_bucket": "short", "category": "plain"},  # dup text
        {"id": "c", "text": "Too short for long.", "length_bucket": "long", "category": "plain"},  # bucket
        {"id": "d", "text": "x", "length_bucket": "short", "category": "nope"},  # category
        {"id": "e", "text": "x", "length_bucket": "short", "category": "plain", "extra": 1},  # extra field
        {"id": "f", "text": " padded", "length_bucket": "short", "category": "plain"},  # whitespace
    ]
    p = tmp_path / "bad.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in bad) + "\nnot json\n")
    errs = validate_prompt_file(p)
    joined = "\n".join(errs)
    assert "duplicate id a" in joined
    assert "duplicate text" in joined
    assert "bucket short" in joined
    assert "unknown category" in joined
    assert "line 6" in joined and "line 7" in joined and "line 8" in joined
    assert "expected 60-90 prompts" in joined
