"""Fixed prompt set: schema, loader and validator.

Each JSONL line: {"id", "text", "length_bucket", "category"}.
Length buckets are defined by whitespace word count so they are reproducible:
short <= 12 words, medium 13-40, long >= 41.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

SHORT_MAX_WORDS = 12
MEDIUM_MAX_WORDS = 40
CATEGORIES = {
    "plain",
    "numbers",
    "names",
    "punctuation",
    "abbreviations",
    "dates_times",
    "questions",
    "technical",
    "dialogue",
    "long_paragraph",
}
MIN_PROMPTS, MAX_PROMPTS = 60, 90


def bucket_for(text: str) -> str:
    n = len(text.split())
    if n <= SHORT_MAX_WORDS:
        return "short"
    if n <= MEDIUM_MAX_WORDS:
        return "medium"
    return "long"


class Prompt(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    id: str
    text: str
    length_bucket: Literal["short", "medium", "long"]
    category: str

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not v or any(c.isspace() for c in v):
            raise ValueError("id must be non-empty without whitespace")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be non-empty")
        if v != v.strip():
            raise ValueError("text must not have leading/trailing whitespace")
        return v

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v!r}; allowed: {sorted(CATEGORIES)}")
        return v

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def load_prompts(path: str | Path, limit: int | None = None) -> list[Prompt]:
    prompts = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(Prompt.model_validate_json(line))
    return prompts[:limit] if limit else prompts


def validate_prompt_file(path: str | Path, enforce_count: bool = True) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errors: list[str] = []
    ids: set[str] = set()
    texts: set[str] = set()
    buckets: dict[str, int] = {}
    n = 0
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                raw = json.loads(line)
                p = Prompt.model_validate(raw)
            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"line {lineno}: {e}")
                continue
            if p.id in ids:
                errors.append(f"line {lineno}: duplicate id {p.id}")
            if p.text in texts:
                errors.append(f"line {lineno}: duplicate text")
            ids.add(p.id)
            texts.add(p.text)
            expect = bucket_for(p.text)
            if p.length_bucket != expect:
                errors.append(f"line {lineno}: {p.id} has {p.n_words} words -> bucket {expect}, file says {p.length_bucket}")
            buckets[p.length_bucket] = buckets.get(p.length_bucket, 0) + 1
    if enforce_count and not (MIN_PROMPTS <= n <= MAX_PROMPTS):
        errors.append(f"expected {MIN_PROMPTS}-{MAX_PROMPTS} prompts, found {n}")
    for b in ("short", "medium", "long"):
        if enforce_count and buckets.get(b, 0) == 0:
            errors.append(f"no prompts in bucket {b}")
    return errors


def prompt_set_hash(path: str | Path) -> str:
    """sha256 of the prompt file bytes; recorded in every run for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
