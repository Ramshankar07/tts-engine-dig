"""Frozen English text normalizer for WER.

Applied identically to reference text and ASR hypotheses *before* alignment.
The rules are frozen under NORMALIZER_VERSION; any change must bump the version
so WER numbers from different versions are never mixed (report.py prints it).

Rules (in order):
 1. Unicode NFKC, lowercase, typographic quotes/dashes -> ASCII / space
 2. "&" -> " and ", "%" -> " percent", "$X" -> "X dollars"
 3. drop thousands separators (4,200 -> 4200); times 6:45 -> 6 45; a/b -> a b
 4. ordinals 3rd/14th -> third/fourteenth; decimals 2.5 -> two point five
 5. integers -> cardinal words (1969 -> one thousand nine hundred sixty nine)
 6. strip periods from abbreviations, expand a tiny title list (dr -> doctor)
 7. remove apostrophes (don't -> dont) and all other punctuation; hyphens -> space
 8. collapse whitespace

This is intentionally simpler than Whisper's EnglishTextNormalizer and has no
external dependency; it is the same on the laptop and the GPU box.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "tts-bench-norm-v1"

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORD_IRREG = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}
_TITLES = {"dr": "doctor", "mr": "mister", "mrs": "missus", "mt": "mount"}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + int_to_words(r))
    for value, word in _SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return int_to_words(q) + " " + word + ("" if r == 0 else " " + int_to_words(r))
    raise AssertionError("unreachable")


def ordinal_words(n: int) -> str:
    words = int_to_words(n).split()
    last = words[-1]
    if last in _ORD_IRREG:
        last = _ORD_IRREG[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return " ".join(words[:-1] + [last])


def _digits_to_words(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": " ", "–": " "}))
    s = s.replace("&", " and ")
    s = re.sub(r"(\d)\s*%", r"\1 percent", s)
    s = re.sub(r"\$\s*(\d[\d,]*(?:\.\d+)?)", r"\1 dollars", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 4,200 -> 4200 (repeatable)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(\d+):(\d{2})\b", r"\1 \2", s)
    s = re.sub(r"(\d+)/(\d+)", r"\1 \2", s)
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1))), s)
    s = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: f"{int_to_words(int(m.group(1)))} point {_digits_to_words(m.group(2))}", s)
    s = re.sub(r"\b\d+\b", lambda m: int_to_words(int(m.group(0))), s)
    # abbreviations: "a.m." -> "am", "e.g." -> "eg", "dr." -> "dr"
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    s = re.sub(r"\b(dr|mr|mrs|ms|mt|st|jr|sr)\.", r"\1", s)
    s = " ".join(_TITLES.get(w, w) for w in s.split())
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
