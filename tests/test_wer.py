"""WER math against hand-computed cases + frozen normalizer behaviour."""

import pytest

from tts_bench.quality import WordErrors, corpus_wer, word_errors
from tts_bench.textnorm import NORMALIZER_VERSION, int_to_words, normalize, ordinal_words


def test_single_deletion():
    e = word_errors("the cat sat on the mat", "the cat sat on mat")
    assert (e.substitutions, e.deletions, e.insertions, e.ref_words) == (0, 1, 0, 6)
    assert e.wer == pytest.approx(1 / 6)


def test_single_substitution_and_insertion():
    e = word_errors("a b c", "a x c")
    assert (e.substitutions, e.deletions, e.insertions) == (1, 0, 0)
    e = word_errors("a b", "a b c")
    assert (e.substitutions, e.deletions, e.insertions, e.ref_words) == (0, 0, 1, 2)
    assert e.wer == pytest.approx(0.5)


def test_mixed_errors_hand_computed():
    # ref: the quick brown fox jumps ; hyp: the quack fox jumps high
    # optimal alignment: quick->quack (S), brown (D), high (I); all-substitution path costs 4
    e = word_errors("the quick brown fox jumps", "the quack fox jumps high")
    assert (e.substitutions, e.deletions, e.insertions) == (1, 1, 1)
    assert e.wer == pytest.approx(3 / 5)


def test_corpus_wer_is_ratio_of_sums_not_mean_of_ratios():
    a = WordErrors(0, 1, 0, 6)  # 1/6
    b = WordErrors(1, 1, 0, 3)  # 2/3
    assert corpus_wer([a, b]) == pytest.approx(3 / 9)
    assert corpus_wer([a, b]) != pytest.approx((1 / 6 + 2 / 3) / 2)


def test_failed_synthesis_counts_as_full_deletion():
    e = word_errors("please leave the folder", "")
    assert (e.deletions, e.ref_words, e.wer) == (4, 4, 1.0)
    # and it enters the corpus total instead of being dropped
    assert corpus_wer([WordErrors(0, 0, 0, 4), e]) == pytest.approx(4 / 8)


def test_empty_reference():
    e = word_errors("", "hello")
    assert (e.insertions, e.ref_words, e.wer) == (1, 0, None)
    assert corpus_wer([e]) is None


def test_hypothesis_more_errors_than_words():
    e = word_errors("hi", "oh hello there friend")
    assert e.errors == 4 and e.wer == pytest.approx(4.0)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The total is $48.75.", "the total is forty eight point seven five dollars"),
        ("Dr. Okonkwo met Mr. Abernathy", "doctor okonkwo met mister abernathy"),
        ("March 3rd, the 14th and the 20th", "march third the fourteenth and the twentieth"),
        ("4,200 respondents", "four thousand two hundred respondents"),
        ("Don't stop", "dont stop"),
        ("at 6:45", "at six forty five"),
        ("68% said", "sixty eight percent said"),
        ("e.g. NASA at 9 a.m.", "eg nasa at nine am"),
        ("The forecast — sunny", "the forecast sunny"),
        ("least-recently-used", "least recently used"),
        ("“Quoted” ‘text’", "quoted text"),
        ("Tom & Jerry", "tom and jerry"),
    ],
)
def test_normalizer_rules(raw, expected):
    assert normalize(raw) == expected


def test_number_words():
    assert int_to_words(0) == "zero"
    assert int_to_words(15) == "fifteen"
    assert int_to_words(1969) == "one thousand nine hundred sixty nine"
    assert int_to_words(1_000_001) == "one million one"
    assert ordinal_words(12) == "twelfth"
    assert ordinal_words(21) == "twenty first"
    assert ordinal_words(40) == "fortieth"


def test_normalization_makes_formatting_differences_free():
    e = word_errors("It's 3 p.m. on the 2nd.", "its three pm on the second")
    assert e.errors == 0


def test_normalizer_version_is_frozen():
    assert NORMALIZER_VERSION == "tts-bench-norm-v1"
