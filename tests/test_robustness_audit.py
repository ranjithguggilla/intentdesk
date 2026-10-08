import random

import numpy as np

from intentdesk.audit import confused_pairs, flag_issues
from intentdesk.robustness import add_typos, perturb_series


def test_zero_rate_is_identity():
    assert add_typos("my card is missing", 0.0, random.Random(1)) == "my card is missing"


def test_typos_change_text_and_are_reproducible():
    a = perturb_series(["where is my new card"] * 5, 0.3, 11)
    b = perturb_series(["where is my new card"] * 5, 0.3, 11)
    assert a == b
    assert any(t != "where is my new card" for t in a)


def test_typo_rate_roughly_matches():
    rng = random.Random(0)
    text = "abcdefghij" * 200
    out = add_typos(text, 0.1, rng)
    assert 0.5 * len(text) < len(out) < 1.5 * len(text)
    assert out != text


def test_spaces_and_punctuation_untouched():
    out = add_typos("a b? c!", 1.0, random.Random(2))
    assert "?" in out and "!" in out


def test_flag_issues_finds_flipped_label():
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    probs = np.full((n, 3), 0.05)
    probs[np.arange(n), y] = 0.9
    y_noisy = y.copy()
    y_noisy[0] = 1  # row 0 truly belongs to class 0
    flagged = flag_issues(probs, y_noisy)
    assert 0 in set(flagged.tolist())


def test_confused_pairs_merges_directions():
    t = np.array(["a", "b", "a", "b", "c"])
    p = np.array(["b", "a", "b", "a", "c"])
    out = confused_pairs(t, p)
    assert len(out) == 1 and out.iloc[0]["errors"] == 4
