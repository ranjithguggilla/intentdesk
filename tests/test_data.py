import pandas as pd
import pytest

from intentdesk.config import DATA_DIR
from intentdesk.data import exact_overlap, load_frame, load_intents, load_splits


def test_splits_are_disjoint_and_stratified():
    train, val, test = load_splits()
    assert len(train) + len(val) == 10003
    assert len(test) == 3080
    assert train.intent.nunique() == 77 == val.intent.nunique()
    share = val.intent.value_counts(normalize=True)
    assert share.max() < 0.03 and share.min() > 0.001


def test_split_is_deterministic():
    a = load_splits()[1]
    b = load_splits()[1]
    assert a.equals(b)


def test_intents_cover_labels():
    intents = set(load_intents())
    train, _, test = load_splits()
    assert set(train.intent) <= intents and set(test.intent) <= intents
    assert len(intents) == 77


def test_load_frame_rejects_bad_schema(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_frame(p)


def test_load_frame_rejects_empty_text(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("text,category\n   ,card_arrival\n")
    with pytest.raises(ValueError):
        load_frame(p)


def test_exact_overlap_ignores_case_and_punctuation():
    a = pd.DataFrame({"text": ["Where is my card?"], "intent": ["x"]})
    b = pd.DataFrame({"text": ["where is my CARD", "something else"], "intent": ["x", "y"]})
    assert exact_overlap(a, b) == 1


def test_source_files_present():
    for f in ("train.csv", "test.csv", "categories.json"):
        assert (DATA_DIR / f).exists()
