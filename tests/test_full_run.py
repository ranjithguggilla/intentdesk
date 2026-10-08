import json

import pandas as pd

from intentdesk.pipeline import run


def _write_tiny_dataset(d):
    topics = {
        "card_arrival": ["my card has not arrived", "where is my new card", "card is late", "still no card"],
        "exchange_rate": ["what exchange rate do you use", "currency rate question", "rate for dollars", "fx rate"],
        "pin_blocked": ["my pin is blocked", "wrong pin froze it", "pin locked", "cannot use pin"],
    }
    rows = []
    for intent, texts in topics.items():
        for rep in range(8):
            for t in texts:
                rows.append({"text": f"{t} {'please ' * (rep % 3)}v{rep}".strip(), "category": intent})
    df = pd.DataFrame(rows)
    df.to_csv(d / "train.csv", index=False)
    df.sample(frac=0.4, random_state=1).to_csv(d / "test.csv", index=False)
    (d / "categories.json").write_text(json.dumps(sorted(topics)))


def test_quick_pipeline_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_tiny_dataset(data)
    m = run(out_dir=tmp_path / "out", asset_dir=tmp_path / "assets", data_dir=data, seed=3, quick=True)
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "intent_router.skops").exists()
    assert len(list((tmp_path / "assets").glob("*.png"))) == 6
    assert m["test"]["accuracy"] > 0.9
    assert 0 <= m["routing"]["threshold"] <= 0.99
    assert m["typo_stress"]["word + char"][0] >= m["typo_stress"]["word + char"][-1]
