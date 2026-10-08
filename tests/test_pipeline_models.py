import json

import pandas as pd

from intentdesk.cli import main as cli_main
from intentdesk.models import candidates, compare, make_pipeline
from intentdesk.persist import load_bundle, save_bundle
from intentdesk.pipeline import pick_winner, topk_accuracy


def test_pick_winner_prefers_cheaper_model_inside_tolerance():
    t = pd.DataFrame(
        {
            "model": ["slow", "fast", "bad"],
            "val_accuracy": [0.915, 0.913, 0.80],
            "fit_seconds": [20.0, 1.0, 0.1],
        }
    )
    assert pick_winner(t) == "fast"
    t.loc[1, "val_accuracy"] = 0.90
    assert pick_winner(t) == "slow"


def test_every_candidate_fits_on_toy_data(toy_frame):
    for name, (kind, est) in candidates(1).items():
        if "kNN" in name:
            est.set_params(n_neighbors=3)
        pipe = make_pipeline(kind, est)
        pipe.fit(toy_frame["text"], toy_frame["intent"])
        assert len(pipe.predict(["my card has not arrived"])) == 1


def test_compare_returns_sorted_table(toy_frame):
    tbl = compare(toy_frame, toy_frame, 1)
    assert tbl["val_accuracy"].is_monotonic_decreasing
    assert tbl.loc[tbl.model == "majority class", "val_accuracy"].iloc[0] <= 0.3


def test_topk_accuracy():
    import numpy as np

    sc = np.array([[0.1, 0.9, 0.0], [0.5, 0.3, 0.2]])
    y = np.array(["b", "b"])
    assert topk_accuracy(sc, ["a", "b", "c"], y, k=1) == 0.5
    assert topk_accuracy(sc, ["a", "b", "c"], y, k=2) == 1.0


def test_persist_roundtrip_and_cli(toy_bundle, tmp_path, capsys):
    path = tmp_path / "m.skops"
    save_bundle(toy_bundle, path)
    again = load_bundle(path)
    assert again.threshold == toy_bundle.threshold
    assert cli_main(["my pin is blocked", "--model", str(path)]) == 0
    assert "pin_blocked" in capsys.readouterr().out
    csv = tmp_path / "q.csv"
    csv.write_text("text\nmy top up failed\n")
    assert cli_main(["--csv", str(csv), "--model", str(path)]) == 0
    assert "top_up_failed" in capsys.readouterr().out


def test_cli_missing_model(tmp_path):
    assert cli_main(["hello", "--model", str(tmp_path / "nope.skops")]) == 2


def test_committed_metrics_are_consistent():
    from intentdesk.config import ARTIFACT_DIR

    p = ARTIFACT_DIR / "metrics.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    assert 0.85 < m["test"]["accuracy"] < 1
    assert m["test"]["ece_temperature_scaled"] < m["test"]["ece_raw_softmax"]
    assert m["data"]["intents"] == 77
    assert 0 < m["routing"]["auto_routed_share"] <= 1
