import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.svm import LinearSVC  # noqa: E402

from intentdesk.calibration import fit_temperature  # noqa: E402
from intentdesk.models import make_pipeline, scores  # noqa: E402
from intentdesk.router import Bundle  # noqa: E402

TOPICS = {
    "card_arrival": ["my card has not arrived", "where is my new card", "card still not here after weeks"],
    "exchange_rate": [
        "what exchange rate do you use",
        "how is the currency rate decided",
        "rate for converting dollars",
    ],
    "pin_blocked": ["my pin is blocked", "i entered the wrong pin and it froze", "pin locked after tries"],
    "top_up_failed": ["my top up failed", "top up did not go through", "failed to add money to account"],
}


@pytest.fixture(scope="session")
def toy_frame():
    rows = []
    for intent, texts in TOPICS.items():
        for t in texts:
            for suffix in ("", " please", " thanks", " help"):
                rows.append({"text": t + suffix, "intent": intent})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def toy_bundle(toy_frame):
    pipe = make_pipeline("both", LinearSVC(C=1.0))
    pipe.fit(toy_frame["text"], toy_frame["intent"])
    idx = {c: i for i, c in enumerate(pipe.classes_)}
    temp = fit_temperature(scores(pipe, toy_frame["text"]), toy_frame["intent"].map(idx).to_numpy())
    return Bundle(pipeline=pipe, temperature=temp, threshold=0.5)
