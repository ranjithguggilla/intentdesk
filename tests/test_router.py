import pytest

from intentdesk.app import format_decision, score_query
from intentdesk.explain import reason, top_terms
from intentdesk.router import route


def test_route_returns_expected_intent(toy_bundle):
    d = route(toy_bundle, "my pin is blocked again")
    assert d.intent == "pin_blocked"
    assert len(d.top3) == 3
    assert d.top3[0][1] >= d.top3[1][1] >= d.top3[2][1]
    assert abs(sum(p for _, p in d.top3) - 1) <= 1


def test_route_rejects_empty(toy_bundle):
    with pytest.raises(ValueError):
        route(toy_bundle, "   ")


def test_threshold_controls_queue(toy_bundle):
    from dataclasses import replace

    strict = replace(toy_bundle, threshold=1.01)
    assert route(strict, "my top up failed").routed is False
    loose = replace(toy_bundle, threshold=0.0)
    assert route(loose, "my top up failed").routed is True


def test_terms_come_from_the_query(toy_bundle):
    d = route(toy_bundle, "card has not arrived")
    terms = top_terms(toy_bundle.pipeline, "card has not arrived", list(toy_bundle.pipeline.classes_).index(d.intent))
    assert terms and all(any(w in "card has not arrived" for w in t.split()) for t in terms)


def test_reason_mentions_queue_state():
    assert "straight to the queue" in reason("card_arrival", ["card"], 0.9, True)
    assert "held for a person" in reason("card_arrival", [], 0.4, False)


def test_score_query_handles_blank_and_text(toy_bundle):
    labels, md = score_query(toy_bundle, "")
    assert labels == {} and "Type a customer message" in md
    labels, md = score_query(toy_bundle, "what exchange rate do you use")
    assert "exchange rate" in md and len(labels) == 3


def test_format_decision_labels_are_readable(toy_bundle):
    labels, _ = format_decision(route(toy_bundle, "top up did not go through"))
    assert all("_" not in k for k in labels)


def test_gradio_app_builds(toy_bundle):
    gr = pytest.importorskip("gradio")
    from intentdesk.app import build_app

    assert isinstance(build_app(toy_bundle), gr.Blocks)
