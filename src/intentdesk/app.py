"""Gradio front end. All scoring goes through intentdesk.router.route."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ARTIFACT_DIR
from .persist import load_bundle
from .router import Bundle, Decision, route

EXAMPLES = [
    "My card still has not arrived after two weeks",
    "why was i charged a fee when i took money out abroad",
    "the top up i made yesterday is still pending",
    "how do i change my pin",
    "wat is the exchnage rate you use",
]


def format_decision(d: Decision) -> tuple[dict[str, float], str]:
    """Turn a Decision into (label->probability dict, markdown summary)."""
    labels = {name.replace("_", " "): p for name, p in d.top3}
    verdict = "Auto-route" if d.routed else "Send to human queue"
    md = f"**{verdict}** to `{d.intent}` ({d.confidence:.0%} confident)\n\n{d.reason}"
    return labels, md


def score_query(bundle: Bundle, text: str) -> tuple[dict[str, float], str]:
    try:
        return format_decision(route(bundle, text))
    except ValueError:
        return {}, "Type a customer message to see where it would go."


def build_app(bundle: Bundle):
    import gradio as gr

    with gr.Blocks(title="intentdesk") as demo:
        gr.Markdown(
            "# intentdesk\nPaste a customer message. The router picks one of 77 banking "
            f"intents and holds anything below {bundle.threshold:.0%} confidence for a person."
        )
        box = gr.Textbox(label="Customer message", lines=2)
        go = gr.Button("Route it", variant="primary")
        top = gr.Label(label="Top three intents", num_top_classes=3)
        out = gr.Markdown()
        fn = lambda t: score_query(bundle, t)  # noqa: E731
        go.click(fn, box, [top, out])
        box.submit(fn, box, [top, out])
        gr.Examples(EXAMPLES, box)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ARTIFACT_DIR / "intent_router.skops")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_app(load_bundle(a.model)).launch(server_port=a.port)


if __name__ == "__main__":
    main()
