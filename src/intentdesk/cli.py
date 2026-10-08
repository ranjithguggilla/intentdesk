"""Command line: route one message or a CSV of messages."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .config import ARTIFACT_DIR
from .persist import load_bundle
from .router import route


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="intentdesk-route")
    ap.add_argument("text", nargs="?", help="one customer message")
    ap.add_argument("--csv", type=Path, help="CSV with a 'text' column; writes routed rows to stdout")
    ap.add_argument("--model", type=Path, default=ARTIFACT_DIR / "intent_router.skops")
    a = ap.parse_args(argv)
    if not a.model.exists():
        print(f"model file not found: {a.model}. Run `python -m intentdesk.pipeline` first.", file=sys.stderr)
        return 2
    bundle = load_bundle(a.model)
    if a.csv:
        with a.csv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        w = csv.writer(sys.stdout)
        w.writerow(["text", "intent", "confidence", "queue"])
        for r in rows:
            d = route(bundle, r["text"])
            w.writerow([d.text, d.intent, f"{d.confidence:.3f}", "auto" if d.routed else "human"])
        return 0
    if not a.text:
        ap.error("give a message or --csv")
    d = route(bundle, a.text)
    print(f"{d.intent}  ({d.confidence:.0%})  {'auto-route' if d.routed else 'human queue'}")
    print(d.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
