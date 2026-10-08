"""Save and load the router with skops, which refuses to unpickle arbitrary code."""

from __future__ import annotations

from pathlib import Path

import skops.io as sio

from .router import Bundle


def save_bundle(bundle: Bundle, path: Path) -> None:
    sio.dump(
        {"pipeline": bundle.pipeline, "temperature": bundle.temperature, "threshold": bundle.threshold},
        path,
    )


def load_bundle(path: Path) -> Bundle:
    untrusted = sio.get_untrusted_types(file=path)
    # Only types from this package and numpy/scipy/sklearn internals should appear.
    allowed = [t for t in untrusted if t.split(".")[0] in {"numpy", "scipy", "sklearn", "intentdesk"}]
    if len(allowed) != len(untrusted):
        extra = sorted(set(untrusted) - set(allowed))
        raise ValueError(f"unexpected types in model file: {extra}")
    obj = sio.load(path, trusted=allowed)
    return Bundle(obj["pipeline"], float(obj["temperature"]), float(obj["threshold"]))
