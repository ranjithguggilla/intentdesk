"""Typo and noise perturbations used to stress the models."""

from __future__ import annotations

import random

_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEIGHBOURS: dict[str, str] = {}
for r, row in enumerate(_ROWS):
    for c, ch in enumerate(row):
        near = ""
        if c > 0:
            near += row[c - 1]
        if c < len(row) - 1:
            near += row[c + 1]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(_ROWS):
                near += _ROWS[rr][min(c, len(_ROWS[rr]) - 1)]
        _NEIGHBOURS[ch] = near


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt a fraction of letters: neighbour-key swap, drop, double or transpose."""
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            op = rng.choice(("swap", "drop", "double", "transpose"))
            if op == "swap" and ch.lower() in _NEIGHBOURS:
                out.append(rng.choice(_NEIGHBOURS[ch.lower()]))
            elif op == "drop":
                pass
            elif op == "double":
                out.extend([ch, ch])
            elif op == "transpose" and i + 1 < len(chars):
                out.extend([chars[i + 1], ch])
                i += 1
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def perturb_series(texts, rate: float, seed: int):
    rng = random.Random(seed)
    return [add_typos(t, rate, rng) for t in texts]
