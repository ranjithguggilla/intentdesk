"""README charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, ACCENT, GREY, WARM = "#1f2933", "#1b6ca8", "#9aa5b1", "#c8553d"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def model_comparison(table, path: Path):
    t = table.sort_values("val_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = [ACCENT if i == len(t) - 1 else GREY for i in range(len(t))]
    ax.barh(t["model"], t["val_accuracy"], color=colors)
    for y, v in enumerate(t["val_accuracy"]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    _style(ax, "Validation accuracy by model", "accuracy", "")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def coverage_plot(grid, cov, acc, threshold, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(cov, acc, color=ACCENT, lw=2)
    m = np.argmin(np.abs(grid - threshold))
    ax.scatter([cov[m]], [acc[m]], color=WARM, zorder=3)
    ax.annotate(
        f"cost-optimal cut {threshold:.2f}\n{cov[m]:.0%} auto-routed, {acc[m]:.1%} correct",
        (cov[m], acc[m]),
        xytext=(-190, -55),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": GREY},
    )
    _style(ax, "Test accuracy on the auto-routed share", "share of queries auto-routed", "accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def calibration_plot(conf, correct, path: Path, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--")
    ax.plot(xs, ys, marker="o", color=ACCENT)
    _style(ax, "Reliability on the test fold", "predicted confidence", "observed accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def robustness_plot(rates, series: dict[str, list[float]], path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = [ACCENT, WARM, GREY, "#3e7c59"]
    for (name, ys), col in zip(series.items(), palette, strict=False):
        ax.plot([r * 100 for r in rates], ys, marker="o", label=name, color=col)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Accuracy as typos increase", "letters corrupted (%)", "test accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def learning_curve_plot(sizes, series: dict[str, list[float]], path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = [ACCENT, WARM, GREY]
    for (name, ys), col in zip(series.items(), palette, strict=False):
        ax.plot(sizes, ys, marker="o", label=name, color=col)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Accuracy against labelled examples per intent", "examples per intent (log scale)", "test accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_plot(thresholds, costs, all_human, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thresholds, costs, color=ACCENT, lw=2, label="confidence-gated routing")
    ax.axhline(all_human, color=GREY, ls="--", label="every query to a person")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "Cost per query by confidence cut (test fold)", "confidence threshold", "dollars per query")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
