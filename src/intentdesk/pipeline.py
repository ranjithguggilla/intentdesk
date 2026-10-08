"""End-to-end run: compare, select, calibrate, audit, stress, price, persist."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from . import plots
from .audit import confused_pairs, flag_issues, issue_table, out_of_fold_probs
from .calibration import (
    best_threshold,
    coverage_curve,
    expected_calibration_error,
    fit_temperature,
    policy_cost,
    probabilities,
)
from .config import ARTIFACT_DIR, ASSET_DIR, COSTS, DATA_DIR, SEED
from .data import exact_overlap, load_intents, load_splits
from .models import candidates, compare, make_pipeline, scores
from .persist import save_bundle
from .robustness import perturb_series
from .router import Bundle

TOLERANCE = 0.003  # a simpler model within this much accuracy of the best one wins


def pick_winner(table: pd.DataFrame) -> str:
    best = table["val_accuracy"].max()
    close = table[table["val_accuracy"] >= best - TOLERANCE]
    return str(close.sort_values("fit_seconds").iloc[0]["model"])


def fit_named(name: str, frame: pd.DataFrame, seed: int):
    kind, est = candidates(seed)[name]
    pipe = make_pipeline(kind, clone(est))
    pipe.fit(frame["text"], frame["intent"])
    return pipe


def topk_accuracy(sc: np.ndarray, classes, y, k: int = 3) -> float:
    top = np.argsort(sc, axis=1)[:, -k:]
    cls = np.asarray(classes)
    return float(np.mean([y[i] in cls[top[i]] for i in range(len(y))]))


def run(
    out_dir: Path = ARTIFACT_DIR,
    asset_dir: Path = ASSET_DIR,
    data_dir: Path = DATA_DIR,
    seed: int = SEED,
    quick: bool = False,
) -> dict:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(data_dir, seed)
    intents = load_intents(data_dir)
    m: dict = {
        "data": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "intents": len(intents),
            "train_per_intent_min": int(train.intent.value_counts().min()),
            "train_per_intent_max": int(train.intent.value_counts().max()),
            "test_exact_overlap_with_train": exact_overlap(pd.concat([train, val]), test),
        }
    }

    # 1. model comparison on validation
    table = compare(train, val, seed)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    plots.model_comparison(table, asset_dir / "model_comparison.png")
    winner = pick_winner(table)
    m["comparison"] = table.round(4).to_dict(orient="records")
    m["winner"] = winner
    m["best_by_validation"] = str(table.iloc[0]["model"])

    # 2. final model, temperature on validation, test report
    pipe = fit_named(winner, train, seed)
    classes = pipe.classes_
    idx = {c: i for i, c in enumerate(classes)}
    sc_val = scores(pipe, val["text"])
    yv = val["intent"].map(idx).to_numpy()
    temp = fit_temperature(sc_val, yv)
    sc_test = scores(pipe, test["text"])
    yt = test["intent"].map(idx).to_numpy()
    pred = sc_test.argmax(1)
    p_test = probabilities(sc_test, temp)
    conf = p_test.max(1)
    correct = pred == yt
    m["test"] = {
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "top3_accuracy": topk_accuracy(sc_test, classes, test["intent"].to_numpy()),
        "temperature": temp,
        "ece_raw_softmax": expected_calibration_error(probabilities(sc_test, 1.0).max(1), correct),
        "ece_temperature_scaled": expected_calibration_error(conf, correct),
    }
    per = pd.DataFrame({"intent": test["intent"], "ok": correct}).groupby("intent").ok.mean()
    m["test"]["worst_intents"] = {k: round(float(v), 3) for k, v in per.sort_values().head(5).items()}
    m["test"]["intents_at_100pct"] = int((per == 1.0).sum())
    m["confused_pairs"] = confused_pairs(test["intent"].to_numpy(), classes[pred]).to_dict(orient="records")

    # bootstrap interval on accuracy
    rng = np.random.default_rng(seed)
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]
    m["test"]["accuracy_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # 3. abstention and cost
    val_conf = probabilities(sc_val, temp).max(1)
    val_correct = sc_val.argmax(1) == yv
    thr = best_threshold(val_conf, val_correct, COSTS)
    grid, cov, acc = coverage_curve(conf, correct)
    auto = conf >= thr
    gated = policy_cost(conf, correct, thr, COSTS)
    all_human = COSTS.human_triage + (1 - COSTS.human_accuracy) * COSTS.misroute
    all_auto = COSTS.auto_route + (1 - correct.mean()) * COSTS.misroute
    boot_saving = []
    for _ in range(1000):
        b = rng.integers(0, len(correct), len(correct))
        boot_saving.append((all_human - policy_cost(conf[b], correct[b], thr, COSTS)) * COSTS.queries_per_year)
    m["routing"] = {
        "threshold": thr,
        "auto_routed_share": float(auto.mean()),
        "auto_routed_accuracy": float(correct[auto].mean()),
        "human_queue_share": float(1 - auto.mean()),
        "accuracy_of_human_queue_if_machine_guessed": float(correct[~auto].mean()) if (~auto).any() else None,
        "cost_per_query_gated": gated,
        "cost_per_query_all_human": all_human,
        "cost_per_query_all_auto": all_auto,
        "annual_saving_vs_all_human": (all_human - gated) * COSTS.queries_per_year,
        "annual_saving_ci95": [float(np.percentile(boot_saving, 2.5)), float(np.percentile(boot_saving, 97.5))],
        "annual_saving_vs_all_auto": (all_auto - gated) * COSTS.queries_per_year,
        "assumptions": COSTS.__dict__,
    }
    plots.coverage_plot(grid, cov, acc, thr, asset_dir / "coverage_accuracy.png")
    plots.calibration_plot(conf, correct, asset_dir / "reliability.png")
    ths = np.round(np.linspace(0, 0.99, 100), 2)
    plots.cost_plot(ths, [policy_cost(conf, correct, t, COSTS) for t in ths], all_human, asset_dir / "cost_curve.png")

    # 4. label audit
    lr_name = "logistic regression (word+char)"
    kind, est = candidates(seed)[lr_name]
    full = pd.concat([train, val], ignore_index=True)
    lr_pipe = make_pipeline(kind, est)
    if quick:
        lr_pipe = make_pipeline("both", clone(est).set_params(max_iter=60))
    probs = out_of_fold_probs(lr_pipe, full["text"], full["intent"], seed, folds=3 if quick else 5)
    lr_classes = np.array(sorted(full.intent.unique()))
    yf = full["intent"].map({c: i for i, c in enumerate(lr_classes)}).to_numpy()
    issues = flag_issues(probs, yf)
    tbl = issue_table(full, probs, issues, lr_classes)
    tbl.head(200).to_csv(out_dir / "suspected_label_errors.csv", index=False)
    n_train = len(train)
    train_flagged = issues[issues < n_train]
    clean_train = train.drop(index=train_flagged).reset_index(drop=True)
    pipe_clean = fit_named(winner, clean_train, seed)
    pc_val = pipe_clean.predict(val["text"])
    pc_test = pipe_clean.predict(test["text"])
    m["audit"] = {
        "flagged_in_train_plus_val": int(len(issues)),
        "flagged_share": float(len(issues) / len(full)),
        "flagged_in_train_part": int(len(train_flagged)),
        "val_accuracy_original": float(accuracy_score(val["intent"], pipe.predict(val["text"]))),
        "val_accuracy_after_dropping_flagged": float(accuracy_score(val["intent"], pc_val)),
        "test_accuracy_original": m["test"]["accuracy"],
        "test_accuracy_after_dropping_flagged": float(accuracy_score(test["intent"], pc_test)),
        "intent_pairs_most_flagged": tbl.assign(
            pair=[" -> ".join(sorted([a, b])) for a, b in zip(tbl.given, tbl.suggested, strict=True)]
        )
        .pair.value_counts()
        .head(5)
        .to_dict(),
    }
    # flagged test rows: how much of the test error is arguably label error
    lr_fit = make_pipeline(kind, clone(est)) if not quick else lr_pipe
    lr_fit.fit(train["text"], train["intent"])
    pt = lr_fit.predict_proba(test["text"])
    yt_lr = test["intent"].map({c: i for i, c in enumerate(lr_fit.classes_)}).to_numpy()
    t_issues = flag_issues(pt, yt_lr)
    wrong = np.where(pt.argmax(1) != yt_lr)[0]
    m["audit"]["test_flagged"] = int(len(t_issues))
    m["audit"]["test_errors_lr"] = int(len(wrong))
    m["audit"]["test_errors_also_flagged"] = int(len(set(wrong) & set(t_issues)))
    keep = np.setdiff1d(np.arange(len(test)), t_issues)
    m["audit"]["test_accuracy_excluding_flagged_rows"] = float(correct[keep].mean())

    # 5. typo stress
    rates = [0.0, 0.03, 0.06, 0.10, 0.15]
    stress: dict[str, list[float]] = {}
    stress_models = [
        ("word only", "linear SVM (word only)"),
        ("char only", "linear SVM (char only)"),
        ("word + char", "linear SVM (word+char)"),
    ]
    for label, name in stress_models:
        p = fit_named(name, train, seed)
        stress[label] = [
            float(accuracy_score(test["intent"], p.predict(perturb_series(test["text"], r, seed)))) for r in rates
        ]
    m["typo_stress"] = {"rates": rates, **stress}
    plots.robustness_plot(rates, stress, asset_dir / "typo_robustness.png")

    # 6. learning curve (examples per intent)
    full_per_intent = int(round(len(train) / train.intent.nunique()))
    floor = int(train.intent.value_counts().min())
    sizes = [s for s in (5, 10, 20, 40) if s <= floor] + [full_per_intent]
    curves = {"linear SVM (word+char)": [], "linear SVM (word only)": [], "complement NB (word)": []}
    n_draws = 1 if quick else 3
    for s in sizes:
        accs = {k: [] for k in curves}
        for d in range(n_draws):
            if s == full_per_intent:
                sub = train
            else:
                sub = train.groupby("intent", group_keys=False).sample(n=s, random_state=seed + d)
            for k in curves:
                accs[k].append(accuracy_score(test["intent"], fit_named(k, sub, seed).predict(test["text"])))
        for k in curves:
            curves[k].append(float(np.mean(accs[k])))
    m["learning_curve"] = {"examples_per_intent": sizes, **curves}
    plots.learning_curve_plot(sizes, curves, asset_dir / "learning_curve.png")

    # 7. persist
    bundle = Bundle(pipeline=pipe, temperature=temp, threshold=thr)
    save_bundle(bundle, out_dir / "intent_router.skops")
    m["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full intentdesk pipeline")
    ap.add_argument("--quick", action="store_true", help="smaller audit and curve, for smoke tests")
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    a = ap.parse_args()
    m = run(out_dir=a.out, quick=a.quick)
    print(json.dumps({k: m[k] for k in ("winner", "test", "routing")}, indent=2))


if __name__ == "__main__":
    main()
