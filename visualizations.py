#!/usr/bin/env python3
"""
Visualization helpers for llm_citations.

Reads:
- results/metrics.csv
- results/predictions.csv
- results/table_fleiss_kappa.csv (optional)
- results/table_switching.csv (optional)

Writes:
- plots/f1_by_condition.png
- plots/f1_comparison_bar.png
- plots/quote_validity_by_vendor.png
- plots/coverage_comparison.png
- plots/correct_grounded_breakdown.png
- plots/accuracy_by_ground_truth.png
- plots/fleiss_kappa_heatmap.png
- plots/confusion_matrices_combined.png
- plots/combined_summary.png
"""

from __future__ import annotations

import csv
from pathlib import Path


LABEL_ORDER = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER"]
VENDOR_ORDER = ["openai", "gemini", "anthropic"]
CONDITION_ORDER = ["baseline", "evidence_required"]


def _try_import_plotting():
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import seaborn as sns  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Plotting dependencies not installed. Install with `pip install -r requirements.txt`.\n"
            f"Original error: {e}"
        )
    return plt, np, sns


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def plot_f1_by_condition(metrics: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    vendors = [v for v in VENDOR_ORDER if v in {r.get("vendor", "") for r in metrics}]
    conditions = list(CONDITION_ORDER)

    values = {(r.get("vendor", ""), r.get("condition", "")): r for r in metrics}
    f1 = [[_to_float(values.get((v, c), {}).get("f1_macro_valid_only")) for c in conditions] for v in vendors]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(vendors))

    ax.bar(x - width / 2, [row[0] for row in f1], width, label="baseline")
    ax.bar(x + width / 2, [row[1] for row in f1], width, label="evidence_required")

    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 (valid only)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = plots_dir / "f1_by_condition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

def plot_f1_comparison_bar(metrics: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    by_key = {(r.get("vendor", ""), r.get("condition", "")): r for r in metrics}
    vendors = [v for v in VENDOR_ORDER if (v, "baseline") in by_key and (v, "evidence_required") in by_key]

    deltas = []
    for vendor in vendors:
        base = _to_float(by_key[(vendor, "baseline")].get("f1_macro_valid_only"))
        evid = _to_float(by_key[(vendor, "evidence_required")].get("f1_macro_valid_only"))
        deltas.append(evid - base)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(vendors))
    ax.bar(x, deltas, color="#4477AA")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylabel("Δ Macro F1 (evidence - baseline)")
    ax.grid(True, axis="y", alpha=0.3)

    out = plots_dir / "f1_comparison_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_quote_validity(metrics: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    rows = [r for r in metrics if r.get("condition") == "evidence_required"]
    if not rows:
        print("No evidence_required rows in metrics; skipping quote validity plot.")
        return

    by_vendor = {r.get("vendor", ""): r for r in rows}
    vendors = [v for v in VENDOR_ORDER if v in by_vendor]
    pct_present = [_to_float(by_vendor[v].get("pct_quote_present")) for v in vendors]
    pct_ok = [_to_float(by_vendor[v].get("pct_quote_is_substring")) for v in vendors]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(vendors))
    width = 0.35
    ax.bar(x - width / 2, pct_present, width, label="% quote present", color="#EE6677")
    ax.bar(x + width / 2, pct_ok, width, label="% quote valid", color="#228833")

    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylabel("% of rows (evidence_required)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    out = plots_dir / "quote_validity_by_vendor.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def _valid_prediction_row(row: dict[str, str]) -> bool:
    truth = (row.get("ground_truth") or "").strip().upper()
    pred = (row.get("parsed_label") or "").strip().upper()
    return truth in LABEL_ORDER and pred in LABEL_ORDER


def plot_coverage_comparison(metrics: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    by_key = {(r.get("vendor", ""), r.get("condition", "")): r for r in metrics}
    vendors = [v for v in VENDOR_ORDER if (v, "baseline") in by_key and (v, "evidence_required") in by_key]

    baseline_valid = [_to_float(by_key[(v, "baseline")].get("pct_valid")) for v in vendors]
    evidence_valid = [_to_float(by_key[(v, "evidence_required")].get("pct_valid")) for v in vendors]
    quote_ok = [_to_float(by_key[(v, "evidence_required")].get("pct_quote_is_substring")) for v in vendors]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(vendors))
    width = 0.25

    ax.bar(x - width, baseline_valid, width, label="baseline % valid", color="#4477AA")
    ax.bar(x, evidence_valid, width, label="evidence % valid", color="#EE6677")
    ax.bar(x + width, quote_ok, width, label="evidence % quote valid", color="#228833")

    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = plots_dir / "coverage_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_correct_grounded_breakdown(predictions: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    rows = [r for r in predictions if r.get("condition") == "evidence_required"]
    if not rows:
        print("No evidence_required rows in predictions; skipping correct/grounded breakdown plot.")
        return

    categories = [
        ("correct_grounded", "Correct + quote valid", "#228833"),
        ("correct_not_grounded", "Correct + quote invalid", "#66CCEE"),
        ("incorrect_grounded", "Incorrect + quote valid", "#EE6677"),
        ("incorrect_not_grounded", "Incorrect + quote invalid", "#AA3377"),
    ]

    vendors = [v for v in VENDOR_ORDER if v in {r.get("vendor", "") for r in rows}]
    counts = {v: {k: 0 for k, _, _ in categories} for v in vendors}

    for r in rows:
        vendor = r.get("vendor", "")
        if vendor not in counts:
            continue
        if not _valid_prediction_row(r):
            continue
        truth = (r.get("ground_truth") or "").strip().upper()
        pred = (r.get("parsed_label") or "").strip().upper()
        quote_ok = _as_bool(r.get("quote_is_substring"))

        correct = pred == truth
        if correct and quote_ok:
            counts[vendor]["correct_grounded"] += 1
        elif correct and not quote_ok:
            counts[vendor]["correct_not_grounded"] += 1
        elif not correct and quote_ok:
            counts[vendor]["incorrect_grounded"] += 1
        else:
            counts[vendor]["incorrect_not_grounded"] += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(vendors))
    bottom = np.zeros(len(vendors))

    for key, label, color in categories:
        vals = np.array([counts[v][key] for v in vendors], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylabel("Count (evidence_required)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = plots_dir / "correct_grounded_breakdown.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_accuracy_by_ground_truth(predictions: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, _ = _try_import_plotting()

    if not predictions:
        print("No predictions.csv found; skipping accuracy-by-ground-truth plot.")
        return

    vendors = [v for v in VENDOR_ORDER if v in {r.get("vendor", "") for r in predictions}]
    fig, axes = plt.subplots(len(vendors), 1, figsize=(10, 3.2 * max(len(vendors), 1)), sharex=True, sharey=True)
    if len(vendors) == 1:
        axes = [axes]

    for ax, vendor in zip(axes, vendors):
        for i, condition in enumerate(CONDITION_ORDER):
            rows = [r for r in predictions if r.get("vendor") == vendor and r.get("condition") == condition and _valid_prediction_row(r)]
            totals = {lab: 0 for lab in LABEL_ORDER}
            correct = {lab: 0 for lab in LABEL_ORDER}
            for r in rows:
                truth = (r.get("ground_truth") or "").strip().upper()
                pred = (r.get("parsed_label") or "").strip().upper()
                totals[truth] += 1
                if pred == truth:
                    correct[truth] += 1

            accs = [(correct[lab] / totals[lab]) if totals[lab] else 0.0 for lab in LABEL_ORDER]
            x = np.arange(len(LABEL_ORDER))
            width = 0.35
            offset = (-width / 2) if i == 0 else (width / 2)
            ax.bar(x + offset, accs, width, label=condition)

        ax.set_title(vendor)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[-1].set_xticks(np.arange(len(LABEL_ORDER)))
    axes[-1].set_xticklabels(LABEL_ORDER, rotation=0)

    out = plots_dir / "accuracy_by_ground_truth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_fleiss_kappa_heatmap(kappa_rows: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, sns = _try_import_plotting()
    if not kappa_rows:
        print("No table_fleiss_kappa.csv found; skipping kappa heatmap.")
        return

    vendors = [v for v in VENDOR_ORDER if v in {r.get("vendor", "") for r in kappa_rows}]
    conditions = [c for c in CONDITION_ORDER if c in {r.get("condition", "") for r in kappa_rows}]
    values = {(r.get("vendor", ""), r.get("condition", "")): _to_float(r.get("fleiss_kappa"), default=float("nan")) for r in kappa_rows}

    mat = np.full((len(vendors), len(conditions)), np.nan, dtype=float)
    for i, v in enumerate(vendors):
        for j, c in enumerate(conditions):
            mat[i, j] = values.get((v, c), np.nan)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=conditions, yticklabels=vendors, cmap="viridis", ax=ax)
    ax.set_title("Fleiss' kappa (agreement across runs)")

    out = plots_dir / "fleiss_kappa_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_confusion_matrices_combined(predictions: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, sns = _try_import_plotting()
    if not predictions:
        print("No predictions.csv found; skipping confusion matrices.")
        return

    vendors = [v for v in VENDOR_ORDER if v in {r.get("vendor", "") for r in predictions}]
    conditions = list(CONDITION_ORDER)

    fig, axes = plt.subplots(len(vendors), len(conditions), figsize=(10, 3.2 * max(len(vendors), 1)))
    if len(vendors) == 1:
        axes = [axes]

    label_to_idx = {lab: i for i, lab in enumerate(LABEL_ORDER)}

    for i, vendor in enumerate(vendors):
        for j, condition in enumerate(conditions):
            ax = axes[i][j] if len(vendors) > 1 else axes[0][j]
            rows = [r for r in predictions if r.get("vendor") == vendor and r.get("condition") == condition and _valid_prediction_row(r)]

            mat = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)
            for r in rows:
                truth = (r.get("ground_truth") or "").strip().upper()
                pred = (r.get("parsed_label") or "").strip().upper()
                mat[label_to_idx[truth], label_to_idx[pred]] += 1

            sns.heatmap(
                mat,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=LABEL_ORDER,
                yticklabels=LABEL_ORDER,
                cbar=False,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground truth")
            ax.set_title(f"{vendor} / {condition}")

    out = plots_dir / "confusion_matrices_combined.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_combined_summary(metrics: list[dict[str, str]], switching_rows: list[dict[str, str]], kappa_rows: list[dict[str, str]], plots_dir: Path) -> None:
    plt, np, sns = _try_import_plotting()

    by_key = {(r.get("vendor", ""), r.get("condition", "")): r for r in metrics}
    vendors = [v for v in VENDOR_ORDER if (v, "baseline") in by_key and (v, "evidence_required") in by_key]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ΔF1
    deltas = []
    for vendor in vendors:
        base = _to_float(by_key[(vendor, "baseline")].get("f1_macro_valid_only"))
        evid = _to_float(by_key[(vendor, "evidence_required")].get("f1_macro_valid_only"))
        deltas.append(evid - base)
    ax = axes[0][0]
    x = np.arange(len(vendors))
    ax.bar(x, deltas, color="#4477AA")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_title("Δ Macro F1")
    ax.grid(True, axis="y", alpha=0.3)

    # Quote validity
    ax = axes[0][1]
    quote_ok = [_to_float(by_key[(v, "evidence_required")].get("pct_quote_is_substring")) for v in vendors]
    ax.bar(x, quote_ok, color="#228833")
    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylim(0, 100)
    ax.set_title("% quote valid (evidence_required)")
    ax.grid(True, axis="y", alpha=0.3)

    # Switching
    ax = axes[1][0]
    switching_by_vendor = {r.get("vendor", ""): r for r in switching_rows}
    switched = [_to_float(switching_by_vendor.get(v, {}).get("pct_switched")) for v in vendors]
    ax.bar(x, switched, color="#EE6677")
    ax.set_xticks(x)
    ax.set_xticklabels(vendors)
    ax.set_ylim(0, 100)
    ax.set_title("% switched (baseline → evidence)")
    ax.grid(True, axis="y", alpha=0.3)

    # Kappa heatmap
    ax = axes[1][1]
    if kappa_rows:
        conditions = [c for c in CONDITION_ORDER if c in {r.get("condition", "") for r in kappa_rows}]
        values = {
            (r.get("vendor", ""), r.get("condition", "")): _to_float(r.get("fleiss_kappa"), default=float("nan"))
            for r in kappa_rows
        }
        mat = np.full((len(vendors), len(conditions)), np.nan, dtype=float)
        for i, v in enumerate(vendors):
            for j, c in enumerate(conditions):
                mat[i, j] = values.get((v, c), np.nan)
        sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=conditions, yticklabels=vendors, cmap="viridis", ax=ax)
        ax.set_title("Fleiss' kappa")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No kappa table found", ha="center", va="center")

    out = plots_dir / "combined_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> int:
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    metrics_path = results_dir / "metrics.csv"
    predictions_path = results_dir / "predictions.csv"
    kappa_path = results_dir / "table_fleiss_kappa.csv"
    switching_path = results_dir / "table_switching.csv"

    plots_dir = script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics = load_csv(metrics_path)
    predictions = load_csv(predictions_path)
    kappa_rows = load_csv(kappa_path)
    switching_rows = load_csv(switching_path)

    plot_f1_by_condition(metrics, plots_dir)
    plot_f1_comparison_bar(metrics, plots_dir)
    plot_quote_validity(metrics, plots_dir)
    plot_coverage_comparison(metrics, plots_dir)
    plot_correct_grounded_breakdown(predictions, plots_dir)
    plot_accuracy_by_ground_truth(predictions, plots_dir)
    plot_fleiss_kappa_heatmap(kappa_rows, plots_dir)
    plot_confusion_matrices_combined(predictions, plots_dir)
    plot_combined_summary(metrics, switching_rows, kappa_rows, plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
