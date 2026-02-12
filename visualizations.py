#!/usr/bin/env python3
"""
Create manuscript-style figures for LLM citation analysis.

Outputs:
- plots/figure_1_summary.png
- plots/figure_2_confusion_matrices.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


SCORING_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER"]
SCORING_LABELS_DISPLAY = ["Localized", "Metastatic", "Both", "Neither"]
VENDOR_ORDER = ["openai", "gemini", "anthropic"]
CONDITION_ORDER = ["baseline", "evidence_required"]

VENDOR_LABEL = {
    "openai": "GPT-5.2",
    "gemini": "Gemini 3 Flash",
    "anthropic": "Claude Opus 4.5",
}
CONDITION_LABEL = {"baseline": "Baseline", "evidence_required": "Evidence Required"}

# Palette aligned to the provided reference screenshot.
COLOR_BASELINE = "#CD4C64"
COLOR_EVIDENCE = "#93387E"
COLOR_CORRECT_QUOTE_PRESENT = "#2B6F92"
COLOR_CORRECT_QUOTE_NOT_PRESENT = "#4FA79E"
COLOR_INCORRECT = "#B13D74"
COLOR_UNCLEAR_INVALID = "#8A8F99"
FIG_BG = "#FFFFFF"
CONF_TEXT_DARK = "#2D437C"
CONF_BORDER = "#4F525B"
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "confusion_blues",
    ["#FFFFFF", "#67A8CF", "#0E3A7D"],
)


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["parsed_label"] = out["parsed_label"].fillna("INVALID").astype(str).str.strip().str.upper()
    out["ground_truth"] = out["ground_truth"].fillna("").astype(str).str.strip().str.upper()

    if "format_valid" in out.columns:
        out["format_valid"] = to_bool(out["format_valid"])
    else:
        out["format_valid"] = out["parsed_label"].isin([*SCORING_LABELS, "UNCLEAR"])

    if "quote_valid" in out.columns:
        out["quote_valid"] = to_bool(out["quote_valid"])
    else:
        out["quote_valid"] = False

    out["invalid_output"] = out["parsed_label"].isin({"INVALID", "ERROR"}) | (~out["format_valid"])
    out["answered_output"] = out["parsed_label"].isin(SCORING_LABELS) & (~out["invalid_output"])
    out["correct_answer"] = out["answered_output"] & (out["parsed_label"] == out["ground_truth"])
    # Mechanical validity check: quote is an exact abstract substring.
    out["correct_and_quote_present"] = out["correct_answer"] & out["quote_valid"]
    return out


def prepare_judgments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_vendor"] = out["source_vendor"].fillna("").astype(str).str.strip().str.lower()
    out["source_condition"] = out["source_condition"].fillna("").astype(str).str.strip()
    out["source_label"] = out["source_label"].fillna("").astype(str).str.strip().str.upper()
    out["ground_truth"] = out["ground_truth"].fillna("").astype(str).str.strip().str.upper()
    out["judge_vendor"] = out["judge_vendor"].fillna("").astype(str).str.strip().str.lower()

    if "format_valid" in out.columns:
        out["format_valid"] = to_bool(out["format_valid"])
    else:
        out["format_valid"] = out.get("verdict", pd.Series([""] * len(out))).fillna("").astype(str).str.strip().str.upper().isin(
            {"SUPPORTED", "NOT_SUPPORTED"}
        )

    if "supports_label" in out.columns:
        out["supports_label"] = to_bool(out["supports_label"])
    else:
        out["supports_label"] = out.get("verdict", pd.Series([""] * len(out))).fillna("").astype(str).str.strip().str.upper().eq(
            "SUPPORTED"
        )

    if "source_repeat_idx" in out.columns:
        out["source_repeat_idx"] = pd.to_numeric(out["source_repeat_idx"], errors="coerce").fillna(1).astype(int)
    else:
        out["source_repeat_idx"] = 1

    return out


def metric_value(metrics_df: pd.DataFrame, vendor: str, condition: str, column: str) -> float:
    row = metrics_df[(metrics_df["vendor"] == vendor) & (metrics_df["condition"] == condition)]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def build_evidence_breakdown(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evidence = pred_df[pred_df["condition"] == "evidence_required"].copy()

    for vendor in VENDOR_ORDER:
        subset = evidence[evidence["vendor"] == vendor]
        if subset.empty:
            continue

        n_total = len(subset)
        rows.append(
            {
                "vendor": vendor,
                "correct_and_quote_present": subset["correct_and_quote_present"].sum() / n_total * 100.0,
                "correct_not_quote_present": (subset["correct_answer"] & ~subset["quote_valid"]).sum() / n_total * 100.0,
                "incorrect_answered": (subset["answered_output"] & ~subset["correct_answer"]).sum() / n_total * 100.0,
                "unclear_or_invalid": (~subset["answered_output"]).sum() / n_total * 100.0,
            }
        )

    return pd.DataFrame(rows)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_title(ax, label: str, title: str):
    ax.set_title(f"{label}) {title}", loc="left", fontweight="bold", fontsize=13)


def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_figure_1(metrics_df: pd.DataFrame, pred_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(FIG_BG)
    for axis in axes.flat:
        axis.set_facecolor(FIG_BG)

    # A) Coverage
    ax = axes[0, 0]
    x = np.arange(len(VENDOR_ORDER))
    width = 0.36

    baseline_vals = [metric_value(metrics_df, v, "baseline", "coverage_pct") for v in VENDOR_ORDER]
    evidence_vals = [metric_value(metrics_df, v, "evidence_required", "coverage_pct") for v in VENDOR_ORDER]

    bars_baseline = ax.bar(
        x - width / 2,
        baseline_vals,
        width=width,
        label="Baseline",
        color=COLOR_BASELINE,
    )
    bars_evidence = ax.bar(
        x + width / 2,
        evidence_vals,
        width=width,
        label="Evidence Required",
        color=COLOR_EVIDENCE,
    )
    add_bar_labels(ax, bars_baseline)
    add_bar_labels(ax, bars_evidence)

    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([VENDOR_LABEL[v] for v in VENDOR_ORDER], fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Condition", frameon=True, fontsize=10, title_fontsize=10, loc="lower left")
    panel_title(ax, "A", "Coverage")
    style_axis(ax)

    # B) Correct & Mechanically-Valid Breakdown
    ax = axes[0, 1]
    breakdown = build_evidence_breakdown(pred_df)
    x = np.arange(len(breakdown))
    bottom = np.zeros(len(breakdown))

    stacks = [
        ("correct_and_quote_present", "Correct + Mechanically-Valid", COLOR_CORRECT_QUOTE_PRESENT),
        ("correct_not_quote_present", "Correct + Not Mechanically-Valid", COLOR_CORRECT_QUOTE_NOT_PRESENT),
        ("incorrect_answered", "Incorrect (Answered)", COLOR_INCORRECT),
        ("unclear_or_invalid", "Unclear/Invalid", COLOR_UNCLEAR_INVALID),
    ]

    for key, label, color in stacks:
        vals = breakdown[key].to_numpy() if key in breakdown.columns else np.zeros(len(breakdown))
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals

    ax.set_ylabel("Percentage of Predictions (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([VENDOR_LABEL[v] for v in breakdown["vendor"]], fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    panel_title(ax, "B", "Correct & Mechanically-Valid Breakdown")
    style_axis(ax)

    # C) Label Agreement (Fleiss' kappa)
    ax = axes[1, 0]
    kappa_matrix = np.array(
        [
            [
                metric_value(metrics_df, vendor, "baseline", "fleiss_kappa"),
                metric_value(metrics_df, vendor, "evidence_required", "fleiss_kappa"),
            ]
            for vendor in VENDOR_ORDER
        ]
    )
    sns.heatmap(
        kappa_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#D0D0D0",
        xticklabels=[CONDITION_LABEL[c] for c in CONDITION_ORDER],
        yticklabels=[VENDOR_LABEL[v] for v in VENDOR_ORDER],
        cbar_kws={"label": "Fleiss' κ"},
        annot_kws={"fontsize": 11},
        ax=ax,
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, labelrotation=90)
    panel_title(ax, "C", "Label Agreement (Fleiss' κ)")

    # D) Citation Similarity (Jaccard)
    ax = axes[1, 1]
    jaccard_matrix = np.array(
        [
            [metric_value(metrics_df, vendor, "evidence_required", "citation_jaccard")]
            for vendor in VENDOR_ORDER
        ]
    )
    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#D0D0D0",
        xticklabels=["Jaccard"],
        yticklabels=[VENDOR_LABEL[v] for v in VENDOR_ORDER],
        cbar_kws={"label": "Jaccard Similarity"},
        annot_kws={"fontsize": 11},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, labelrotation=90)
    panel_title(ax, "D", "Citation Similarity (Jaccard)")

    plt.tight_layout()
    out = output_dir / "figure_1_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def f1_from_subset(subset: pd.DataFrame, pred_col: str) -> float:
    if subset.empty:
        return np.nan
    return float(
        f1_score(
            subset["ground_truth"],
            subset[pred_col],
            average="macro",
            labels=SCORING_LABELS,
            zero_division=0,
        )
    )


def _rank_judges(scores: dict[str, float]) -> list[str]:
    return sorted(
        scores.keys(),
        key=lambda vendor: (-scores[vendor], VENDOR_ORDER.index(vendor) if vendor in VENDOR_ORDER else 999),
    )


def pick_middle_judge(
    summary_df: pd.DataFrame,
    judgments_df: pd.DataFrame,
    source_vendor: str,
    source_condition: str = "evidence_required",
) -> tuple[str, str]:
    summary = summary_df.copy()
    source_vendor_norm = str(source_vendor).strip().lower()
    source_condition_norm = str(source_condition).strip()
    if not summary.empty and "judge_vendor" in summary.columns:
        if "source_vendor" in summary.columns:
            summary["source_vendor"] = summary["source_vendor"].fillna("").astype(str).str.strip().str.lower()
            summary = summary[summary["source_vendor"] == source_vendor_norm]
        if "source_condition" in summary.columns:
            summary["source_condition"] = summary["source_condition"].fillna("").astype(str).str.strip()
            summary = summary[summary["source_condition"] == source_condition_norm]
        summary["judge_vendor"] = summary["judge_vendor"].fillna("").astype(str).str.strip().str.lower()
        if "macro_f1_judge_valid_only" in summary.columns:
            summary["macro_f1_judge_valid_only"] = pd.to_numeric(
                summary["macro_f1_judge_valid_only"], errors="coerce"
            )
            if "n_scored_f1" in summary.columns:
                summary["n_scored_f1"] = pd.to_numeric(summary["n_scored_f1"], errors="coerce").fillna(0)
                weighted = summary[
                    summary["macro_f1_judge_valid_only"].notna() & summary["n_scored_f1"].gt(0)
                ].copy()
                if not weighted.empty:
                    scores = (
                        weighted.groupby("judge_vendor", sort=False)
                        .apply(
                            lambda x: float(
                                np.average(
                                    x["macro_f1_judge_valid_only"].to_numpy(),
                                    weights=x["n_scored_f1"].to_numpy(),
                                )
                            )
                        )
                        .to_dict()
                    )
                    if scores:
                        ranked = _rank_judges(scores)
                        middle = ranked[len(ranked) // 2]
                        return middle, "middle rank by weighted macro F1 (semantically grounded only)"

            scores_series = (
                summary.groupby("judge_vendor", sort=False)["macro_f1_judge_valid_only"].mean().dropna()
            )
            if not scores_series.empty:
                scores = scores_series.to_dict()
                ranked = _rank_judges(scores)
                middle = ranked[len(ranked) // 2]
                return middle, "middle rank by mean macro F1 (semantically grounded only)"

        grounding_rate_col = None
        if "semantic_grounding_rate_valid_pct" in summary.columns:
            grounding_rate_col = "semantic_grounding_rate_valid_pct"
        elif "support_rate_valid_pct" in summary.columns:
            grounding_rate_col = "support_rate_valid_pct"

        if grounding_rate_col is not None:
            summary[grounding_rate_col] = pd.to_numeric(summary[grounding_rate_col], errors="coerce")
            scores_series = (
                summary.groupby("judge_vendor", sort=False)[grounding_rate_col].mean().dropna()
            )
            if not scores_series.empty:
                scores = scores_series.to_dict()
                ranked = _rank_judges(scores)
                middle = ranked[len(ranked) // 2]
                return middle, "middle rank by mean semantic grounding rate among format-valid judgments"

    valid = judgments_df.copy()
    if "source_vendor" in valid.columns:
        valid = valid[valid["source_vendor"] == source_vendor_norm]
    if "source_condition" in valid.columns:
        valid = valid[valid["source_condition"] == source_condition_norm]
    valid = valid[valid["format_valid"]].copy()
    if valid.empty:
        raise RuntimeError(
            f"No format-valid judgments available to select a middle judge model for source vendor '{source_vendor_norm}'."
        )
    support_scores = (
        valid.groupby("judge_vendor", sort=False)["supports_label"].mean().mul(100.0).to_dict()
    )
    ranked = _rank_judges(support_scores)
    middle = ranked[len(ranked) // 2]
    return middle, "middle rank by semantic grounding rate among format-valid judgments"


def plot_figure_2_confusions(
    pred_df: pd.DataFrame,
    judgments_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
):
    variants = ["original", "mechanically_valid", "grounded"]
    fig, axes = plt.subplots(len(VENDOR_ORDER), len(variants), figsize=(18.5, 15.0))
    fig.patch.set_facecolor(FIG_BG)

    matrices: dict[tuple[str, str], np.ndarray] = {}
    stats: dict[tuple[str, str], tuple[float, int]] = {}
    grounded_judge_label: dict[str, str] = {}
    global_max = 1

    for vendor in VENDOR_ORDER:
        row_middle_judge_vendor, row_middle_judge_metric = pick_middle_judge(
            summary_df=summary_df,
            judgments_df=judgments_df,
            source_vendor=vendor,
            source_condition="evidence_required",
        )
        row_middle_judge_label = VENDOR_LABEL.get(row_middle_judge_vendor, row_middle_judge_vendor)
        grounded_judge_label[vendor] = row_middle_judge_label
        print(
            f"{VENDOR_LABEL.get(vendor, vendor)}: selected middle-performing judge model: "
            f"{row_middle_judge_label} ({row_middle_judge_metric})"
        )

        original_subset = pred_df[
            (pred_df["vendor"] == vendor)
            & (pred_df["condition"] == "evidence_required")
            & pred_df["answered_output"]
            & pred_df["ground_truth"].isin(SCORING_LABELS)
        ].copy()
        cm_original = confusion_matrix(
            original_subset["ground_truth"],
            original_subset["parsed_label"],
            labels=SCORING_LABELS,
        )
        matrices[(vendor, "original")] = cm_original
        stats[(vendor, "original")] = (
            f1_from_subset(original_subset, pred_col="parsed_label"),
            len(original_subset),
        )
        global_max = max(global_max, int(cm_original.max()))

        mechanical_subset = pred_df[
            (pred_df["vendor"] == vendor)
            & (pred_df["condition"] == "evidence_required")
            & pred_df["answered_output"]
            & pred_df["quote_valid"]
            & pred_df["ground_truth"].isin(SCORING_LABELS)
        ].copy()
        cm_mechanical = confusion_matrix(
            mechanical_subset["ground_truth"],
            mechanical_subset["parsed_label"],
            labels=SCORING_LABELS,
        )
        matrices[(vendor, "mechanically_valid")] = cm_mechanical
        stats[(vendor, "mechanically_valid")] = (
            f1_from_subset(mechanical_subset, pred_col="parsed_label"),
            len(mechanical_subset),
        )
        global_max = max(global_max, int(cm_mechanical.max()))

        grounded_subset = judgments_df[
            (judgments_df["source_vendor"] == vendor)
            & (judgments_df["source_condition"] == "evidence_required")
            & (judgments_df["judge_vendor"] == row_middle_judge_vendor)
            & judgments_df["format_valid"]
            & judgments_df["supports_label"]
            & judgments_df["source_label"].isin(SCORING_LABELS)
            & judgments_df["ground_truth"].isin(SCORING_LABELS)
        ].copy()
        dedupe_cols = [
            "trial_idx",
            "source_vendor",
            "source_model",
            "source_condition",
            "source_repeat_idx",
            "source_label",
            "source_quote_sha1",
            "judge_vendor",
        ]
        dedupe_cols = [col for col in dedupe_cols if col in grounded_subset.columns]
        if dedupe_cols:
            grounded_subset = grounded_subset.drop_duplicates(subset=dedupe_cols, keep="last")

        cm_grounded = confusion_matrix(
            grounded_subset["ground_truth"],
            grounded_subset["source_label"],
            labels=SCORING_LABELS,
        )
        matrices[(vendor, "grounded")] = cm_grounded
        stats[(vendor, "grounded")] = (
            f1_from_subset(grounded_subset, pred_col="source_label"),
            len(grounded_subset),
        )
        global_max = max(global_max, int(cm_grounded.max()))

    for row_i, vendor in enumerate(VENDOR_ORDER):
        for col_i, variant in enumerate(variants):
            ax = axes[row_i, col_i]
            cm = matrices[(vendor, variant)]
            f1_val, n_rows = stats[(vendor, variant)]
            f1_text = f"{f1_val:.2f}" if not np.isnan(f1_val) else "N/A"

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap=CONFUSION_CMAP,
                cbar=False,
                linewidths=0,
                vmin=0,
                vmax=global_max,
                square=True,
                xticklabels=SCORING_LABELS_DISPLAY,
                yticklabels=SCORING_LABELS_DISPLAY,
                annot_kws={"fontsize": 12},
                ax=ax,
            )
            ax.set_facecolor(FIG_BG)

            threshold = global_max * 0.58
            for text in ax.texts:
                try:
                    val = int(text.get_text())
                except ValueError:
                    val = 0
                text.set_color("white" if val >= threshold else CONF_TEXT_DARK)

            if variant == "original":
                title_prefix = "Evidence Required (Original)"
            elif variant == "mechanically_valid":
                title_prefix = "Mechanically-Valid Quotes Only"
            else:
                title_prefix = (
                    f"Semantically Grounded Only ({grounded_judge_label.get(vendor, 'Unknown')} Judge)"
                )

            if row_i == 0:
                ax.set_title(
                    f"{title_prefix}\n(F1={f1_text}, n={n_rows})",
                    fontsize=14,
                    fontweight="bold",
                )
            else:
                if variant == "grounded":
                    ax.set_title(
                        f"({grounded_judge_label.get(vendor, 'Unknown')} Judge, F1={f1_text}, n={n_rows})",
                        fontsize=14,
                        fontweight="bold",
                    )
                else:
                    ax.set_title(f"(F1={f1_text}, n={n_rows})", fontsize=14, fontweight="bold")

            if col_i == 0:
                ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
                ax.text(
                    -0.40,
                    0.5,
                    VENDOR_LABEL[vendor],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=15,
                    fontweight="bold",
                )
            else:
                ax.set_ylabel("")

            if row_i == len(VENDOR_ORDER) - 1:
                ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="x", labelrotation=45, labelsize=9)
            ax.tick_params(axis="y", labelrotation=0, labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color(CONF_BORDER)

    plt.subplots_adjust(left=0.11, right=0.985, top=0.95, bottom=0.08, hspace=0.38, wspace=0.20)
    out = output_dir / "figure_2_confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate plots for LLM citation analysis")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    parser.add_argument("--metrics", type=str, default=None, help="Path to metrics.csv")
    parser.add_argument("--judgments", type=str, default=None, help="Path to evidence_judgments.csv")
    parser.add_argument(
        "--evidence-summary",
        type=str,
        default=None,
        help="Path to evidence_judgment_summary.csv",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Plot output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    predictions_path = Path(args.predictions) if args.predictions else script_dir / "results" / "predictions.csv"
    metrics_path = Path(args.metrics) if args.metrics else script_dir / "results" / "metrics.csv"
    judgments_path = Path(args.judgments) if args.judgments else script_dir / "results" / "evidence_judgments.csv"
    summary_path = (
        Path(args.evidence_summary)
        if args.evidence_summary
        else script_dir / "results" / "evidence_judgment_summary.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        print(f"Missing predictions: {predictions_path}")
        return 1
    if not metrics_path.exists():
        print(f"Missing metrics: {metrics_path}")
        print("Run analyze_results.py first.")
        return 1

    metrics_df = pd.read_csv(metrics_path)
    pred_df = prepare_predictions(pd.read_csv(predictions_path))

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    sns.set_style(
        "whitegrid",
        {
            "axes.facecolor": FIG_BG,
            "figure.facecolor": FIG_BG,
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.5,
        },
    )

    plot_figure_1(metrics_df, pred_df, output_dir)
    if not judgments_path.exists():
        print(f"Skipping figure 2 confusion plot; missing judgments: {judgments_path}")
        return 0

    judgments_df = prepare_judgments(pd.read_csv(judgments_path))
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        print(
            f"Evidence summary not found ({summary_path}); selecting middle judge from judgments fallback."
        )
        summary_df = pd.DataFrame()

    plot_figure_2_confusions(pred_df, judgments_df, summary_df, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
