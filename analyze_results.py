#!/usr/bin/env python3
"""
Analyze LLM citation evaluation outputs.

Primary outputs:
- results/metrics.csv
- results/table_manuscript.csv
- results/table_manuscript_raw.csv
- results/table_evidence_breakdown.csv
- results/table_confusion_long.csv
- results/table_mcnemar.csv
- results/table_fleiss_drivers.csv
- results/table_fleiss_drivers_raw.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ALLOWED_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER", "UNCLEAR"]
SCORING_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER"]
KAPPA_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER", "UNCLEAR", "INVALID"]
VENDOR_ORDER = ["openai", "gemini", "anthropic"]
CONDITION_ORDER = ["baseline", "evidence_required"]
Z_95 = 1.959963984540054


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normal_approx_ci_from_counts(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p_hat = successes / total
    return normal_approx_ci_from_proportion(p_hat, total, z=z)


def normal_approx_ci_from_proportion(value: float, total: int, z: float = Z_95) -> tuple[float, float]:
    if total <= 0 or np.isnan(value):
        return np.nan, np.nan
    bounded_value = min(max(float(value), 0.0), 1.0)
    variance = max(bounded_value * (1.0 - bounded_value), 0.0)
    se = math.sqrt(variance / total)
    low = max(0.0, bounded_value - z * se)
    high = min(1.0, bounded_value + z * se)
    return low, high


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return np.nan
    cumulative = 0.0
    for i in range(0, k + 1):
        cumulative += math.comb(n, i) * (0.5**n)
    return min(1.0, 2.0 * cumulative)


def normalize_quote(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] in '"\'“”‘’' and cleaned[-1] in '"\'“”‘’':
        cleaned = cleaned[1:-1].strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def quote_token_set(text: str) -> set[str]:
    normalized = normalize_quote(text).lower()
    return set(re.findall(r"[a-z0-9]+", normalized))


def pairwise_jaccard(quotes: list[str]) -> float:
    if len(quotes) < 2:
        return np.nan
    scores: list[float] = []
    token_sets = [quote_token_set(q) for q in quotes]
    for i, j in itertools.combinations(range(len(token_sets)), 2):
        a = token_sets[i]
        b = token_sets[j]
        union = a | b
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(a & b) / len(union))
    return float(np.mean(scores)) if scores else np.nan


def compute_fleiss_kappa(df: pd.DataFrame) -> float:
    if "repeat_idx" not in df.columns or df.empty:
        return np.nan

    expected_raters = int(df["repeat_idx"].max())
    if expected_raters < 2:
        return np.nan

    rating_rows: list[list[int]] = []
    for _, trial_df in df.groupby("trial_idx", sort=False):
        repeat_view = (
            trial_df.sort_values("repeat_idx")
            .drop_duplicates(subset=["repeat_idx"], keep="last")
            .copy()
        )
        if len(repeat_view) != expected_raters:
            continue

        # Include all task labels plus parser failures as INVALID.
        labels = np.where(
            repeat_view["invalid_output"].astype(bool),
            "INVALID",
            repeat_view["parsed_label"].astype(str).str.strip().str.upper(),
        )
        labels = [label if label in KAPPA_LABELS else "INVALID" for label in labels.tolist()]
        counts = [labels.count(label) for label in KAPPA_LABELS]
        rating_rows.append(counts)

    if not rating_rows:
        return np.nan

    mat = np.array(rating_rows, dtype=float)
    n_items = mat.shape[0]
    n_raters = expected_raters

    p_i = (np.sum(mat * mat, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)

    p_j = np.sum(mat, axis=0) / (n_items * n_raters)
    p_bar_e = np.sum(p_j * p_j)

    denom = 1.0 - p_bar_e
    if denom <= 0:
        return np.nan
    return float((p_bar - p_bar_e) / denom)


def compute_citation_jaccard(df: pd.DataFrame) -> float:
    trial_scores: list[float] = []
    for _, trial_df in df.groupby("trial_idx"):
        quotes = [
            str(q)
            for q in trial_df.loc[trial_df["quote_present"] & trial_df["quote"].astype(str).str.len().gt(0), "quote"]
            .astype(str)
            .tolist()
        ]
        score = pairwise_jaccard(quotes)
        if not np.isnan(score):
            trial_scores.append(score)
    return float(np.mean(trial_scores)) if trial_scores else np.nan


def prepare_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["parsed_label"] = out["parsed_label"].fillna("INVALID").astype(str).str.strip().str.upper()
    out["ground_truth"] = out["ground_truth"].fillna("").astype(str).str.strip().str.upper()

    if "format_valid" in out.columns:
        out["format_valid"] = to_bool(out["format_valid"])
    else:
        out["format_valid"] = out["parsed_label"].isin(ALLOWED_LABELS)

    if "quote_present" in out.columns:
        out["quote_present"] = to_bool(out["quote_present"])
    else:
        out["quote_present"] = out.get("quote", pd.Series([""] * len(out))).fillna("").astype(str).str.strip().ne("")

    if "quote_valid" in out.columns:
        out["quote_valid"] = to_bool(out["quote_valid"])
    else:
        out["quote_valid"] = False

    if "quote" not in out.columns:
        out["quote"] = ""
    out["quote"] = out["quote"].fillna("").astype(str)

    out["invalid_output"] = out["parsed_label"].isin({"INVALID", "ERROR"}) | (~out["format_valid"])
    out["unclear_output"] = (out["parsed_label"] == "UNCLEAR") & (~out["invalid_output"])
    out["answered_output"] = out["parsed_label"].isin(SCORING_LABELS) & (~out["invalid_output"])
    out["correct_answer"] = out["answered_output"] & (out["parsed_label"] == out["ground_truth"])
    # Mechanical validity check: quote is an exact abstract substring.
    out["correct_and_quote_present"] = out["correct_answer"] & out["quote_valid"]

    return out


def analyze_predictions(predictions_path: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    df = prepare_predictions(df)

    rows: list[dict[str, object]] = []

    grouped = df.groupby(["vendor", "model", "condition"], sort=False)
    for (vendor, model, condition), group in grouped:
        n_total = len(group)
        if n_total == 0:
            continue

        n_invalid = int(group["invalid_output"].sum())
        n_unclear = int(group["unclear_output"].sum())
        n_answered = int(group["answered_output"].sum())
        n_correct_total = int(group["correct_answer"].sum())

        coverage_pct = (n_answered / n_total) * 100.0
        unclear_pct = (n_unclear / n_total) * 100.0
        invalid_pct = (n_invalid / n_total) * 100.0
        coverage_ci_low, coverage_ci_high = normal_approx_ci_from_counts(n_answered, n_total)
        unclear_ci_low, unclear_ci_high = normal_approx_ci_from_counts(n_unclear, n_total)
        invalid_ci_low, invalid_ci_high = normal_approx_ci_from_counts(n_invalid, n_total)

        eval_df = group[group["answered_output"] & group["ground_truth"].isin(SCORING_LABELS)]
        if len(eval_df) > 0:
            y_true = eval_df["ground_truth"]
            y_pred = eval_df["parsed_label"]
            n_eval = len(eval_df)
            n_correct_eval = int((y_true == y_pred).sum())
            accuracy = float(accuracy_score(y_true, y_pred))
            f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=SCORING_LABELS, zero_division=0))
            accuracy_ci_low, accuracy_ci_high = normal_approx_ci_from_counts(n_correct_eval, n_eval)
            macro_f1_ci_low, macro_f1_ci_high = normal_approx_ci_from_proportion(f1_macro, n_eval)
        else:
            accuracy = np.nan
            f1_macro = np.nan
            accuracy_ci_low, accuracy_ci_high = np.nan, np.nan
            macro_f1_ci_low, macro_f1_ci_high = np.nan, np.nan

        overall_accuracy = float(n_correct_total / n_total)
        overall_accuracy_ci_low, overall_accuracy_ci_high = normal_approx_ci_from_counts(n_correct_total, n_total)

        fleiss = compute_fleiss_kappa(group)

        with_quote_pct = np.nan
        with_quote_ci_low = np.nan
        with_quote_ci_high = np.nan
        with_valid_quote_pct = np.nan
        with_valid_quote_ci_low = np.nan
        with_valid_quote_ci_high = np.nan
        quote_validity_pct = np.nan
        quote_validity_ci_low = np.nan
        quote_validity_ci_high = np.nan
        f1_valid_quote_only = np.nan
        f1_valid_quote_only_ci_low = np.nan
        f1_valid_quote_only_ci_high = np.nan
        correct_quote_present_pct = np.nan
        correct_quote_present_ci_low = np.nan
        correct_quote_present_ci_high = np.nan
        citation_jaccard = np.nan

        if condition == "evidence_required":
            n_with_quote = int(group["quote_present"].sum())
            with_quote_pct = (n_with_quote / n_total) * 100.0
            with_quote_ci_low, with_quote_ci_high = normal_approx_ci_from_counts(n_with_quote, n_total)

            n_quote_valid = int(group["quote_valid"].sum())
            with_valid_quote_pct = float(n_quote_valid / n_total * 100.0)
            with_valid_quote_ci_low, with_valid_quote_ci_high = normal_approx_ci_from_counts(n_quote_valid, n_total)

            if n_with_quote > 0:
                n_quote_valid = int(group.loc[group["quote_present"], "quote_valid"].sum())
                quote_validity_pct = float(group.loc[group["quote_present"], "quote_valid"].mean() * 100.0)
                quote_validity_ci_low, quote_validity_ci_high = normal_approx_ci_from_counts(n_quote_valid, n_with_quote)

            valid_quote_df = group[
                group["answered_output"] & group["quote_valid"] & group["ground_truth"].isin(SCORING_LABELS)
            ]
            if len(valid_quote_df) > 0:
                n_valid_quote = len(valid_quote_df)
                f1_valid_quote_only = float(
                    f1_score(
                        valid_quote_df["ground_truth"],
                        valid_quote_df["parsed_label"],
                        average="macro",
                        labels=SCORING_LABELS,
                        zero_division=0,
                    )
                )
                f1_valid_quote_only_ci_low, f1_valid_quote_only_ci_high = normal_approx_ci_from_proportion(
                    f1_valid_quote_only, n_valid_quote
                )

            n_correct_quote_present = int(group["correct_and_quote_present"].sum())
            correct_quote_present_pct = float(n_correct_quote_present / n_total * 100.0)
            correct_quote_present_ci_low, correct_quote_present_ci_high = normal_approx_ci_from_counts(
                n_correct_quote_present, n_total
            )
            citation_jaccard = compute_citation_jaccard(group)

        rows.append(
            {
                "vendor": vendor,
                "model": model,
                "condition": condition,
                "n_total": n_total,
                "n_answered": n_answered,
                "n_unclear": n_unclear,
                "n_invalid": n_invalid,
                "coverage_pct": coverage_pct,
                "coverage_ci_low_pct": coverage_ci_low * 100.0 if not np.isnan(coverage_ci_low) else np.nan,
                "coverage_ci_high_pct": coverage_ci_high * 100.0 if not np.isnan(coverage_ci_high) else np.nan,
                "unclear_pct": unclear_pct,
                "unclear_ci_low_pct": unclear_ci_low * 100.0 if not np.isnan(unclear_ci_low) else np.nan,
                "unclear_ci_high_pct": unclear_ci_high * 100.0 if not np.isnan(unclear_ci_high) else np.nan,
                "invalid_pct": invalid_pct,
                "invalid_ci_low_pct": invalid_ci_low * 100.0 if not np.isnan(invalid_ci_low) else np.nan,
                "invalid_ci_high_pct": invalid_ci_high * 100.0 if not np.isnan(invalid_ci_high) else np.nan,
                "accuracy_conditional": accuracy,
                "accuracy_conditional_ci_low": accuracy_ci_low,
                "accuracy_conditional_ci_high": accuracy_ci_high,
                "macro_f1": f1_macro,
                "macro_f1_ci_low": macro_f1_ci_low,
                "macro_f1_ci_high": macro_f1_ci_high,
                "overall_accuracy": overall_accuracy,
                "overall_accuracy_ci_low": overall_accuracy_ci_low,
                "overall_accuracy_ci_high": overall_accuracy_ci_high,
                "fleiss_kappa": fleiss,
                "with_quote_pct": with_quote_pct,
                "with_quote_ci_low_pct": with_quote_ci_low * 100.0 if not np.isnan(with_quote_ci_low) else np.nan,
                "with_quote_ci_high_pct": with_quote_ci_high * 100.0 if not np.isnan(with_quote_ci_high) else np.nan,
                "with_valid_quote_pct": with_valid_quote_pct,
                "with_valid_quote_ci_low_pct": (
                    with_valid_quote_ci_low * 100.0 if not np.isnan(with_valid_quote_ci_low) else np.nan
                ),
                "with_valid_quote_ci_high_pct": (
                    with_valid_quote_ci_high * 100.0 if not np.isnan(with_valid_quote_ci_high) else np.nan
                ),
                "quote_validity_pct": quote_validity_pct,
                "quote_validity_ci_low_pct": (
                    quote_validity_ci_low * 100.0 if not np.isnan(quote_validity_ci_low) else np.nan
                ),
                "quote_validity_ci_high_pct": (
                    quote_validity_ci_high * 100.0 if not np.isnan(quote_validity_ci_high) else np.nan
                ),
                "macro_f1_valid_quote_only": f1_valid_quote_only,
                "macro_f1_valid_quote_only_ci_low": f1_valid_quote_only_ci_low,
                "macro_f1_valid_quote_only_ci_high": f1_valid_quote_only_ci_high,
                "correct_and_quote_present_pct": correct_quote_present_pct,
                "correct_and_quote_present_ci_low_pct": (
                    correct_quote_present_ci_low * 100.0 if not np.isnan(correct_quote_present_ci_low) else np.nan
                ),
                "correct_and_quote_present_ci_high_pct": (
                    correct_quote_present_ci_high * 100.0 if not np.isnan(correct_quote_present_ci_high) else np.nan
                ),
                "citation_jaccard": citation_jaccard,
            }
        )

    metrics_df = pd.DataFrame(rows)

    # Sort with stable manuscript order.
    vendor_rank = {name: idx for idx, name in enumerate(VENDOR_ORDER)}
    condition_rank = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    metrics_df["_vendor_rank"] = metrics_df["vendor"].map(vendor_rank).fillna(999)
    metrics_df["_condition_rank"] = metrics_df["condition"].map(condition_rank).fillna(999)
    metrics_df = metrics_df.sort_values(["_vendor_rank", "_condition_rank"]).drop(
        columns=["_vendor_rank", "_condition_rank"]
    )

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    generate_manuscript_table(metrics_df, output_dir)
    generate_evidence_breakdown_table(df, output_dir)
    generate_confusion_table(df, output_dir)
    generate_mcnemar_table(df, output_dir)
    generate_fleiss_driver_table(df, metrics_df, output_dir)

    return metrics_df


def generate_manuscript_table(metrics_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    model_map = {
        "openai": "GPT-5.2",
        "gemini": "Gemini 3 Flash",
        "anthropic": "Claude Opus 4.5",
    }
    condition_map = {
        "baseline": "Baseline",
        "evidence_required": "Evidence Required",
    }

    table = metrics_df.copy()
    table["Model"] = table["vendor"].map(model_map).fillna(table["vendor"])
    table["Condition"] = table["condition"].map(condition_map).fillna(table["condition"])

    keep_cols = [
        "Model",
        "Condition",
        "coverage_pct",
        "unclear_pct",
        "invalid_pct",
        "macro_f1",
        "fleiss_kappa",
        "with_quote_pct",
        "with_valid_quote_pct",
        "macro_f1_valid_quote_only",
        "correct_and_quote_present_pct",
        "citation_jaccard",
    ]
    table = table[keep_cols]
    table = table.rename(
        columns={
            "coverage_pct": "Coverage [%]",
            "unclear_pct": "Unclear [%]",
            "invalid_pct": "Invalid [%]",
            "macro_f1": "Macro F1",
            "fleiss_kappa": "Fleiss' kappa",
            "with_quote_pct": "With quote [%]",
            "with_valid_quote_pct": "With mechanically valid quote [%]",
            "macro_f1_valid_quote_only": "Macro F1 (mechanically-valid only)",
            "correct_and_quote_present_pct": "Correct and mechanically-valid [%]",
            "citation_jaccard": "Citation Jaccard",
        }
    )

    # Save a machine-readable numeric version for downstream analysis.
    raw_csv_path = output_dir / "table_manuscript_raw.csv"
    table.to_csv(raw_csv_path, index=False)

    formatted = table.copy()
    pct_cols = [
        "Coverage [%]",
        "Unclear [%]",
        "Invalid [%]",
        "With quote [%]",
        "With mechanically valid quote [%]",
        "Correct and mechanically-valid [%]",
    ]
    score_cols = ["Macro F1", "Fleiss' kappa", "Macro F1 (mechanically-valid only)", "Citation Jaccard"]

    evidence_only_cols = [
        "With quote [%]",
        "With mechanically valid quote [%]",
        "Macro F1 (mechanically-valid only)",
        "Correct and mechanically-valid [%]",
        "Citation Jaccard",
    ]
    baseline_mask = formatted["Condition"] == "Baseline"
    formatted.loc[baseline_mask, evidence_only_cols] = np.nan

    for col in pct_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
    for col in score_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

    # Use one model label per two-row condition block to mirror manuscript tables.
    formatted["Model"] = formatted["Model"].where(~formatted.duplicated(subset=["Model"]), "")

    csv_path = output_dir / "table_manuscript.csv"
    formatted.to_csv(csv_path, index=False)

    txt_path = output_dir / "table_manuscript.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(formatted.to_string(index=False))

    print(f"Saved manuscript table to {csv_path}")
    return formatted


def generate_evidence_breakdown_table(df: pd.DataFrame, output_dir: Path):
    rows: list[dict[str, object]] = []
    evidence = df[df["condition"] == "evidence_required"].copy()

    for vendor in VENDOR_ORDER:
        vendor_df = evidence[evidence["vendor"] == vendor]
        if vendor_df.empty:
            continue

        n_total = len(vendor_df)
        correct_quote_present = int(vendor_df["correct_and_quote_present"].sum())
        correct_not_quote_present = int((vendor_df["correct_answer"] & ~vendor_df["quote_valid"]).sum())
        incorrect_answered = int((vendor_df["answered_output"] & ~vendor_df["correct_answer"]).sum())
        unclear_or_invalid = int((~vendor_df["answered_output"]).sum())

        rows.append(
            {
                "vendor": vendor,
                "n_total": n_total,
                "correct_and_quote_present": correct_quote_present,
                "correct_not_quote_present": correct_not_quote_present,
                "incorrect_answered": incorrect_answered,
                "unclear_or_invalid": unclear_or_invalid,
                "correct_and_quote_present_pct": correct_quote_present / n_total * 100.0,
                "correct_not_quote_present_pct": correct_not_quote_present / n_total * 100.0,
                "incorrect_answered_pct": incorrect_answered / n_total * 100.0,
                "unclear_or_invalid_pct": unclear_or_invalid / n_total * 100.0,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "table_evidence_breakdown.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved evidence breakdown to {out_path}")


def generate_confusion_table(df: pd.DataFrame, output_dir: Path):
    rows: list[dict[str, object]] = []

    for vendor in VENDOR_ORDER:
        for condition in CONDITION_ORDER:
            subset = df[
                (df["vendor"] == vendor)
                & (df["condition"] == condition)
                & df["answered_output"]
                & df["ground_truth"].isin(SCORING_LABELS)
            ]
            if subset.empty:
                continue

            cm = confusion_matrix(subset["ground_truth"], subset["parsed_label"], labels=SCORING_LABELS)

            for i, true_label in enumerate(SCORING_LABELS):
                for j, pred_label in enumerate(SCORING_LABELS):
                    rows.append(
                        {
                            "vendor": vendor,
                            "condition": condition,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": int(cm[i, j]),
                        }
                    )

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "table_confusion_long.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved confusion counts to {out_path}")


def generate_mcnemar_table(df: pd.DataFrame, output_dir: Path):
    rows: list[dict[str, object]] = []

    grouped = df.groupby(["vendor", "model"], sort=False)
    for (vendor, model), subset in grouped:
        trial_condition = (
            subset.groupby(["trial_idx", "condition"], as_index=False)
            .agg(correct_rate=("correct_answer", "mean"))
            .copy()
        )
        trial_condition["trial_correct"] = trial_condition["correct_rate"] > 0.5

        paired = trial_condition.pivot(index="trial_idx", columns="condition", values="trial_correct")
        if "baseline" not in paired.columns or "evidence_required" not in paired.columns:
            continue

        paired = paired.dropna(subset=["baseline", "evidence_required"])
        n_paired = len(paired)
        if n_paired == 0:
            continue

        baseline_correct = paired["baseline"].astype(bool)
        evidence_correct = paired["evidence_required"].astype(bool)

        both_correct = int((baseline_correct & evidence_correct).sum())
        both_incorrect = int((~baseline_correct & ~evidence_correct).sum())
        baseline_only_correct = int((baseline_correct & ~evidence_correct).sum())
        evidence_only_correct = int((~baseline_correct & evidence_correct).sum())

        discordant = baseline_only_correct + evidence_only_correct
        if discordant == 0:
            mcnemar_chi2 = 0.0
            mcnemar_chi2_cc = 0.0
            mcnemar_p = 1.0
            mcnemar_p_cc = 1.0
            mcnemar_p_exact = 1.0
        else:
            mcnemar_chi2 = ((baseline_only_correct - evidence_only_correct) ** 2) / discordant
            mcnemar_chi2_cc = (max(abs(baseline_only_correct - evidence_only_correct) - 1, 0) ** 2) / discordant
            mcnemar_p = math.erfc(math.sqrt(mcnemar_chi2 / 2.0))
            mcnemar_p_cc = math.erfc(math.sqrt(mcnemar_chi2_cc / 2.0))
            mcnemar_p_exact = exact_binomial_two_sided(min(baseline_only_correct, evidence_only_correct), discordant)

        baseline_trial_acc = float((both_correct + baseline_only_correct) / n_paired)
        evidence_trial_acc = float((both_correct + evidence_only_correct) / n_paired)
        baseline_trial_acc_ci_low, baseline_trial_acc_ci_high = normal_approx_ci_from_counts(
            both_correct + baseline_only_correct, n_paired
        )
        evidence_trial_acc_ci_low, evidence_trial_acc_ci_high = normal_approx_ci_from_counts(
            both_correct + evidence_only_correct, n_paired
        )

        rows.append(
            {
                "vendor": vendor,
                "model": model,
                "n_paired_trials": n_paired,
                "baseline_trial_accuracy": baseline_trial_acc,
                "baseline_trial_accuracy_ci_low": baseline_trial_acc_ci_low,
                "baseline_trial_accuracy_ci_high": baseline_trial_acc_ci_high,
                "evidence_trial_accuracy": evidence_trial_acc,
                "evidence_trial_accuracy_ci_low": evidence_trial_acc_ci_low,
                "evidence_trial_accuracy_ci_high": evidence_trial_acc_ci_high,
                "both_correct": both_correct,
                "both_incorrect": both_incorrect,
                "baseline_only_correct": baseline_only_correct,
                "evidence_only_correct": evidence_only_correct,
                "discordant_total": discordant,
                "mcnemar_chi2": mcnemar_chi2,
                "mcnemar_p": mcnemar_p,
                "mcnemar_chi2_cc": mcnemar_chi2_cc,
                "mcnemar_p_cc": mcnemar_p_cc,
                "mcnemar_p_exact": mcnemar_p_exact,
            }
        )

    out_df = pd.DataFrame(rows)
    vendor_rank = {name: idx for idx, name in enumerate(VENDOR_ORDER)}
    if not out_df.empty:
        out_df["_vendor_rank"] = out_df["vendor"].map(vendor_rank).fillna(999)
        out_df = out_df.sort_values(["_vendor_rank"]).drop(columns=["_vendor_rank"])

    out_path = output_dir / "table_mcnemar.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved McNemar comparisons to {out_path}")


def generate_fleiss_driver_table(df: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path):
    model_map = {
        "openai": "GPT-5.2",
        "gemini": "Gemini 3 Flash",
        "anthropic": "Claude Opus 4.5",
    }
    condition_map = {
        "baseline": "Baseline",
        "evidence_required": "Evidence Required",
    }

    fleiss_lookup = (
        metrics_df.set_index(["vendor", "model", "condition"])["fleiss_kappa"].to_dict()
        if not metrics_df.empty
        else {}
    )

    rows: list[dict[str, object]] = []
    grouped = df.groupby(["vendor", "model", "condition"], sort=False)

    for (vendor, model, condition), group in grouped:
        if "repeat_idx" not in group.columns or group.empty:
            continue

        expected_repeats = int(group["repeat_idx"].max())
        if expected_repeats < 2:
            continue

        complete_trials = 0
        fully_consistent_trials = 0
        disagreement_trials = 0
        disagreement_with_invalid = 0
        disagreement_with_unclear = 0
        disagreement_with_label_flip = 0
        primary_invalid_driver = 0
        primary_unclear_driver = 0
        primary_label_flip_driver = 0
        pattern_counts: dict[tuple[str, ...], int] = {}

        for _, trial_df in group.groupby("trial_idx", sort=False):
            repeat_view = (
                trial_df.sort_values("repeat_idx")
                .drop_duplicates(subset=["repeat_idx"], keep="last")
                .copy()
            )
            if len(repeat_view) != expected_repeats:
                continue

            complete_trials += 1

            labels = np.where(
                repeat_view["invalid_output"].astype(bool),
                "INVALID",
                repeat_view["parsed_label"].astype(str).str.strip().str.upper(),
            )
            labels = [label if label in KAPPA_LABELS else "INVALID" for label in labels.tolist()]
            unique_labels = set(labels)

            if len(unique_labels) == 1:
                fully_consistent_trials += 1
                continue

            disagreement_trials += 1
            has_invalid = "INVALID" in unique_labels
            has_unclear = "UNCLEAR" in unique_labels
            has_label_flip = len({label for label in unique_labels if label in SCORING_LABELS}) >= 2

            if has_invalid:
                disagreement_with_invalid += 1
            if has_unclear:
                disagreement_with_unclear += 1
            if has_label_flip:
                disagreement_with_label_flip += 1

            # Primary-driver precedence keeps categories mutually exclusive.
            if has_invalid:
                primary_invalid_driver += 1
            elif has_unclear:
                primary_unclear_driver += 1
            elif has_label_flip:
                primary_label_flip_driver += 1

            pattern = tuple(sorted(labels))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        top_patterns = ""
        if pattern_counts:
            top_items = sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True)[:3]
            top_patterns = "; ".join([f"{'/'.join(pattern)}:{count}" for pattern, count in top_items])

        disagree_denom = disagreement_trials
        rows.append(
            {
                "vendor": vendor,
                "model": model,
                "condition": condition,
                "Model": model_map.get(vendor, vendor),
                "Condition": condition_map.get(condition, condition),
                "fleiss_kappa": fleiss_lookup.get((vendor, model, condition), np.nan),
                "complete_trials": complete_trials,
                "expected_repeats": expected_repeats,
                "fully_consistent_trials": fully_consistent_trials,
                "fully_consistent_pct": (
                    fully_consistent_trials / complete_trials * 100.0 if complete_trials > 0 else np.nan
                ),
                "disagreement_trials": disagreement_trials,
                "disagreement_pct": disagreement_trials / complete_trials * 100.0 if complete_trials > 0 else np.nan,
                "disagreement_with_invalid": disagreement_with_invalid,
                "disagreement_with_invalid_pct_of_disagreements": (
                    disagreement_with_invalid / disagree_denom * 100.0 if disagree_denom > 0 else np.nan
                ),
                "disagreement_with_unclear": disagreement_with_unclear,
                "disagreement_with_unclear_pct_of_disagreements": (
                    disagreement_with_unclear / disagree_denom * 100.0 if disagree_denom > 0 else np.nan
                ),
                "disagreement_with_label_flip": disagreement_with_label_flip,
                "disagreement_with_label_flip_pct_of_disagreements": (
                    disagreement_with_label_flip / disagree_denom * 100.0 if disagree_denom > 0 else np.nan
                ),
                "primary_invalid_driver": primary_invalid_driver,
                "primary_unclear_driver": primary_unclear_driver,
                "primary_label_flip_driver": primary_label_flip_driver,
                "invalid_output_rows": int(group["invalid_output"].sum()),
                "unclear_output_rows": int(group["unclear_output"].sum()),
                "top_disagreement_patterns": top_patterns,
            }
        )

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        out_raw = output_dir / "table_fleiss_drivers_raw.csv"
        out_raw.write_text("", encoding="utf-8")
        out_pretty = output_dir / "table_fleiss_drivers.csv"
        out_pretty.write_text("", encoding="utf-8")
        print(f"Saved Fleiss-driver table to {out_pretty}")
        return

    vendor_rank = {name: idx for idx, name in enumerate(VENDOR_ORDER)}
    condition_rank = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    raw_df["_vendor_rank"] = raw_df["vendor"].map(vendor_rank).fillna(999)
    raw_df["_condition_rank"] = raw_df["condition"].map(condition_rank).fillna(999)
    raw_df = raw_df.sort_values(["_vendor_rank", "_condition_rank"]).drop(columns=["_vendor_rank", "_condition_rank"])

    out_raw = output_dir / "table_fleiss_drivers_raw.csv"
    raw_df.to_csv(out_raw, index=False)

    formatted = raw_df.copy()
    formatted["Fleiss' kappa"] = formatted["fleiss_kappa"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    formatted["Complete trials"] = formatted["complete_trials"].astype(int).astype(str)
    formatted["Fully consistent n (%)"] = formatted.apply(
        lambda r: (
            ""
            if int(r["complete_trials"]) == 0
            else f"{int(r['fully_consistent_trials'])} ({float(r['fully_consistent_pct']):.1f}%)"
        ),
        axis=1,
    )
    formatted["Disagreement n (%)"] = formatted.apply(
        lambda r: (
            ""
            if int(r["complete_trials"]) == 0
            else f"{int(r['disagreement_trials'])} ({float(r['disagreement_pct']):.1f}%)"
        ),
        axis=1,
    )
    formatted["INVALID in disagreement n (% of disagreements)"] = formatted.apply(
        lambda r: (
            ""
            if int(r["disagreement_trials"]) == 0
            else f"{int(r['disagreement_with_invalid'])} ({float(r['disagreement_with_invalid_pct_of_disagreements']):.1f}%)"
        ),
        axis=1,
    )
    formatted["UNCLEAR in disagreement n (% of disagreements)"] = formatted.apply(
        lambda r: (
            ""
            if int(r["disagreement_trials"]) == 0
            else f"{int(r['disagreement_with_unclear'])} ({float(r['disagreement_with_unclear_pct_of_disagreements']):.1f}%)"
        ),
        axis=1,
    )
    formatted["Label-flip in disagreement n (% of disagreements)"] = formatted.apply(
        lambda r: (
            ""
            if int(r["disagreement_trials"]) == 0
            else f"{int(r['disagreement_with_label_flip'])} ({float(r['disagreement_with_label_flip_pct_of_disagreements']):.1f}%)"
        ),
        axis=1,
    )
    formatted["Primary INVALID driver n"] = formatted["primary_invalid_driver"].astype(int).astype(str)
    formatted["Primary UNCLEAR driver n"] = formatted["primary_unclear_driver"].astype(int).astype(str)
    formatted["Primary label-flip driver n"] = formatted["primary_label_flip_driver"].astype(int).astype(str)
    formatted["INVALID outputs (rows)"] = formatted["invalid_output_rows"].astype(int).astype(str)
    formatted["UNCLEAR outputs (rows)"] = formatted["unclear_output_rows"].astype(int).astype(str)
    formatted["Top disagreement patterns (sorted repeats)"] = formatted["top_disagreement_patterns"].fillna("")

    keep_cols = [
        "Model",
        "Condition",
        "Fleiss' kappa",
        "Complete trials",
        "Fully consistent n (%)",
        "Disagreement n (%)",
        "INVALID in disagreement n (% of disagreements)",
        "UNCLEAR in disagreement n (% of disagreements)",
        "Label-flip in disagreement n (% of disagreements)",
        "Primary INVALID driver n",
        "Primary UNCLEAR driver n",
        "Primary label-flip driver n",
        "INVALID outputs (rows)",
        "UNCLEAR outputs (rows)",
        "Top disagreement patterns (sorted repeats)",
    ]
    formatted = formatted[keep_cols]

    out_pretty = output_dir / "table_fleiss_drivers.csv"
    formatted.to_csv(out_pretty, index=False)

    txt_path = output_dir / "table_fleiss_drivers.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(formatted.to_string(index=False))

    print(f"Saved Fleiss-driver table to {out_pretty}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze LLM citation experiment results")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for tables")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    predictions_path = Path(args.predictions) if args.predictions else script_dir / "results" / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        print(f"Error: predictions file not found: {predictions_path}")
        print("Run run_evaluation.py first.")
        return 1

    metrics_df = analyze_predictions(predictions_path, output_dir)

    if not metrics_df.empty:
        print("\nSummary (macro F1):")
        print(metrics_df[["vendor", "condition", "macro_f1", "coverage_pct", "fleiss_kappa"]].to_string(index=False))
    else:
        print("No metrics generated (empty input).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
