#!/usr/bin/env python3
"""
Analyze second-stage semantic grounding judgments.

Primary outputs:
- results/evidence_judgment_summary.csv
- results/evidence_judgment_by_label.csv
- results/evidence_judgment_f1_valid_only.csv
- results/table_judge_grounding.csv
- results/table_judge_grounding_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


SCORING_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER"]
ALLOWED_LABELS = [*SCORING_LABELS, "UNCLEAR"]
VENDOR_ORDER = ["openai", "gemini", "anthropic"]
MODEL_LABEL = {
    "openai": "GPT-5.2",
    "gemini": "Gemini 3 Flash",
    "anthropic": "Claude Opus 4.5",
}


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def summarize_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, sort=False, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}

        n_total = len(group)
        n_format_valid = int(group["format_valid"].sum())
        n_invalid = int((~group["format_valid"]).sum())

        n_semantically_grounded_all = int(group["supports_label"].sum())
        valid_group = group[group["format_valid"]]
        n_semantically_grounded_valid = int(valid_group["supports_label"].sum())

        semantic_grounding_rate_all = (100.0 * n_semantically_grounded_all / n_total) if n_total else 0.0
        semantic_grounding_rate_valid = (
            100.0 * n_semantically_grounded_valid / n_format_valid if n_format_valid else 0.0
        )
        format_valid_rate = (100.0 * n_format_valid / n_total) if n_total else 0.0

        row.update(
            {
                "n_total": n_total,
                "n_format_valid": n_format_valid,
                "n_invalid": n_invalid,
                "format_valid_rate_pct": format_valid_rate,
                "n_semantically_grounded_all": n_semantically_grounded_all,
                "semantic_grounding_rate_all_pct": semantic_grounding_rate_all,
                "n_semantically_grounded_valid": n_semantically_grounded_valid,
                "semantic_grounding_rate_valid_pct": semantic_grounding_rate_valid,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def macro_f1_valid_only(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, sort=False, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}

        semantically_grounded = group[group["format_valid"] & group["supports_label"]]
        scored = semantically_grounded[
            semantically_grounded["source_label"].isin(SCORING_LABELS)
            & semantically_grounded["ground_truth"].isin(SCORING_LABELS)
        ]

        n_total = len(group)
        n_semantically_grounded_valid = len(semantically_grounded)
        n_scored = len(scored)

        if n_scored > 0:
            macro_f1 = float(
                f1_score(
                    scored["ground_truth"],
                    scored["source_label"],
                    average="macro",
                    labels=SCORING_LABELS,
                    zero_division=0,
                )
            )
            accuracy = float(
                (scored["source_label"] == scored["ground_truth"]).mean()
            )
        else:
            macro_f1 = np.nan
            accuracy = np.nan

        semantic_grounding_rate_valid = (
            100.0 * n_semantically_grounded_valid / n_total if n_total else np.nan
        )

        row.update(
            {
                "n_total": n_total,
                "n_semantically_grounded_valid": n_semantically_grounded_valid,
                "semantic_grounding_rate_valid_pct": semantic_grounding_rate_valid,
                "n_scored_f1": n_scored,
                "macro_f1_judge_valid_only": macro_f1,
                "accuracy_judge_valid_only": accuracy,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_predictions_for_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vendor"] = out["vendor"].fillna("").astype(str).str.strip().str.lower()
    out["condition"] = out["condition"].fillna("").astype(str).str.strip()
    out["parsed_label"] = out["parsed_label"].fillna("INVALID").astype(str).str.strip().str.upper()
    out["ground_truth"] = out["ground_truth"].fillna("").astype(str).str.strip().str.upper()

    if "repeat_idx" in out.columns:
        out["repeat_idx"] = pd.to_numeric(out["repeat_idx"], errors="coerce").fillna(1).astype(int)
    else:
        out["repeat_idx"] = 1

    if "format_valid" in out.columns:
        out["format_valid"] = to_bool(out["format_valid"])
    else:
        out["format_valid"] = out["parsed_label"].isin(ALLOWED_LABELS)

    out["invalid_output"] = out["parsed_label"].isin({"INVALID", "ERROR"}) | (~out["format_valid"])
    out["answered_output"] = out["parsed_label"].isin(SCORING_LABELS) & (~out["invalid_output"])
    return out


def compute_baseline_metrics(
    predictions_df: pd.DataFrame,
    source_vendor: str,
    source_condition: str,
    source_repeats: list[int],
) -> tuple[float, float, int]:
    if predictions_df.empty:
        return np.nan, np.nan, 0

    subset = predictions_df[
        (predictions_df["vendor"] == source_vendor)
        & (predictions_df["condition"] == source_condition)
        & (predictions_df["repeat_idx"].isin(source_repeats))
    ].copy()

    n_total = len(subset)
    if n_total == 0:
        return np.nan, np.nan, 0

    n_answered = int(subset["answered_output"].sum())
    coverage_pct = (100.0 * n_answered / n_total) if n_total else np.nan

    scored = subset[subset["answered_output"] & subset["ground_truth"].isin(SCORING_LABELS)]
    if len(scored) > 0:
        macro_f1 = float(
            f1_score(
                scored["ground_truth"],
                scored["parsed_label"],
                average="macro",
                labels=SCORING_LABELS,
                zero_division=0,
            )
        )
    else:
        macro_f1 = np.nan

    return macro_f1, coverage_pct, n_total


def generate_judge_grounding_table(
    summary_df: pd.DataFrame,
    judgments_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vendor_rank = {name: idx for idx, name in enumerate(VENDOR_ORDER)}

    grouped = summary_df.groupby(["source_vendor", "source_model", "source_condition"], sort=False, dropna=False)
    for (source_vendor, source_model, source_condition), summary_group in grouped:
        source_rows = judgments_df[
            (judgments_df["source_vendor"] == source_vendor)
            & (judgments_df["source_model"] == source_model)
            & (judgments_df["source_condition"] == source_condition)
        ].copy()
        source_repeats = sorted(
            pd.to_numeric(source_rows["source_repeat_idx"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not source_repeats:
            source_repeats = [1]

        baseline_macro_f1, baseline_coverage_pct, baseline_n_total = compute_baseline_metrics(
            predictions_df=predictions_df,
            source_vendor=str(source_vendor),
            source_condition=str(source_condition),
            source_repeats=source_repeats,
        )

        summary_group = summary_group.copy()
        summary_group["_judge_rank"] = summary_group["judge_vendor"].map(vendor_rank).fillna(999)
        summary_group = summary_group.sort_values(["_judge_rank", "judge_vendor"], kind="stable").drop(
            columns=["_judge_rank"]
        )

        for _, row in summary_group.iterrows():
            if baseline_n_total > 0:
                semantically_grounded_predictions_pct = float(row["n_semantically_grounded_valid"]) / float(baseline_n_total) * 100.0
            else:
                semantically_grounded_predictions_pct = np.nan

            rows.append(
                {
                    "source_vendor": source_vendor,
                    "source_model": source_model,
                    "source_condition": source_condition,
                    "source_repeats": ",".join(str(rep) for rep in source_repeats),
                    "classification_model": MODEL_LABEL.get(str(source_vendor), str(source_model)),
                    "judge_vendor": row["judge_vendor"],
                    "judge_model": row["judge_model"],
                    "judge_llm": MODEL_LABEL.get(str(row["judge_vendor"]), str(row["judge_model"])),
                    "baseline_n_total": baseline_n_total,
                    "judged_rows": row["n_total"],
                    "semantically_grounded_rows": row["n_semantically_grounded_valid"],
                    "baseline_macro_f1": baseline_macro_f1,
                    "baseline_coverage_pct": baseline_coverage_pct,
                    "semantically_grounded_predictions_pct": semantically_grounded_predictions_pct,
                    "macro_f1_semantically_grounded": row["macro_f1_judge_valid_only"],
                }
            )

    table_raw = pd.DataFrame(rows)
    if table_raw.empty:
        out_raw = output_dir / "table_judge_grounding_raw.csv"
        out_pretty = output_dir / "table_judge_grounding.csv"
        out_txt = output_dir / "table_judge_grounding.txt"
        table_raw.to_csv(out_raw, index=False)
        table_raw.to_csv(out_pretty, index=False)
        out_txt.write_text("", encoding="utf-8")
        print(f"Saved judge-grounding comparison table to {out_pretty}")
        return table_raw

    table_raw["_source_rank"] = table_raw["source_vendor"].map(vendor_rank).fillna(999)
    table_raw["_judge_rank"] = table_raw["judge_vendor"].map(vendor_rank).fillna(999)
    table_raw = table_raw.sort_values(
        ["_source_rank", "source_condition", "_judge_rank", "judge_vendor"], kind="stable"
    ).drop(columns=["_source_rank", "_judge_rank"])

    out_raw = output_dir / "table_judge_grounding_raw.csv"
    table_raw.to_csv(out_raw, index=False)

    formatted = table_raw[
        [
            "classification_model",
            "judge_llm",
            "semantically_grounded_predictions_pct",
            "macro_f1_semantically_grounded",
        ]
    ].copy()
    formatted = formatted.rename(
        columns={
            "classification_model": "Classification model",
            "judge_llm": "Judge LLM",
            "semantically_grounded_predictions_pct": "Predictions deemed semantically grounded [%]",
            "macro_f1_semantically_grounded": "Macro F1 (semantically grounded only)",
        }
    )

    f1_cols = ["Macro F1 (semantically grounded only)"]
    pct_cols = ["Predictions deemed semantically grounded [%]"]
    for col in f1_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    for col in pct_cols:
        formatted[col] = formatted[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")

    # Render repeated model labels once to mimic merged cells in table software.
    duplicate_model_mask = formatted.duplicated(subset=["Classification model"])
    formatted.loc[duplicate_model_mask, "Classification model"] = ""

    out_pretty = output_dir / "table_judge_grounding.csv"
    out_txt = output_dir / "table_judge_grounding.txt"
    formatted.to_csv(out_pretty, index=False)
    out_txt.write_text(formatted.to_string(index=False), encoding="utf-8")
    print(f"Saved judge-grounding comparison table to {out_pretty}")
    return table_raw


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze evidence_judgments.csv and produce summary tables."
    )
    parser.add_argument(
        "--judgments",
        type=str,
        default=None,
        help="Path to evidence_judgments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary CSV outputs (default: results/)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions.csv for baseline F1/coverage values in judge table.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    judgments_path = (
        Path(args.judgments)
        if args.judgments
        else script_dir / "results" / "evidence_judgments.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else script_dir / "results" / "predictions.csv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not judgments_path.exists():
        print(f"Judgments file not found: {judgments_path}")
        return 1

    df = pd.read_csv(judgments_path)
    required_cols = {
        "source_vendor",
        "source_model",
        "source_condition",
        "source_repeat_idx",
        "source_label",
        "ground_truth",
        "judge_vendor",
        "judge_model",
        "format_valid",
        "supports_label",
    }
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        print(f"Judgments file missing required columns: {missing_str}")
        return 1

    df = df.copy()
    df["format_valid"] = to_bool(df["format_valid"])
    df["supports_label"] = to_bool(df["supports_label"])
    df["source_vendor"] = df["source_vendor"].fillna("").astype(str).str.strip().str.lower()
    df["judge_vendor"] = df["judge_vendor"].fillna("").astype(str).str.strip().str.lower()
    df["source_condition"] = df["source_condition"].fillna("").astype(str).str.strip()
    df["source_repeat_idx"] = pd.to_numeric(df["source_repeat_idx"], errors="coerce").fillna(1).astype(int)
    df["source_label"] = df["source_label"].fillna("").astype(str).str.strip().str.upper()
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str).str.strip().str.upper()

    if predictions_path.exists():
        predictions_df = prepare_predictions_for_baseline(pd.read_csv(predictions_path))
    else:
        print(f"Warning: predictions file not found; baseline columns will be blank in judge table: {predictions_path}")
        predictions_df = pd.DataFrame()

    summary_df = summarize_groups(
        df,
        group_cols=["source_vendor", "source_model", "source_condition", "judge_vendor", "judge_model"],
    )
    by_label_df = summarize_groups(
        df,
        group_cols=[
            "source_vendor",
            "source_model",
            "source_condition",
            "judge_vendor",
            "judge_model",
            "source_label",
        ],
    )
    f1_df = macro_f1_valid_only(
        df,
        group_cols=["source_vendor", "source_model", "source_condition", "judge_vendor", "judge_model"],
    )

    summary_df = summary_df.sort_values(
        ["source_vendor", "source_condition", "judge_vendor"], kind="stable"
    ).reset_index(drop=True)
    by_label_df = by_label_df.sort_values(
        ["source_vendor", "source_condition", "judge_vendor", "source_label"], kind="stable"
    ).reset_index(drop=True)
    f1_df = f1_df.sort_values(
        ["source_vendor", "source_condition", "judge_vendor"], kind="stable"
    ).reset_index(drop=True)

    summary_df = summary_df.merge(
        f1_df[
            [
                "source_vendor",
                "source_model",
                "source_condition",
                "judge_vendor",
                "judge_model",
                "n_scored_f1",
                "macro_f1_judge_valid_only",
                "accuracy_judge_valid_only",
            ]
        ],
        on=["source_vendor", "source_model", "source_condition", "judge_vendor", "judge_model"],
        how="left",
    )

    summary_path = output_dir / "evidence_judgment_summary.csv"
    by_label_path = output_dir / "evidence_judgment_by_label.csv"
    f1_path = output_dir / "evidence_judgment_f1_valid_only.csv"

    summary_df.to_csv(summary_path, index=False)
    by_label_df.to_csv(by_label_path, index=False)
    f1_df.to_csv(f1_path, index=False)
    generate_judge_grounding_table(summary_df, df, predictions_df, output_dir)

    print(f"Saved summary to {summary_path}")
    print(f"Saved label-level summary to {by_label_path}")
    print(f"Saved semantically-grounded-only F1 table to {f1_path}")
    if not summary_df.empty:
        display_cols = [
            "source_vendor",
            "judge_vendor",
            "n_total",
            "format_valid_rate_pct",
            "semantic_grounding_rate_valid_pct",
            "n_scored_f1",
            "macro_f1_judge_valid_only",
        ]
        print("\nTop-level summary:")
        print(summary_df[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
