#!/usr/bin/env python3
"""
Second-stage semantic grounding judgment pipeline.

This script audits whether a source model's quote semantically grounds the
source model's assigned label. By default it judges all available first-stage
source repeats, while still calling each judge model once per selected source row.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# API clients (lazy init)
openai_client = None
genai = None
anthropic_client = None


MODELS = {
    "openai": "gpt-5.2-2025-12-11",
    "gemini": "gemini-3-flash-preview",
    "anthropic": "claude-opus-4-5-20251101",
}

ALLOWED_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER", "UNCLEAR"]
ALLOWED_VERDICTS = ["SUPPORTED", "NOT_SUPPORTED"]
SOURCE_CONDITIONS = ["evidence_required"]

OUTPUT_COLUMNS = [
    "trial_idx",
    "source_vendor",
    "source_model",
    "source_condition",
    "source_repeat_idx",
    "source_label",
    "source_quote",
    "source_quote_sha1",
    "source_quote_valid",
    "ground_truth",
    "judge_vendor",
    "judge_model",
    "raw_output",
    "verdict",
    "format_valid",
    "supports_label",
]

SYSTEM_PROMPT_JUDGE = """You are auditing whether a quoted passage semantically grounds a pre-assigned eligibility-scope label.

Label definitions:
- LOCALIZED: only patients with localized/locally advanced disease that has not formed distant metastases are eligible.
- METASTATIC: only patients with distant metastatic disease are eligible.
- BOTH: both patients with localized/locally advanced and metastatic disease are eligible.
- NEITHER: patients with neither localized/locally advanced nor metastatic disease are eligible.

Judging rule:
- Decide whether the provided QUOTE semantically grounds the provided ASSIGNED LABEL.
- If the quote is too weak, ambiguous, irrelevant, or points to a different label, return NOT_SUPPORTED.
- Ignore substring/exact-match mechanics; this is a semantic grounding check.

Output exactly one line:
VERDICT: <SUPPORTED|NOT_SUPPORTED>

Do not output anything else."""


def init_openai():
    global openai_client
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI()
    return openai_client


def init_gemini():
    global genai
    if genai is None:
        from google import genai as genai_module

        genai = genai_module.Client(api_key=os.environ["GEMINI_API_KEY"])
    return genai


def init_anthropic():
    global anthropic_client
    if anthropic_client is None:
        import anthropic

        anthropic_client = anthropic.Anthropic()
    return anthropic_client


def call_openai(model: str, system_prompt: str, user_prompt: str) -> str | None:
    try:
        client = init_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=300,
        )
        return (
            response.choices[0].message.content.strip()
            if response.choices[0].message.content
            else None
        )
    except Exception as exc:
        print(f"  OpenAI API error: {exc}", file=sys.stderr)
        return None


def call_gemini(model: str, system_prompt: str, user_prompt: str) -> str | None:
    try:
        from google.genai import types

        client = init_gemini()
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=300,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception as exc:
        print(f"  Gemini API error: {exc}", file=sys.stderr)
        return None


def call_anthropic(model: str, system_prompt: str, user_prompt: str) -> str | None:
    try:
        client = init_anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as exc:
        print(f"  Anthropic API error: {exc}", file=sys.stderr)
        return None


def call_api(vendor: str, model: str, system_prompt: str, user_prompt: str) -> str | None:
    if vendor == "openai":
        return call_openai(model, system_prompt, user_prompt)
    if vendor == "gemini":
        return call_gemini(model, system_prompt, user_prompt)
    if vendor == "anthropic":
        return call_anthropic(model, system_prompt, user_prompt)
    raise ValueError(f"Unknown vendor: {vendor}")


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_judges(raw: str) -> list[str]:
    parsed: list[str] = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in MODELS:
            raise ValueError(
                f"Unknown judge vendor '{name}'. Expected one of: {', '.join(MODELS.keys())}"
            )
        if name not in parsed:
            parsed.append(name)
    if not parsed:
        raise ValueError("No judge vendors configured.")
    return parsed


def parse_source_vendors(raw: str) -> list[str]:
    candidate = raw.strip().lower()
    if candidate == "all":
        return list(MODELS.keys())

    parsed: list[str] = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in MODELS:
            raise ValueError(
                f"Unknown source vendor '{name}'. Expected one of: {', '.join(MODELS.keys())}, or 'all'"
            )
        if name not in parsed:
            parsed.append(name)
    if not parsed:
        raise ValueError("No source vendors configured.")
    return parsed


def build_user_prompt(source_label: str, source_quote: str) -> str:
    return (
        f"ASSIGNED LABEL: {source_label}\n\n"
        f'QUOTE:\n"""\n{source_quote}\n"""\n'
    )


def parse_judgment_output(raw_output: str | None) -> dict[str, object]:
    if raw_output is None:
        return {
            "verdict": "INVALID",
            "format_valid": False,
            "supports_label": None,
        }

    verdict_match = re.search(
        r"(?im)^\s*VERDICT\s*:\s*(SUPPORTED|NOT_SUPPORTED)\s*$", raw_output
    )

    if not verdict_match:
        return {
            "verdict": "INVALID",
            "format_valid": False,
            "supports_label": None,
        }

    verdict = verdict_match.group(1).strip().upper()

    format_valid = verdict in ALLOWED_VERDICTS
    supports_label = verdict == "SUPPORTED" if format_valid else None

    return {
        "verdict": verdict if format_valid else "INVALID",
        "format_valid": format_valid,
        "supports_label": supports_label,
    }


def ensure_judgments_schema(judgments_path: Path):
    if not judgments_path.exists():
        return

    try:
        existing = pd.read_csv(judgments_path, nrows=0)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read existing judgments file: {exc}"
        ) from exc

    existing_cols = list(existing.columns)
    if existing_cols != OUTPUT_COLUMNS:
        raise RuntimeError(
            "Existing evidence_judgments.csv has incompatible columns.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Found:    {existing_cols}\n"
            "Move/rename the file or set --judgments-out to a new path."
        )


def resume_key_from_row(row: pd.Series) -> tuple[int, str, str, str, int, str, str, str]:
    return (
        int(row["trial_idx"]),
        str(row["source_vendor"]),
        str(row["source_model"]),
        str(row["source_condition"]),
        int(row["source_repeat_idx"]),
        str(row["source_label"]),
        str(row["source_quote_sha1"]),
        str(row["judge_vendor"]),
    )


def load_existing_judgments(
    judgments_path: Path,
) -> set[tuple[int, str, str, str, int, str, str, str]]:
    completed: set[tuple[int, str, str, str, int, str, str, str]] = set()
    if not judgments_path.exists():
        return completed

    df = pd.read_csv(judgments_path)
    needed = {
        "trial_idx",
        "source_vendor",
        "source_model",
        "source_condition",
        "source_repeat_idx",
        "source_label",
        "source_quote_sha1",
        "judge_vendor",
    }
    if not needed.issubset(df.columns):
        print(
            "Existing evidence_judgments.csv missing resume columns; ignoring resume state.",
            file=sys.stderr,
        )
        return completed

    for _, row in df.iterrows():
        completed.add(resume_key_from_row(row))
    return completed


def save_judgment(
    judgments_path: Path,
    trial_idx: int,
    source_vendor: str,
    source_model: str,
    source_condition: str,
    source_repeat_idx: int,
    source_label: str,
    source_quote: str,
    source_quote_sha1: str,
    source_quote_valid: bool,
    ground_truth: str,
    judge_vendor: str,
    judge_model: str,
    raw_output: str | None,
    verdict: str,
    format_valid: bool,
    supports_label: bool | None,
):
    row = {
        "trial_idx": trial_idx,
        "source_vendor": source_vendor,
        "source_model": source_model,
        "source_condition": source_condition,
        "source_repeat_idx": source_repeat_idx,
        "source_label": source_label,
        "source_quote": source_quote,
        "source_quote_sha1": source_quote_sha1,
        "source_quote_valid": source_quote_valid,
        "ground_truth": ground_truth,
        "judge_vendor": judge_vendor,
        "judge_model": judge_model,
        "raw_output": raw_output,
        "verdict": verdict,
        "format_valid": format_valid,
        "supports_label": supports_label,
    }
    df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
    write_header = not judgments_path.exists()
    df.to_csv(judgments_path, mode="a", header=write_header, index=False)


def load_source_rows(
    predictions_path: Path,
    source_vendors: list[str],
    source_condition: str,
    source_repeat: int,
    all_source_repeats: bool,
    allow_format_invalid: bool,
    require_mechanical_valid: bool,
    max_trials: int | None,
) -> pd.DataFrame:
    if not predictions_path.exists():
        raise RuntimeError(f"Predictions file not found: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    required_predictions_cols = {
        "trial_idx",
        "vendor",
        "model",
        "condition",
        "parsed_label",
        "quote",
        "ground_truth",
    }
    missing_predictions_cols = required_predictions_cols - set(predictions_df.columns)
    if missing_predictions_cols:
        missing_str = ", ".join(sorted(missing_predictions_cols))
        raise RuntimeError(f"predictions.csv is missing required columns: {missing_str}")

    if "repeat_idx" not in predictions_df.columns:
        predictions_df["repeat_idx"] = 1
    else:
        predictions_df["repeat_idx"] = (
            pd.to_numeric(predictions_df["repeat_idx"], errors="coerce")
            .fillna(1)
            .astype(int)
        )

    if "format_valid" in predictions_df.columns:
        predictions_df["format_valid"] = to_bool(predictions_df["format_valid"])
    else:
        predictions_df["format_valid"] = predictions_df["parsed_label"].fillna("").astype(str).str.strip().str.upper().isin(ALLOWED_LABELS)

    if "quote_valid" in predictions_df.columns:
        predictions_df["quote_valid"] = to_bool(predictions_df["quote_valid"])
    else:
        predictions_df["quote_valid"] = False

    if "quote_present" in predictions_df.columns:
        predictions_df["quote_present"] = to_bool(predictions_df["quote_present"])
    else:
        predictions_df["quote_present"] = predictions_df["quote"].fillna("").astype(str).str.strip().ne("")

    predictions_df["vendor"] = predictions_df["vendor"].fillna("").astype(str).str.strip().str.lower()
    predictions_df["condition"] = predictions_df["condition"].fillna("").astype(str).str.strip()
    predictions_df["parsed_label"] = (
        predictions_df["parsed_label"].fillna("").astype(str).str.strip().str.upper()
    )
    predictions_df["quote"] = predictions_df["quote"].fillna("").astype(str)
    predictions_df["ground_truth"] = (
        predictions_df["ground_truth"].fillna("").astype(str).str.strip().str.upper()
    )
    predictions_df["trial_idx"] = pd.to_numeric(
        predictions_df["trial_idx"], errors="coerce"
    ).astype("Int64")
    predictions_df = predictions_df[predictions_df["trial_idx"].notna()].copy()
    predictions_df["trial_idx"] = predictions_df["trial_idx"].astype(int)

    subset = predictions_df[
        (predictions_df["vendor"].isin(source_vendors))
        & (predictions_df["condition"] == source_condition)
    ].copy()

    if not all_source_repeats:
        subset = subset[subset["repeat_idx"] == source_repeat].copy()

    # Always exclude rows that do not provide actionable evidence for judgment.
    subset = subset[subset["parsed_label"] != "UNCLEAR"].copy()
    subset = subset[subset["quote"].str.strip().ne("")].copy()

    if not allow_format_invalid:
        subset = subset[subset["format_valid"]].copy()
    if require_mechanical_valid:
        subset = subset[subset["quote_valid"]].copy()

    subset = (
        subset.sort_values(["trial_idx", "repeat_idx"])
        .drop_duplicates(subset=["trial_idx", "vendor", "condition", "repeat_idx"], keep="last")
        .copy()
    )

    if max_trials is not None:
        trial_ids = subset["trial_idx"].drop_duplicates().head(max_trials).tolist()
        subset = subset[subset["trial_idx"].isin(trial_ids)].copy()

    merged = subset.rename(
        columns={
            "vendor": "source_vendor",
            "model": "source_model",
            "condition": "source_condition",
            "repeat_idx": "source_repeat_idx",
            "parsed_label": "source_label",
            "quote": "source_quote",
            "quote_valid": "source_quote_valid",
        }
    )

    merged["source_quote_sha1"] = merged["source_quote"].map(sha1_text)
    merged = merged.sort_values(["trial_idx", "source_repeat_idx"]).reset_index(drop=True)
    return merged


def print_prompt_preview(source_rows: pd.DataFrame):
    print("\n=== SYSTEM PROMPT (Judge) ===")
    print(SYSTEM_PROMPT_JUDGE)
    if source_rows.empty:
        print("\nNo source rows selected; cannot render user prompt example.")
        return

    row = source_rows.iloc[0]
    sample = build_user_prompt(
        source_label=str(row["source_label"]),
        source_quote=str(row["source_quote"]),
    )
    print("\n=== USER PROMPT EXAMPLE (first selected row) ===")
    print(sample)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run second-stage semantic grounding judgments."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows and planned API call count without calling APIs.",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print system prompt and one user prompt example.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit to first N unique trials after filtering source rows.",
    )
    parser.add_argument(
        "--source-vendors",
        type=str,
        default="all",
        help="Comma-separated source vendors, or 'all' (default).",
    )
    parser.add_argument(
        "--source-vendor",
        type=str,
        default=None,
        help="Single source vendor (deprecated alias for --source-vendors).",
    )
    parser.add_argument(
        "--source-condition",
        choices=SOURCE_CONDITIONS,
        default="evidence_required",
        help="Source condition in predictions.csv.",
    )
    parser.add_argument(
        "--source-repeat",
        type=int,
        default=1,
        help="Source repeat index used only with --single-source-repeat (default: 1).",
    )
    parser.add_argument(
        "--all-source-repeats",
        dest="all_source_repeats",
        action="store_true",
        default=True,
        help="Judge all available source repeats (default).",
    )
    parser.add_argument(
        "--single-source-repeat",
        dest="all_source_repeats",
        action="store_false",
        help="Judge only one source repeat selected by --source-repeat.",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default="openai,gemini,anthropic",
        help="Comma-separated judge vendors.",
    )
    parser.add_argument(
        "--allow-format-invalid",
        action="store_true",
        help="Include source rows where first-stage format_valid is false.",
    )
    parser.add_argument(
        "--require-mechanical-valid",
        action="store_true",
        help="Only include source rows where first-stage mechanically-valid quote checks (quote_valid) are true.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5, help="Delay between API calls (seconds)."
    )
    parser.add_argument(
        "--predictions-in",
        type=str,
        default=None,
        help="Path to first-stage predictions.csv.",
    )
    parser.add_argument(
        "--judgments-out",
        type=str,
        default=None,
        help="Path to second-stage evidence_judgments.csv.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing evidence_judgments.csv and rerun all selected calls.",
    )
    args = parser.parse_args()

    if args.max_trials is not None and args.max_trials < 1:
        print("--max-trials must be >= 1", file=sys.stderr)
        return 1
    if args.source_repeat < 1:
        print("--source-repeat must be >= 1", file=sys.stderr)
        return 1

    if args.source_vendor is not None and args.source_vendors != "all":
        print(
            "Use either --source-vendor or --source-vendors, not both.",
            file=sys.stderr,
        )
        return 1

    source_vendor_raw = args.source_vendor if args.source_vendor is not None else args.source_vendors

    try:
        source_vendors = parse_source_vendors(source_vendor_raw)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        judge_vendors = parse_judges(args.judges)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    script_dir = Path(__file__).parent
    predictions_path = (
        Path(args.predictions_in)
        if args.predictions_in
        else script_dir / "results" / "predictions.csv"
    )
    judgments_path = (
        Path(args.judgments_out)
        if args.judgments_out
        else script_dir / "results" / "evidence_judgments.csv"
    )
    judgments_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        source_rows = load_source_rows(
            predictions_path=predictions_path,
            source_vendors=source_vendors,
            source_condition=args.source_condition,
            source_repeat=args.source_repeat,
            all_source_repeats=args.all_source_repeats,
            allow_format_invalid=args.allow_format_invalid,
            require_mechanical_valid=args.require_mechanical_valid,
            max_trials=args.max_trials,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    planned_calls = len(source_rows) * len(judge_vendors)
    print(f"Selected source rows: {len(source_rows)}")
    print(f"Source vendors: {', '.join(source_vendors)}")
    print(f"Source condition: {args.source_condition}")
    if args.all_source_repeats:
        print("Source repeats: all (default)")
    else:
        print(f"Source repeat: {args.source_repeat}")
    print(f"Judge vendors: {', '.join(judge_vendors)}")
    print("Source filtering: UNCLEAR and empty-quote rows are excluded")
    print(f"Planned API calls: {planned_calls}")

    if not source_rows.empty:
        print("Source row breakdown:")
        per_source = source_rows["source_vendor"].value_counts().sort_index()
        for source_vendor, n_rows in per_source.items():
            print(
                f"  {source_vendor}: {int(n_rows)} rows -> {int(n_rows) * len(judge_vendors)} calls"
            )

    if args.print_prompts:
        print_prompt_preview(source_rows)

    if args.dry_run:
        return 0

    if source_rows.empty:
        print("No source rows selected; nothing to do.", file=sys.stderr)
        return 1

    load_dotenv()
    if "openai" in judge_vendors and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)
    if "gemini" in judge_vendors and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set", file=sys.stderr)
    if "anthropic" in judge_vendors and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set", file=sys.stderr)

    try:
        ensure_judgments_schema(judgments_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    completed = set() if args.no_resume else load_existing_judgments(judgments_path)
    print(f"Loaded {len(completed)} completed judgments")

    pending = 0
    for row in source_rows.itertuples(index=False):
        for judge_vendor in judge_vendors:
            key = (
                int(row.trial_idx),
                str(row.source_vendor),
                str(row.source_model),
                str(row.source_condition),
                int(row.source_repeat_idx),
                str(row.source_label),
                str(row.source_quote_sha1),
                judge_vendor,
            )
            if key not in completed:
                pending += 1

    print(f"Pending API calls: {pending}")
    if pending == 0:
        print("Nothing to do.")
        return 0

    call_count = 0
    for row in source_rows.itertuples(index=False):
        for judge_vendor in judge_vendors:
            key = (
                int(row.trial_idx),
                str(row.source_vendor),
                str(row.source_model),
                str(row.source_condition),
                int(row.source_repeat_idx),
                str(row.source_label),
                str(row.source_quote_sha1),
                judge_vendor,
            )
            if key in completed:
                continue

            judge_model = MODELS[judge_vendor]
            user_prompt = build_user_prompt(
                source_label=str(row.source_label),
                source_quote=str(row.source_quote),
            )
            print(
                f"[{call_count + 1}/{pending}] trial={int(row.trial_idx)} "
                f"source={row.source_vendor} repeat={int(row.source_repeat_idx)} "
                f"judge={judge_vendor}"
            )
            raw_output = call_api(
                vendor=judge_vendor,
                model=judge_model,
                system_prompt=SYSTEM_PROMPT_JUDGE,
                user_prompt=user_prompt,
            )
            parsed = parse_judgment_output(raw_output)
            save_judgment(
                judgments_path=judgments_path,
                trial_idx=int(row.trial_idx),
                source_vendor=str(row.source_vendor),
                source_model=str(row.source_model),
                source_condition=str(row.source_condition),
                source_repeat_idx=int(row.source_repeat_idx),
                source_label=str(row.source_label),
                source_quote=str(row.source_quote),
                source_quote_sha1=str(row.source_quote_sha1),
                source_quote_valid=bool(row.source_quote_valid),
                ground_truth=str(row.ground_truth),
                judge_vendor=judge_vendor,
                judge_model=judge_model,
                raw_output=raw_output,
                verdict=str(parsed["verdict"]),
                format_valid=bool(parsed["format_valid"]),
                supports_label=(
                    bool(parsed["supports_label"])
                    if parsed["supports_label"] is not None
                    else None
                ),
            )
            print(
                f"  -> verdict={parsed['verdict']}, format_valid={bool(parsed['format_valid'])}"
            )
            call_count += 1
            time.sleep(args.sleep)

    print(f"\nCompleted {call_count} API calls")
    print(f"Saved judgments to {judgments_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
