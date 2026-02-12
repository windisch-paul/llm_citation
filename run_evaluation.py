#!/usr/bin/env python3
"""
LLM Citations Evaluation Pipeline

Reconstructs the manuscript pipeline for:
- Baseline condition: label only
- Evidence-required condition: label + verbatim quote

Each trial is evaluated repeatedly (default: 3 runs) per vendor and condition.
"""

from __future__ import annotations

import argparse
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

CONDITIONS = ["baseline", "evidence_required"]
ALLOWED_LABELS = ["LOCALIZED", "METASTATIC", "BOTH", "NEITHER", "UNCLEAR"]
OUTPUT_COLUMNS = [
    "trial_idx",
    "vendor",
    "model",
    "condition",
    "repeat_idx",
    "raw_output",
    "parsed_label",
    "format_valid",
    "quote",
    "quote_present",
    "quote_valid",
    "ground_truth",
]

SYSTEM_PROMPT_BASELINE = """You are given the title and abstract of a randomized oncology trial.
Classify eligibility scope into exactly one label using these definitions:
- LOCALIZED: only patients with localized/locally advanced disease that has not formed distant metastases are eligible.
- METASTATIC: only patients with distant metastatic disease are eligible.
- BOTH: both patients with localized/locally advanced and metastatic disease are eligible.
- NEITHER: patients with neither localized/locally advanced nor metastatic disease are eligible.
- UNCLEAR: abstract does not provide enough information to decide.
Return exactly one label in all caps and no other text."""

SYSTEM_PROMPT_EVIDENCE = """You are given the title and abstract of a randomized oncology trial.
Classify eligibility scope into exactly one label using these definitions:
- LOCALIZED: only patients with localized/locally advanced disease that has not formed distant metastases are eligible.
- METASTATIC: only patients with distant metastatic disease are eligible.
- BOTH: both patients with localized/locally advanced and metastatic disease are eligible.
- NEITHER: patients with neither localized/locally advanced nor metastatic disease are eligible.
- UNCLEAR: abstract does not provide enough information to decide.
If the label is LOCALIZED, METASTATIC, BOTH, or NEITHER, provide one supporting verbatim quote from the abstract.
If the label is UNCLEAR, the quote may be blank.
Output exactly two lines in this format:
LABEL: <LOCALIZED|METASTATIC|BOTH|NEITHER|UNCLEAR>
QUOTE: <exact contiguous substring copied from the abstract, or blank only when LABEL is UNCLEAR>
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


def build_user_prompt(title: str, abstract: str) -> str:
    return f"Title: {title}\n\nAbstract: {abstract}"


def normalize_text(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'“”‘’" and cleaned[-1] in "\"'“”‘’":
        cleaned = cleaned[1:-1].strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_baseline_output(raw_output: str | None) -> dict[str, object]:
    if raw_output is None:
        return {
            "parsed_label": "ERROR",
            "format_valid": False,
            "quote": "",
            "quote_present": False,
        }

    candidate = raw_output.strip().upper()
    if candidate in ALLOWED_LABELS:
        return {
            "parsed_label": candidate,
            "format_valid": True,
            "quote": "",
            "quote_present": False,
        }

    return {
        "parsed_label": "INVALID",
        "format_valid": False,
        "quote": "",
        "quote_present": False,
    }


def parse_evidence_output(raw_output: str | None) -> dict[str, object]:
    if raw_output is None:
        return {
            "parsed_label": "ERROR",
            "format_valid": False,
            "quote": "",
            "quote_present": False,
        }

    lines = raw_output.strip().splitlines()
    if len(lines) != 2:
        return {
            "parsed_label": "INVALID",
            "format_valid": False,
            "quote": "",
            "quote_present": False,
        }

    label_match = re.fullmatch(r"(?i)\s*LABEL\s*:\s*([A-Z_]+)\s*", lines[0])
    quote_match = re.fullmatch(r"(?i)\s*QUOTE\s*:\s*(.*)\s*", lines[1])

    parsed_label = "INVALID"
    if label_match:
        label = label_match.group(1).strip().upper()
        if label in ALLOWED_LABELS:
            parsed_label = label

    quote = quote_match.group(1).strip() if quote_match else ""
    quote_present = bool(quote)
    format_valid = parsed_label in ALLOWED_LABELS and (
        quote_present or parsed_label == "UNCLEAR"
    )

    return {
        "parsed_label": parsed_label,
        "format_valid": format_valid,
        "quote": quote,
        "quote_present": quote_present,
    }


def quote_is_valid(abstract: str, quote: str) -> bool:
    if not quote:
        return False
    normalized_abstract = normalize_text(abstract)
    normalized_quote = normalize_text(quote)
    return bool(normalized_quote) and normalized_quote in normalized_abstract


def call_openai(model: str, system_prompt: str, user_prompt: str) -> str | None:
    try:
        client = init_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=500,
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
                max_output_tokens=500,
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
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as exc:
        print(f"  Anthropic API error: {exc}", file=sys.stderr)
        return None


def call_api(
    vendor: str, model: str, system_prompt: str, user_prompt: str
) -> str | None:
    if vendor == "openai":
        return call_openai(model, system_prompt, user_prompt)
    if vendor == "gemini":
        return call_gemini(model, system_prompt, user_prompt)
    if vendor == "anthropic":
        return call_anthropic(model, system_prompt, user_prompt)
    raise ValueError(f"Unknown vendor: {vendor}")


def load_existing_predictions(predictions_path: Path) -> set[tuple[int, str, str, int]]:
    completed: set[tuple[int, str, str, int]] = set()
    if not predictions_path.exists():
        return completed

    df = pd.read_csv(predictions_path)
    needed = {"trial_idx", "vendor", "condition", "repeat_idx"}
    if not needed.issubset(df.columns):
        print(
            "Existing predictions.csv missing repeat-aware columns; ignoring resume state.",
            file=sys.stderr,
        )
        return completed

    for _, row in df.iterrows():
        completed.add(
            (
                int(row["trial_idx"]),
                str(row["vendor"]),
                str(row["condition"]),
                int(row["repeat_idx"]),
            )
        )
    return completed


def ensure_predictions_schema(predictions_path: Path):
    """Refuse writing to an existing file with incompatible schema."""
    if not predictions_path.exists():
        return

    try:
        existing = pd.read_csv(predictions_path, nrows=0)
    except Exception as exc:
        raise RuntimeError(f"Could not read existing predictions file: {exc}") from exc

    existing_cols = list(existing.columns)
    if existing_cols != OUTPUT_COLUMNS:
        raise RuntimeError(
            "Existing predictions.csv has incompatible columns.\n"
            f"Expected: {OUTPUT_COLUMNS}\n"
            f"Found:    {existing_cols}\n"
            "Move/rename the file or set --predictions to a new path."
        )


def save_prediction(
    predictions_path: Path,
    trial_idx: int,
    vendor: str,
    model: str,
    condition: str,
    repeat_idx: int,
    raw_output: str | None,
    parsed_label: str,
    format_valid: bool,
    quote: str,
    quote_present: bool,
    quote_valid: bool,
    ground_truth: str,
):
    row = {
        "trial_idx": trial_idx,
        "vendor": vendor,
        "model": model,
        "condition": condition,
        "repeat_idx": repeat_idx,
        "raw_output": raw_output,
        "parsed_label": parsed_label,
        "format_valid": format_valid,
        "quote": quote,
        "quote_present": quote_present,
        "quote_valid": quote_valid,
        "ground_truth": ground_truth,
    }

    df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
    write_header = not predictions_path.exists()
    df.to_csv(predictions_path, mode="a", header=write_header, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM citation evaluation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expected API call count without running",
    )
    parser.add_argument(
        "--max-trials", type=int, default=None, help="Limit number of trials"
    )
    parser.add_argument(
        "--condition", choices=CONDITIONS, default=None, help="Run one condition only"
    )
    parser.add_argument(
        "--vendor",
        choices=list(MODELS.keys()),
        default=None,
        help="Run one vendor only",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated runs per trial/setting",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5, help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--predictions", type=str, default=None, help="Path to predictions.csv"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing predictions and rerun all",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 1

    load_dotenv()

    script_dir = Path(__file__).parent
    trials_path = script_dir / "data" / "trials.csv"
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else script_dir / "results" / "predictions.csv"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    trials_df = pd.read_csv(trials_path)
    if args.max_trials is not None:
        trials_df = trials_df.head(args.max_trials)

    conditions = [args.condition] if args.condition else CONDITIONS
    vendors = [args.vendor] if args.vendor else list(MODELS.keys())

    total_calls = len(trials_df) * len(vendors) * len(conditions) * args.repeats

    if args.dry_run:
        print(f"Trials: {len(trials_df)}")
        print(f"Vendors: {', '.join(vendors)}")
        print(f"Conditions: {', '.join(conditions)}")
        print(f"Repeats per setting: {args.repeats}")
        print(f"Total API calls: {total_calls}")
        return 0

    try:
        ensure_predictions_schema(predictions_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if "openai" in vendors and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)
    if "gemini" in vendors and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set", file=sys.stderr)
    if "anthropic" in vendors and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set", file=sys.stderr)

    completed = set() if args.no_resume else load_existing_predictions(predictions_path)
    print(f"Loaded {len(completed)} completed predictions")

    pending = 0
    for _, row in trials_df.iterrows():
        trial_idx = int(row["trial_idx"])
        for condition in conditions:
            for vendor in vendors:
                for repeat_idx in range(1, args.repeats + 1):
                    key = (trial_idx, vendor, condition, repeat_idx)
                    if key not in completed:
                        pending += 1

    print(f"Total planned calls: {total_calls}; pending: {pending}")

    if pending == 0:
        print("Nothing to do.")
        return 0

    call_count = 0
    for condition in conditions:
        system_prompt = (
            SYSTEM_PROMPT_BASELINE
            if condition == "baseline"
            else SYSTEM_PROMPT_EVIDENCE
        )
        print(f"\n=== Condition: {condition} ===")

        for _, row in trials_df.iterrows():
            trial_idx = int(row["trial_idx"])
            title = str(row["title"])
            abstract = str(row["abstract"])
            ground_truth = str(row["ground_truth"]).strip().upper()

            for vendor in vendors:
                model = MODELS[vendor]

                for repeat_idx in range(1, args.repeats + 1):
                    key = (trial_idx, vendor, condition, repeat_idx)
                    if key in completed:
                        continue

                    user_prompt = build_user_prompt(title, abstract)
                    print(
                        f"[{call_count + 1}/{pending}] trial={trial_idx} vendor={vendor} condition={condition} repeat={repeat_idx}"
                    )
                    raw_output = call_api(vendor, model, system_prompt, user_prompt)

                    if condition == "baseline":
                        parsed = parse_baseline_output(raw_output)
                    else:
                        parsed = parse_evidence_output(raw_output)

                    quote = str(parsed["quote"])
                    quote_valid = (
                        quote_is_valid(abstract, quote)
                        if condition == "evidence_required"
                        and bool(parsed["format_valid"])
                        else False
                    )

                    save_prediction(
                        predictions_path=predictions_path,
                        trial_idx=trial_idx,
                        vendor=vendor,
                        model=model,
                        condition=condition,
                        repeat_idx=repeat_idx,
                        raw_output=raw_output,
                        parsed_label=str(parsed["parsed_label"]),
                        format_valid=bool(parsed["format_valid"]),
                        quote=quote,
                        quote_present=bool(parsed["quote_present"]),
                        quote_valid=quote_valid,
                        ground_truth=ground_truth,
                    )

                    parsed_label = str(parsed["parsed_label"])
                    if condition == "evidence_required":
                        print(
                            f"  -> label={parsed_label}, format_valid={bool(parsed['format_valid'])}, mechanically_valid={quote_valid}"
                        )
                    else:
                        print(f"  -> label={parsed_label}")

                    call_count += 1
                    time.sleep(args.sleep)

    print(f"\nCompleted {call_count} API calls")
    print(f"Saved predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
