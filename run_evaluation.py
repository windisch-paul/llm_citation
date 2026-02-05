#!/usr/bin/env python3
"""
LLM Citations Evaluation Pipeline

Evaluates whether requiring verbatim evidence (quotes) improves trustworthiness
for eligibility classification of oncology RCT abstracts.

Task labels:
  - LOCALIZED
  - METASTATIC
  - BOTH
  - NEITHER

Conditions:
  - baseline: output label only
  - evidence_required: output JSON {label, quote} where quote is a verbatim
    substring of the provided abstract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


# API clients (imported lazily)
openai_client = None
genai_client = None
anthropic_client = None


VALID_LABELS = {"LOCALIZED", "METASTATIC", "BOTH", "NEITHER"}

MODELS = {
    "openai": "gpt-5.2-2025-12-11",
    "gemini": "gemini-3-flash-preview",
    "anthropic": "claude-opus-4-5-20251101",
}

CONDITIONS = ["baseline", "evidence_required"]


SYSTEM_PROMPT_BASE = """You will be provided with the Title and Abstract of a randomized controlled oncology trial.
Your task is to determine whether the trial eligibility criteria permit patients with:
- LOCALIZED disease (non-metastatic)
- METASTATIC disease
- BOTH localized and metastatic disease
- or NEITHER (the abstract does not specify localized/metastatic eligibility).

Choose exactly one label: LOCALIZED, METASTATIC, BOTH, or NEITHER."""

SYSTEM_PROMPT_BASELINE = SYSTEM_PROMPT_BASE + "\n\nRespond with ONLY the label (all caps). Do not output anything else."

SYSTEM_PROMPT_EVIDENCE = (
    SYSTEM_PROMPT_BASE
    + """

Return ONLY a JSON object with these keys:
- "label": one of LOCALIZED, METASTATIC, BOTH, NEITHER
- "quote": an exact verbatim quote copied from the provided abstract that supports the label.

Rules:
- The quote MUST be a contiguous substring of the abstract (verbatim).
- If the label is NEITHER and there is no supporting text, set "quote" to "" (empty string).
- Do not wrap the JSON in markdown or add any extra text."""
)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()

def _quote_is_valid(quote: str, abstract: str, label: str) -> bool:
    # For NEITHER, an empty quote is allowed.
    if label == "NEITHER" and not quote.strip():
        return True

    q = _normalize_ws(quote)
    if not q:
        return False
    return q in _normalize_ws(abstract)


def init_openai():
    global openai_client
    if openai_client is None:
        from openai import OpenAI  # type: ignore

        openai_client = OpenAI()
    return openai_client


def init_gemini():
    global genai_client
    if genai_client is None:
        from google import genai as genai_module  # type: ignore

        genai_client = genai_module.Client(api_key=os.environ["GEMINI_API_KEY"])
    return genai_client


def init_anthropic():
    global anthropic_client
    if anthropic_client is None:
        import anthropic  # type: ignore

        anthropic_client = anthropic.Anthropic()
    return anthropic_client


def build_user_prompt(title: str, abstract: str, doi: str) -> str:
    parts = []
    if doi:
        parts.append(f"DOI: {doi}")
    parts.append(f"Title: {title}")
    parts.append(f"Abstract: {abstract}")
    return "\n\n".join(parts)


def parse_baseline_output(raw_output: str | None) -> tuple[str, str, bool]:
    if raw_output is None:
        return ("ERROR", "", False)

    text = raw_output.strip().upper()
    if text in VALID_LABELS:
        return (text, "", False)

    for label in ("LOCALIZED", "METASTATIC", "BOTH", "NEITHER"):
        if label in text:
            return (label, "", False)

    return ("INVALID", "", False)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```(?:json)?\\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\\s*```\\s*$", "", t)
    return t.strip()


def parse_evidence_output(raw_output: str | None, abstract: str) -> tuple[str, str, bool]:
    if raw_output is None:
        return ("ERROR", "", False)

    raw = _strip_code_fences(raw_output)
    start = raw.find("{")
    end = raw.rfind("}")

    if 0 <= start < end:
        candidate = raw[start : end + 1]
        try:
            obj = json.loads(candidate)
            label = str(obj.get("label", "")).strip().upper()
            quote = str(obj.get("quote", "")).strip()
            parsed_label = label if label in VALID_LABELS else "INVALID"
            return (parsed_label, quote, _quote_is_valid(quote, abstract, parsed_label))
        except Exception:
            pass

    label, _, _ = parse_baseline_output(raw_output)
    return (label, "", False)


def call_openai(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str | None:
    try:
        client = init_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else None
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        return None


def call_gemini(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str | None:
    try:
        from google.genai import types  # type: ignore

        client = init_gemini()
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=500,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"  Gemini API error: {e}", file=sys.stderr)
        return None


def call_anthropic(model: str, system_prompt: str, user_prompt: str, temperature: float) -> str | None:
    try:
        client = init_anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as e:
        print(f"  Anthropic API error: {e}", file=sys.stderr)
        return None


def call_api(vendor: str, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str | None:
    if vendor == "openai":
        return call_openai(model, system_prompt, user_prompt, temperature)
    if vendor == "gemini":
        return call_gemini(model, system_prompt, user_prompt, temperature)
    if vendor == "anthropic":
        return call_anthropic(model, system_prompt, user_prompt, temperature)
    raise ValueError(f"Unknown vendor: {vendor}")


def load_existing_predictions(predictions_path: Path) -> set[tuple[int, str, str, int]]:
    completed: set[tuple[int, str, str, int]] = set()
    if not predictions_path.exists():
        return completed

    with predictions_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(
                    (
                        int(row["trial_idx"]),
                        row["vendor"],
                        row["condition"],
                        int(row.get("run", "1")),
                    )
                )
            except Exception:
                continue
    return completed


def append_prediction(predictions_path: Path, row: dict[str, str]) -> None:
    write_header = not predictions_path.exists()
    with predictions_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_idx",
                "doi",
                "vendor",
                "model",
                "condition",
                "run",
                "ground_truth",
                "parsed_label",
                "quote",
                "quote_is_substring",
                "raw_output",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _sanitize_multiline(value: str) -> str:
    # Keep CSV to one-row-per-line (Git + tooling friendliness) while preserving content.
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM citations evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print planned calls without running APIs")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit number of trials to process")
    parser.add_argument("--vendor", type=str, choices=list(MODELS.keys()), default=None, help="Run a specific vendor")
    parser.add_argument("--condition", type=str, choices=CONDITIONS, default=None, help="Run a specific condition")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs per setting")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between calls (seconds)")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N calls (0 to disable)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if load_dotenv is not None:
        # Avoid python-dotenv's find_dotenv() heuristics (can be brittle in some environments).
        load_dotenv(dotenv_path=script_dir / ".env")

    trials_path = script_dir / "data" / "trials.csv"
    predictions_path = script_dir / "results" / "predictions.csv"
    predictions_path.parent.mkdir(exist_ok=True)

    with trials_path.open(newline="", encoding="utf-8", errors="replace") as f:
        trials = list(csv.DictReader(f))
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    vendors = [args.vendor] if args.vendor else list(MODELS.keys())
    conditions = [args.condition] if args.condition else CONDITIONS

    completed = load_existing_predictions(predictions_path)

    total_calls = len(trials) * len(vendors) * len(conditions) * max(args.runs, 1)
    completed_relevant = sum(
        1
        for (trial_idx, vendor, condition, run) in completed
        if vendor in vendors and condition in conditions and 1 <= run <= max(args.runs, 1) and 0 <= trial_idx < len(trials)
    )
    remaining_calls = total_calls - completed_relevant

    if args.dry_run:
        print(f"Trials: {len(trials)}")
        print(f"Vendors: {vendors}")
        print(f"Conditions: {conditions}")
        print(f"Runs: {args.runs}")
        print(f"Total API calls: {total_calls}")
        print(f"Already completed (resume): {completed_relevant}")
        print(f"Remaining: {remaining_calls}")
        return 0

    if "openai" in vendors and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)
    if "gemini" in vendors and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set", file=sys.stderr)
    if "anthropic" in vendors and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set", file=sys.stderr)

    call_idx = 0
    for condition in conditions:
        system_prompt = SYSTEM_PROMPT_BASELINE if condition == "baseline" else SYSTEM_PROMPT_EVIDENCE
        print(f"\n=== Condition: {condition} ===\n")

        for trial_row in trials:
            trial_idx = int(trial_row.get("trial_idx") or 0)
            doi = (trial_row.get("doi") or "").strip()
            title = (trial_row.get("title") or "").strip()
            abstract = (trial_row.get("abstract") or "").strip()
            ground_truth = (trial_row.get("ground_truth") or "").strip().upper()

            user_prompt = build_user_prompt(title, abstract, doi)

            for vendor in vendors:
                model = MODELS[vendor]
                for run in range(1, max(args.runs, 1) + 1):
                    key = (trial_idx, vendor, condition, run)
                    if key in completed:
                        continue

                    call_idx += 1
                    if args.log_every and (call_idx % args.log_every == 0 or call_idx == 1):
                        print(f"[{call_idx}/{remaining_calls}] Trial {trial_idx}, {vendor}, {condition}, run={run}")

                    raw_output = call_api(vendor, model, system_prompt, user_prompt, args.temperature)

                    if condition == "baseline":
                        parsed_label, quote, quote_ok = parse_baseline_output(raw_output)
                    else:
                        parsed_label, quote, quote_ok = parse_evidence_output(raw_output, abstract)

                    append_prediction(
                        predictions_path,
                        {
                            "trial_idx": str(trial_idx),
                            "doi": doi,
                            "vendor": vendor,
                            "model": model,
                            "condition": condition,
                            "run": str(run),
                            "ground_truth": ground_truth,
                            "parsed_label": parsed_label,
                            "quote": _sanitize_multiline(quote),
                            "quote_is_substring": "1" if quote_ok else "0",
                            "raw_output": _sanitize_multiline(raw_output or ""),
                        },
                    )

                    if parsed_label in ("INVALID", "ERROR"):
                        print(f"  -> {parsed_label}")
                    elif condition == "evidence_required":
                        if args.log_every and (call_idx % args.log_every == 0 or call_idx == 1):
                            print(f"  -> {parsed_label} (quote_ok={quote_ok})")
                    else:
                        if args.log_every and (call_idx % args.log_every == 0 or call_idx == 1):
                            print(f"  -> {parsed_label}")

                    if args.sleep > 0:
                        time.sleep(args.sleep)

    print(f"\nDone. Results appended to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
