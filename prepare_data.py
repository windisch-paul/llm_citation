#!/usr/bin/env python3
"""
Prepare `data/trials.csv` for the LLM citations experiment.

Input:  `data/metastatic_local.csv` (local raw artifact; ignored by git)
Output: `data/trials.csv` (cleaned; intended to be tracked)

The raw file encodes labels in the `accept` column as a list-like string, e.g.:
  - ['LOCAL']
  - ['METASTATIC']
  - ['LOCAL', 'METASTATIC']
  - []

These are mapped to:
  - LOCALIZED, METASTATIC, BOTH, NEITHER
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


LABELS = ("LOCALIZED", "METASTATIC", "BOTH", "NEITHER")


def _resolve_path(script_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (script_dir / p)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


def _parse_accept(cell: str | None) -> set[str]:
    if cell is None:
        return set()
    raw = str(cell).strip()
    if raw in ("", "[]"):
        return set()

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set)):
            return {str(x).strip().upper() for x in parsed if str(x).strip()}
    except Exception:
        pass

    # Fallback for mildly malformed list strings
    stripped = raw.strip().strip("[]")
    if not stripped:
        return set()
    parts = [p.strip().strip("'\"") for p in stripped.split(",") if p.strip()]
    return {p.upper() for p in parts if p}


def _label_from_accept(cell: str | None) -> str:
    tags = _parse_accept(cell)

    has_local = any(t in {"LOCAL", "LOCALIZED", "LOCALISED"} for t in tags)
    has_met = any(t in {"METASTATIC", "METASTASIS", "MET"} for t in tags)

    if has_local and has_met:
        return "BOTH"
    if has_local:
        return "LOCALIZED"
    if has_met:
        return "METASTATIC"
    return "NEITHER"


def _best_abstract(row: dict[str, str]) -> str:
    direct = (row.get("abstract") or "").strip()
    if direct:
        return _normalize_ws(direct)

    # Assemble structured abstracts if present
    parts = []
    for key in (
        "abstract_introduction",
        "abstract_methods",
        "abstract_results",
        "abstract_conclusions",
    ):
        val = (row.get(key) or "").strip()
        if val:
            parts.append(val)
    if parts:
        return _normalize_ws("\\n\\n".join(parts))

    fallback = (row.get("text") or "").strip()
    return _normalize_ws(fallback) if fallback else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare cleaned trials.csv for the llm_citations experiment")
    parser.add_argument("--input", default="data/metastatic_local.csv", help="Path to raw input CSV")
    parser.add_argument("--output", default="data/trials.csv", help="Path to write cleaned CSV")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional row limit (for debugging)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = _resolve_path(script_dir, args.input)
    output_path = _resolve_path(script_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if args.max_trials is not None:
        raw_rows = raw_rows[: args.max_trials]

    cleaned_rows: list[dict[str, str]] = []
    for idx, row in enumerate(raw_rows):
        cleaned_rows.append(
            {
                "trial_idx": str(idx),
                "doi": (row.get("doi") or "").strip(),
                "date": (row.get("date") or "").strip(),
                "title": _normalize_ws((row.get("title") or "").strip()),
                "abstract": _best_abstract(row),
                "ground_truth": _label_from_accept(row.get("accept")),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_idx", "doi", "date", "title", "abstract", "ground_truth"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Wrote {len(cleaned_rows)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
