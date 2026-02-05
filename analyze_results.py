#!/usr/bin/env python3
"""
Analyze llm_citations predictions.

Expected input:
  - results/predictions.csv (from run_evaluation.py)

Writes:
  - results/metrics.csv
  - results/table_comparison.csv
  - results/table_f1_valid_quote_only.csv
  - results/table_fleiss_kappa.csv
  - results/table_fleiss_kappa_citation.csv
  - results/table_switching.csv
  - results/table_transition_matrix_openai.csv
  - results/table_transition_matrix_gemini.csv
  - results/table_transition_matrix_anthropic.csv
  - results/table_manuscript.csv
  - results/table_manuscript.txt
  - results/table_paper.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


VALID_LABELS = ("LOCALIZED", "METASTATIC", "BOTH", "NEITHER")
CONDITIONS = ("baseline", "evidence_required")


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return float("nan")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    if not y_true:
        return float("nan")
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def fleiss_kappa(items: list[list[str]], categories: tuple[str, ...]) -> float:
    """
    Compute Fleiss' kappa for N items, n raters, k categories.
    items: list of length N, each is list of length n with category labels.
    """
    if not items:
        return float("nan")
    n = len(items[0])
    if n < 2:
        return float("nan")

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    N = len(items)
    k = len(categories)

    counts = [[0] * k for _ in range(N)]
    for i, labels in enumerate(items):
        if len(labels) != n:
            raise ValueError("All items must have the same number of ratings")
        for lab in labels:
            if lab not in cat_to_idx:
                raise ValueError(f"Unknown category: {lab}")
            counts[i][cat_to_idx[lab]] += 1

    P_i = []
    for i in range(N):
        s = sum(c * (c - 1) for c in counts[i])
        P_i.append(s / (n * (n - 1)))
    P_bar = sum(P_i) / N

    p_j = [0.0] * k
    for j in range(k):
        p_j[j] = sum(counts[i][j] for i in range(N)) / (N * n)
    P_e = sum(p * p for p in p_j)

    denom = 1.0 - P_e
    if denom == 0:
        return float("nan")
    return (P_bar - P_e) / denom


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_empty(value: float) -> str:
    return f"{value:.6f}" if not math.isnan(value) else ""


def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100.0) if d else 0.0:.2f}"


def analyze(predictions_path: Path, output_dir: Path) -> None:
    with predictions_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"No rows found in {predictions_path}")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[(r.get("vendor", ""), r.get("model", ""), r.get("condition", ""))].append(r)

    output_dir.mkdir(exist_ok=True, parents=True)

    metrics_out: list[dict[str, str]] = []
    for (vendor, model, condition), g in sorted(grouped.items()):
        n_total = len(g)
        parsed = [r.get("parsed_label", "").strip().upper() for r in g]
        truth = [r.get("ground_truth", "").strip().upper() for r in g]

        is_valid = [p in VALID_LABELS and t in VALID_LABELS for p, t in zip(parsed, truth)]
        y_true = [t for ok, t in zip(is_valid, truth) if ok]
        y_pred = [p for ok, p in zip(is_valid, parsed) if ok]

        n_valid = len(y_true)
        n_invalid = sum(1 for p in parsed if p == "INVALID")
        n_error = sum(1 for p in parsed if p == "ERROR")
        pct_valid = (n_valid / n_total * 100.0) if n_total else 0.0

        acc = accuracy(y_true, y_pred)
        f1 = macro_f1(y_true, y_pred, VALID_LABELS)

        quote_present = [bool((r.get("quote") or "").strip()) for r in g]
        quote_ok = [_as_bool(r.get("quote_is_substring")) for r in g]
        pct_quote_present = (sum(quote_present) / n_total * 100.0) if n_total else 0.0
        pct_quote_ok = (sum(quote_ok) / n_total * 100.0) if n_total else 0.0

        metrics_out.append(
            {
                "vendor": vendor,
                "model": model,
                "condition": condition,
                "accuracy_valid_only": _float_or_empty(acc),
                "f1_macro_valid_only": _float_or_empty(f1),
                "n_total": str(n_total),
                "n_valid": str(n_valid),
                "n_invalid": str(n_invalid),
                "n_error": str(n_error),
                "pct_valid": f"{pct_valid:.2f}",
                "pct_quote_present": f"{pct_quote_present:.2f}",
                "pct_quote_is_substring": f"{pct_quote_ok:.2f}",
            }
        )

    metrics_path = output_dir / "metrics.csv"
    write_csv(metrics_path, list(metrics_out[0].keys()), metrics_out)
    print(f"Wrote: {metrics_path}")

    by_vendor_condition = {(r["vendor"], r["condition"]): r for r in metrics_out}
    vendors = sorted({r["vendor"] for r in metrics_out})
    conditions = list(CONDITIONS)

    comparison_path = output_dir / "table_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["vendor"] + [f"{c}_f1" for c in conditions] + [f"{c}_acc" for c in conditions]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in vendors:
            row = {"vendor": v}
            for c in conditions:
                m = by_vendor_condition.get((v, c), {})
                row[f"{c}_f1"] = m.get("f1_macro_valid_only", "")
                row[f"{c}_acc"] = m.get("accuracy_valid_only", "")
            writer.writerow(row)
    print(f"Wrote: {comparison_path}")

    # F1 restricted to (valid labels) AND (valid quote) for evidence_required only.
    f1_quote_rows = []
    for vendor in vendors:
        evidence_rows = [
            r
            for r in rows
            if r.get("vendor") == vendor and r.get("condition") == "evidence_required" and _as_bool(r.get("quote_is_substring"))
        ]
        parsed = [(r.get("parsed_label") or "").strip().upper() for r in evidence_rows]
        truth = [(r.get("ground_truth") or "").strip().upper() for r in evidence_rows]

        is_valid = [p in VALID_LABELS and t in VALID_LABELS for p, t in zip(parsed, truth)]
        y_true = [t for ok, t in zip(is_valid, truth) if ok]
        y_pred = [p for ok, p in zip(is_valid, parsed) if ok]

        acc_q = accuracy(y_true, y_pred)
        f1_q = macro_f1(y_true, y_pred, VALID_LABELS)

        f1_quote_rows.append(
            {
                "vendor": vendor,
                "condition": "evidence_required",
                "accuracy_valid_quote_only": _float_or_empty(acc_q),
                "f1_macro_valid_quote_only": _float_or_empty(f1_q),
                "n_total_quote_ok": str(len(evidence_rows)),
                "n_valid_quote_ok": str(len(y_true)),
                "pct_valid_quote_ok": _pct(len(y_true), len(evidence_rows)),
            }
        )

    f1_quote_path = output_dir / "table_f1_valid_quote_only.csv"
    write_csv(f1_quote_path, list(f1_quote_rows[0].keys()), f1_quote_rows)
    print(f"Wrote: {f1_quote_path}")

    # Fleiss' kappa across repeated runs (valid labels only)
    ratings: dict[tuple[str, str], dict[int, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    ratings_quote_ok: dict[tuple[str, str], dict[int, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))

    for r in rows:
        vendor = r.get("vendor", "")
        condition = r.get("condition", "")
        try:
            trial_idx = int(r.get("trial_idx", "0"))
            run = int(r.get("run", "1"))
        except Exception:
            continue

        lab = (r.get("parsed_label") or "").strip().upper()
        if lab not in VALID_LABELS:
            continue

        ratings[(vendor, condition)][trial_idx][run] = lab

        quote_ok = _as_bool(r.get("quote_is_substring"))
        if quote_ok:
            ratings_quote_ok[(vendor, condition)][trial_idx][run] = lab

    kappa_rows = []
    kappa_cite_rows = []

    for (vendor, condition), per_trial in sorted(ratings.items()):
        runs_set = {run for t in per_trial.values() for run in t.keys()}
        n_runs = max(runs_set) if runs_set else 0
        if n_runs < 2:
            continue

        items = []
        for per_run in per_trial.values():
            if len(per_run) != n_runs:
                continue
            items.append([per_run[r] for r in range(1, n_runs + 1)])

        kappa = fleiss_kappa(items, VALID_LABELS) if items else float("nan")
        kappa_rows.append(
            {
                "vendor": vendor,
                "condition": condition,
                "n_runs": str(n_runs),
                "n_items": str(len(items)),
                "fleiss_kappa": _float_or_empty(kappa),
            }
        )

    for (vendor, condition), per_trial in sorted(ratings_quote_ok.items()):
        if condition != "evidence_required":
            continue
        runs_set = {run for t in per_trial.values() for run in t.keys()}
        n_runs = max(runs_set) if runs_set else 0
        if n_runs < 2:
            continue

        items = []
        for per_run in per_trial.values():
            if len(per_run) != n_runs:
                continue
            items.append([per_run[r] for r in range(1, n_runs + 1)])

        kappa = fleiss_kappa(items, VALID_LABELS) if items else float("nan")
        kappa_cite_rows.append(
            {
                "vendor": vendor,
                "condition": condition,
                "n_runs": str(n_runs),
                "n_items": str(len(items)),
                "fleiss_kappa_valid_quote_only": _float_or_empty(kappa),
            }
        )

    if kappa_rows:
        kappa_path = output_dir / "table_fleiss_kappa.csv"
        write_csv(kappa_path, list(kappa_rows[0].keys()), kappa_rows)
        print(f"Wrote: {kappa_path}")

    if kappa_cite_rows:
        kappa_cite_path = output_dir / "table_fleiss_kappa_citation.csv"
        write_csv(kappa_cite_path, list(kappa_cite_rows[0].keys()), kappa_cite_rows)
        print(f"Wrote: {kappa_cite_path}")

    kappa_by_vendor_condition = {(r["vendor"], r["condition"]): r for r in kappa_rows}
    kappa_cite_by_vendor = {r["vendor"]: r for r in kappa_cite_rows}

    # Switching analysis + transition matrices (baseline -> evidence_required), per vendor.
    prediction_by_key: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for r in rows:
        vendor = (r.get("vendor") or "").strip()
        condition = (r.get("condition") or "").strip()
        if condition not in CONDITIONS:
            continue
        try:
            trial_idx = int(r.get("trial_idx") or "0")
            run = int(r.get("run") or "1")
        except Exception:
            continue
        prediction_by_key[(vendor, trial_idx, run, condition)] = r

    transition_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    switching_rows: list[dict[str, str]] = []

    for vendor in vendors:
        n_pairs = 0
        n_switched = 0
        n_pairs_quote_ok = 0
        n_switched_quote_ok = 0

        correct_to_incorrect = 0
        incorrect_to_correct = 0
        correct_to_correct = 0
        incorrect_to_incorrect = 0

        quote_ok_correct = 0
        quote_ok_incorrect = 0

        # Iterate only over keys that exist for baseline; look up evidence_required.
        vendor_keys = [k for k in prediction_by_key.keys() if k[0] == vendor and k[3] == "baseline"]
        for _, trial_idx, run, _ in vendor_keys:
            baseline_row = prediction_by_key.get((vendor, trial_idx, run, "baseline"))
            evidence_row = prediction_by_key.get((vendor, trial_idx, run, "evidence_required"))
            if not baseline_row or not evidence_row:
                continue

            baseline_label = (baseline_row.get("parsed_label") or "").strip().upper()
            evidence_label = (evidence_row.get("parsed_label") or "").strip().upper()
            truth = (baseline_row.get("ground_truth") or "").strip().upper()

            if baseline_label not in VALID_LABELS or evidence_label not in VALID_LABELS or truth not in VALID_LABELS:
                continue

            n_pairs += 1
            transition_counts[vendor][baseline_label][evidence_label] += 1

            switched = baseline_label != evidence_label
            if switched:
                n_switched += 1

            baseline_correct = baseline_label == truth
            evidence_correct = evidence_label == truth

            if baseline_correct and evidence_correct:
                correct_to_correct += 1
            elif baseline_correct and not evidence_correct:
                correct_to_incorrect += 1
            elif not baseline_correct and evidence_correct:
                incorrect_to_correct += 1
            else:
                incorrect_to_incorrect += 1

            quote_ok = _as_bool(evidence_row.get("quote_is_substring"))
            if quote_ok:
                n_pairs_quote_ok += 1
                if switched:
                    n_switched_quote_ok += 1
                if evidence_correct:
                    quote_ok_correct += 1
                else:
                    quote_ok_incorrect += 1

        switching_rows.append(
            {
                "vendor": vendor,
                "n_pairs": str(n_pairs),
                "n_switched": str(n_switched),
                "pct_switched": _pct(n_switched, n_pairs),
                "n_pairs_quote_ok": str(n_pairs_quote_ok),
                "n_switched_quote_ok": str(n_switched_quote_ok),
                "pct_switched_quote_ok": _pct(n_switched_quote_ok, n_pairs_quote_ok),
                "correct_to_correct": str(correct_to_correct),
                "correct_to_incorrect": str(correct_to_incorrect),
                "incorrect_to_correct": str(incorrect_to_correct),
                "incorrect_to_incorrect": str(incorrect_to_incorrect),
                "quote_ok_correct": str(quote_ok_correct),
                "quote_ok_incorrect": str(quote_ok_incorrect),
            }
        )

        # Transition matrix per vendor
        matrix_path = output_dir / f"table_transition_matrix_{vendor}.csv"
        labels = list(VALID_LABELS)
        with matrix_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["from\\to"] + labels)
            for from_label in labels:
                writer.writerow([from_label] + [transition_counts[vendor][from_label].get(to_label, 0) for to_label in labels])
        print(f"Wrote: {matrix_path}")

    switching_path = output_dir / "table_switching.csv"
    write_csv(switching_path, list(switching_rows[0].keys()), switching_rows)
    print(f"Wrote: {switching_path}")

    # Manuscript/paper summary tables
    f1_quote_by_vendor = {r["vendor"]: r for r in f1_quote_rows}
    switching_by_vendor = {r["vendor"]: r for r in switching_rows}

    summary_rows: list[dict[str, str]] = []
    for vendor in vendors:
        baseline = by_vendor_condition.get((vendor, "baseline"), {})
        evidence = by_vendor_condition.get((vendor, "evidence_required"), {})
        f1_quote = f1_quote_by_vendor.get(vendor, {})
        switching = switching_by_vendor.get(vendor, {})
        kappa_base = kappa_by_vendor_condition.get((vendor, "baseline"), {})
        kappa_evid = kappa_by_vendor_condition.get((vendor, "evidence_required"), {})
        kappa_quote = kappa_cite_by_vendor.get(vendor, {})

        base_f1 = float(baseline.get("f1_macro_valid_only") or "nan")
        evid_f1 = float(evidence.get("f1_macro_valid_only") or "nan")
        base_acc = float(baseline.get("accuracy_valid_only") or "nan")
        evid_acc = float(evidence.get("accuracy_valid_only") or "nan")

        summary_rows.append(
            {
                "vendor": vendor,
                "model": evidence.get("model") or baseline.get("model") or "",
                "baseline_f1": baseline.get("f1_macro_valid_only", ""),
                "evidence_f1": evidence.get("f1_macro_valid_only", ""),
                "delta_f1": _float_or_empty(evid_f1 - base_f1) if not (math.isnan(base_f1) or math.isnan(evid_f1)) else "",
                "baseline_acc": baseline.get("accuracy_valid_only", ""),
                "evidence_acc": evidence.get("accuracy_valid_only", ""),
                "delta_acc": _float_or_empty(evid_acc - base_acc) if not (math.isnan(base_acc) or math.isnan(evid_acc)) else "",
                "evidence_pct_quote_present": evidence.get("pct_quote_present", ""),
                "evidence_pct_quote_ok": evidence.get("pct_quote_is_substring", ""),
                "evidence_f1_valid_quote_only": f1_quote.get("f1_macro_valid_quote_only", ""),
                "pct_switched": switching.get("pct_switched", ""),
                "fleiss_kappa_baseline": kappa_base.get("fleiss_kappa", ""),
                "fleiss_kappa_evidence": kappa_evid.get("fleiss_kappa", ""),
                "fleiss_kappa_evidence_valid_quote_only": kappa_quote.get("fleiss_kappa_valid_quote_only", ""),
            }
        )

    manuscript_csv = output_dir / "table_manuscript.csv"
    write_csv(manuscript_csv, list(summary_rows[0].keys()), summary_rows)
    print(f"Wrote: {manuscript_csv}")

    paper_csv = output_dir / "table_paper.csv"
    write_csv(paper_csv, list(summary_rows[0].keys()), summary_rows)
    print(f"Wrote: {paper_csv}")

    manuscript_txt = output_dir / "table_manuscript.txt"
    # Keep a compact subset for the plain-text table.
    txt_cols = [
        "vendor",
        "baseline_f1",
        "evidence_f1",
        "delta_f1",
        "baseline_acc",
        "evidence_acc",
        "delta_acc",
        "evidence_pct_quote_ok",
        "pct_switched",
    ]
    col_widths = {c: len(c) for c in txt_cols}
    for r in summary_rows:
        for c in txt_cols:
            col_widths[c] = max(col_widths[c], len(str(r.get(c, ""))))

    def fmt_row(r: dict[str, str]) -> str:
        return "  ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in txt_cols)

    with manuscript_txt.open("w", encoding="utf-8") as f:
        f.write(fmt_row({c: c for c in txt_cols}) + "\n")
        f.write(fmt_row({c: "-" * col_widths[c] for c in txt_cols}) + "\n")
        for r in summary_rows:
            f.write(fmt_row(r) + "\n")
    print(f"Wrote: {manuscript_txt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze llm_citations predictions")
    parser.add_argument("--predictions", default="results/predictions.csv", help="Predictions CSV path")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = script_dir / predictions_path

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = script_dir / outdir

    analyze(predictions_path, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
