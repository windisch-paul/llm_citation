# LLM Citations

Pipeline for the manuscript:

**Show Your Work: Verbatim Evidence Requirements and Automated Assessment for Large Language Models in Biomedical Text Processing**

The project evaluates whether requiring an exact verbatim quote from the abstract changes LLM behavior on oncology trial eligibility-scope classification.

## Task

Given **title + abstract**, predict exactly one label:

- `LOCALIZED`
- `METASTATIC`
- `BOTH`
- `NEITHER`
- `UNCLEAR`

Two conditions are evaluated:

- `baseline`: label only
- `evidence_required`: label + supporting quote (mechanically-valid only when it is an exact substring after whitespace normalization); quote is required for non-`UNCLEAR` labels and optional for `UNCLEAR`

Each trial is repeated (`--repeats`, default `3`) for stability analysis.

## Models

- OpenAI: `gpt-5.2-2025-12-11`
- Gemini: `gemini-3-flash-preview`
- Anthropic: `claude-opus-4-5-20251101`

## Structure

- `run_evaluation.py`: API evaluation loop (resume-safe, repeat-aware)
- `analyze_results.py`: computes manuscript metrics/tables
- `visualizations.py`: creates manuscript-style figures
- `run_evidence_judgment.py`: second-stage semantic grounding audit
- `analyze_evidence_judgments.py`: summarizes second-stage semantic grounding rates
- `data/trials.csv`: source dataset (200 trials)
- `results/`: predictions and analysis tables
- `plots/`: generated figures

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set API keys in `.env`:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

## Usage

### 1. Cost-safe dry run

```bash
python run_evaluation.py --dry-run
```

### 2. Small smoke run (recommended first)

```bash
python run_evaluation.py --max-trials 5 --repeats 1
```

### 3. Full experiment

```bash
python run_evaluation.py --repeats 3
```

### 4. Analyze

```bash
python analyze_results.py
```

### 5. Plot

```bash
python visualizations.py
```

## Second-stage semantic grounding judgment

This stage asks judge models whether a source model's quote semantically
grounds that source model's assigned label.

Judge input is intentionally minimal:
- only `ASSIGNED LABEL` and `QUOTE` are sent to the judge model
- no abstract/title/source-model metadata is sent
- source rows with `UNCLEAR` label or empty quote are excluded automatically
- judge response is verdict-only: `VERDICT: SUPPORTED|NOT_SUPPORTED`

Default behavior in stage 2:
- all available source repeats are selected (repeat-aware first-stage rows)
- each selected source row is judged once per judge vendor (no stage-2 judge repeats)
- source vendors default to all available models (`--source-vendors all`)

### 1. Preview prompts and planned call count (recommended first)

```bash
python run_evidence_judgment.py \
  --dry-run \
  --print-prompts \
  --source-vendors all \
  --judges openai,gemini,anthropic
```

### 2. Small smoke run

```bash
python run_evidence_judgment.py \
  --source-vendors all \
  --max-trials 5 \
  --judges openai,gemini,anthropic
```

### 3. Full second-stage run (all source models, all source repeats, no stage-2 repeats)

```bash
python run_evidence_judgment.py \
  --source-vendors all \
  --judges openai,gemini,anthropic
```

### 4. Optional stricter source filter

Use only first-stage rows that were mechanically-valid:

```bash
python run_evidence_judgment.py \
  --source-vendors all \
  --judges openai,gemini,anthropic \
  --require-mechanical-valid
```

### 5. Optional single-repeat mode (legacy behavior)

Use only one source repeat:

```bash
python run_evidence_judgment.py \
  --source-vendors all \
  --single-source-repeat \
  --source-repeat 1 \
  --judges openai,gemini,anthropic
```

### 6. Analyze second-stage judgments

```bash
python analyze_evidence_judgments.py
```

This analysis now also reports:
- `macro_f1_judge_valid_only`: macro F1 of source labels vs ground truth after
  filtering to rows where the judge returned a format-valid `SUPPORTED` verdict (semantically grounded).
- `results/table_judge_grounding.csv`: table-ready comparison of baseline
  coverage/F1 vs semantically grounded coverage/F1 by source model and judge model.

## Key outputs

- `results/predictions.csv`
- `results/metrics.csv`
- `results/table_manuscript.csv`
- `results/table_manuscript_raw.csv`
- `results/table_evidence_breakdown.csv`
- `results/table_confusion_long.csv`
- `results/table_mcnemar.csv`
- `results/table_fleiss_drivers.csv`
- `results/table_fleiss_drivers_raw.csv`
- `results/evidence_judgments.csv`
- `results/evidence_judgment_summary.csv`
- `results/evidence_judgment_by_label.csv`
- `results/evidence_judgment_f1_valid_only.csv`
- `results/table_judge_grounding.csv`
- `results/table_judge_grounding_raw.csv`
- `plots/figure_1_summary.png`
- `plots/figure_2_confusion_matrices.png`

## Notes

- Full run can be expensive (200 trials × 3 vendors × 2 conditions × 3 repeats = 3600 calls).
- `predictions.csv` is append-only and resume-aware; use `--no-resume` to force rerun.
- `evidence_judgments.csv` is also append-only and resume-aware; use `--no-resume` to force rerun.
- No secrets are printed by the scripts.
