# LLM Citations

Evaluates whether requiring verbatim evidence (quotes) improves LLM trustworthiness for eligibility classification of oncology RCT abstracts.

## Task
Classify whether trial eligibility permits: `LOCALIZED`, `METASTATIC`, `BOTH`, or `NEITHER`.

## Experimental conditions
- `baseline`: output only a classification label
- `evidence_required`: output label + a verbatim quote from the abstract

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Quick smoke test (no API calls)
```bash
python run_evaluation.py --dry-run --max-trials 3
```

## Notes
- `data/metastatic_local.csv` is treated as a local raw artifact (ignored by git).
- `data/trials.csv` is the cleaned dataset used by the scripts.
