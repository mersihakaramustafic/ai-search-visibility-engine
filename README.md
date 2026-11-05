# ai-search-visibility-engine

Measures how visible a brand is in AI-generated answers. The same 100 CRM-buying queries are sent to OpenAI and Gemini, brand mentions and citations are extracted from the answers, and the target brand (HubSpot) is compared with its competitors to find visibility gaps. The last stage turns those gaps into prioritized recommendations.

Status: paused. Stages 1–4 are complete; stage 5 is partly implemented.

## Stages

| Stage | Package / script | Output |
|---|---|---|
| 1. Dataset | `scripts/generate_initial_dataset.py` | `data/processed/*.csv` |
| 2. Collection | `visibility_collector/`, `scripts/collect_responses.py` | `observations` |
| 3. Extraction | `visibility_extractor/`, `scripts/extract_responses.py` | `brand_mentions`, `citations` |
| 4. Gap analysis | `visibility_analysis/`, `scripts/visibility_gaps.py` | `reports/visibility_gaps_*.{json,md}` |
| 5. Recommendations | `visibility_recommender/` | not yet persisted |

Each stage is configured in `configs/<stage>.yaml`. Design notes are in `docs/`.

## Setup

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env   # DATABASE_URL, OPENAI_API_KEY, GEMINI_API_KEY
.venv/bin/python -m alembic upgrade 0006_classification_output
```

Upgrade only to 0006: migration 0007 has not been tested yet.

## Usage

```bash
.venv/bin/python scripts/generate_initial_dataset.py
.venv/bin/python scripts/import_dataset.py
.venv/bin/python scripts/collect_responses.py --mode pilot   # --dry-run to preview, --mode full for all
.venv/bin/python scripts/extract_responses.py
.venv/bin/python scripts/visibility_gaps.py --target hubspot
.venv/bin/python scripts/check_recommendation_model.py --probe   # one paid Gemini call
```

## Tests

```bash
.venv/bin/python -m pytest                        # no API calls, no database
RUN_DB_TESTS=1 .venv/bin/python -m pytest -m db   # database, rolled back
```

## Data caveats

- `primary` vs `conditional` recommendation labels are unstable between runs; use `recommendation <> 'none'`.
- Gemini citations contain only a domain, so citations to shared domains (zoho.com, freshworks.com, monday.com) have no brand.

## Remaining work

Stage 5 currently has:
- evidence packs (`evidence.py`)
- output validation (`output.py`)
- a Gemini generator with retries (`llm.py`, `runner.py`)
- the `p1` priority score (`scoring.py`)

Migration `0007_recommendations` defines the generation, score, feedback and implementation tables but has not been run.

1. Test migration 0007 in a rolled-back transaction, then apply it.
2. Add `RecommendationRepository` to save generations, candidates and scores, with database tests.
3. Add `scripts/generate_recommendations.py`: gaps → evidence packs → Gemini → score → save, plus a ranked report.
4. Review the generated recommendations; record ratings in `recommendation_feedback` and what was carried out in `recommendation_implementations`.
5. Train a ranking model (`scoring_version: m1`) on the stored score features and ratings. Compare it with `p1` using NDCG@k on held-out opportunities. This needs a few hundred ratings; one run yields about 50–120 recommendations.
6. Optionally, measure visibility again after a recommendation is implemented and use the change as a label.

Gemini 3.8 Flash prices double on 2027-01-01; update `pricing` in `configs/recommendation.yaml` before running.
