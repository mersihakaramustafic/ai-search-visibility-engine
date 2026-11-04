# ai-search-visibility-engine

Portfolio project for analyzing AI search visibility. It asks OpenAI and Google Gemini the same CRM-buying questions, measures brand mentions, recommendations and citations, finds where a brand is less visible than its competitors, and turns those gaps into evidence-backed, prioritized recommendations. The first industry is B2B SaaS CRM, with HubSpot as the tracked brand.

> **Status: paused (2026-11-20).** Stages 1–4 of the pipeline work end to end. Stage 5 (recommendations) is about half done. See [Future plan](#future-plan).

## Pipeline

| Stage | What it does | Code | Status |
|---|---|---|---|
| 1. Dataset | 100 authored queries over 5 intents, brands, collection manifest | `scripts/generate_initial_dataset.py` | done |
| 2. Collection | Answers from OpenAI and Gemini, stored in Supabase Postgres | `visibility_collector/`, `scripts/collect_responses.py` | done |
| 3. Extraction | Brand mentions and citations, with rules first and then an LLM for sentiment, recommendation and ambiguous names | `visibility_extractor/`, `scripts/extract_responses.py` | done (`rules-llm-v3`) |
| 4. Gap analysis | Mention, recommendation and citation rates, plus visibility gaps against competitors and between providers | `visibility_analysis/`, `scripts/visibility_gaps.py` | done |
| 5. Recommendations | Evidence packs → Gemini recommendations → validation → priority score | `visibility_recommender/` | partly done |

Details: [dataset](docs/dataset_design.md) · [collection](docs/collection_pipeline.md) · [extraction](docs/extract_brands.md) · [gaps](docs/visibility_gaps.md)

## Setup

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env    # DATABASE_URL, OPENAI_API_KEY, GEMINI_API_KEY
```

The Supabase project may be paused after inactivity. If you see `tenant/user ... not found`, restore it in the Supabase dashboard.

## Run

```bash
.venv/bin/python scripts/generate_initial_dataset.py           # offline, writes data/processed/
.venv/bin/python -m alembic upgrade 0006_classification_output # not "head": 0007 has not been tested yet (see below)
.venv/bin/python scripts/import_dataset.py
.venv/bin/python scripts/collect_responses.py --mode pilot     # add --dry-run first; --mode full for everything
.venv/bin/python scripts/extract_responses.py                  # see docs/extract_brands.md for flags
.venv/bin/python scripts/visibility_gaps.py                    # writes reports/visibility_gaps_<target>_<version>.{json,md}
.venv/bin/python scripts/check_recommendation_model.py --probe # one paid Gemini call (~$0.003)
```

Each stage has its settings in `configs/*.yaml`. Every LLM step records a version label (`extractor_version`, `generator_version`, `scoring_version`), so results from different versions are kept side by side instead of overwritten.

## Tests

```bash
.venv/bin/python -m pytest                          # mocked: no API calls, no database
RUN_DB_TESTS=1 .venv/bin/python -m pytest -m db     # real database, every test rolled back
```

## Known data caveats

- Use "recommended or not" rather than the exact `primary` / `conditional` label: the exact label is stable for only 84–96% of mentions.
- Gemini citations have only a domain, not a full URL, so citations to domains several brands share (zoho.com, freshworks.com, monday.com) aren't attributed to any brand.
- Cost figures come from list prices checked on the dates noted in the configs.

## Future plan

What Stage 5 (`visibility_recommender/`) has so far:

- **Done:**
  - evidence packs from gaps (`evidence.py`)
  - Pydantic output, with validation against the evidence pack (`output.py`)
  - Gemini client and runner with retries (`llm.py`, `runner.py`), using `gemini-3.8-flash`, `recs-v2`
  - deterministic priority score `p1`: impact 35 / evidence 30 / relevance 25 / effort 10, all weights in `configs/recommendation.yaml` (`scoring.py`)
- **Written but not tested:** migration `0007_recommendations` (generations, LLM candidates, scores, expert feedback, implementations). Test it against the database in a rolled-back transaction before running `alembic upgrade head`.

Still to do:

1. **Persistence:** a `RecommendationRepository` that saves generations, candidates and scores, and refuses a second generation for the same opportunity and `generator_version`. Plus database tests, using the same pattern as `tests/test_extraction_repository.py`.
2. **Run script:** `scripts/generate_recommendations.py` loads the gaps, builds evidence packs, calls Gemini, scores and saves the results, and writes a ranked report (JSON and Markdown). Do a dry run and a small pilot before running on all the HubSpot gaps.
3. **Review and feedback:** rate the recommendations in `recommendation_feedback` (useful, optional 1–5 rating, reason codes, the rank you saw them at). Record what was carried out in `recommendation_implementations`.
4. **ML ranking (`scoring_version: m1`):** learn a score from the expert ratings, using the stored score features.
   - Baseline: the `p1` formula.
   - Models: start with logistic regression, then try gradient-boosted trees or LambdaRank.
   - Evaluate with NDCG@k and precision@k on held-out opportunities, and use the model only if it beats `p1`.
   - Data needed: one run gives about 50–120 recommendations, so expect several runs or brands before you have the few hundred ratings this needs.
5. **Outcome labels (later):** collect answers again some time after a recommendation was implemented, and compare visibility before and after, as a stronger label than expert opinion.

Pricing note: Gemini 3.8 Flash costs $0.75 / $3.75 per 1M input / output tokens until 2026-12-31, and **twice that from 2027-01-01**. Update `pricing` in `configs/recommendation.yaml` before any run in 2027.
