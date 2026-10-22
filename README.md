# ai-search-visibility-engine

Portfolio project for analyzing AI search visibility: compare how OpenAI and Google Gemini answer the same CRM-buying questions, measure brand mentions, rankings, citations, and competitor presence, and eventually recommend actions to improve a brand's visibility in AI-generated answers. The first industry is B2B SaaS CRM, with HubSpot as the primary tracked brand.


## Generate the initial datasets

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python scripts/generate_initial_dataset.py
```

This writes four CSVs to `data/processed/` (brands, queries, collection manifest, synthetic recommendation candidates), validates them, and prints a summary. It runs offline and needs no API keys. Settings live in `configs/dataset.yaml`.

Verify that the committed files are up to date without writing anything:

```bash
.venv/bin/python scripts/generate_initial_dataset.py --check
```

## Collect AI answers

Needs `DATABASE_URL`, `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env` (see `.env.example`).

```bash
.venv/bin/python -m alembic upgrade head
.venv/bin/python scripts/import_dataset.py
.venv/bin/python scripts/collect_responses.py --mode pilot --dry-run   # preview, no API calls
.venv/bin/python scripts/collect_responses.py --mode pilot             # 3-5 queries per provider
.venv/bin/python scripts/collect_responses.py --mode full              # all remaining requests
```

Settings are in `configs/collection.yaml`. See [docs/collection_pipeline.md](docs/collection_pipeline.md) for retries, what is recorded, and SQL to verify results in Supabase.

See [docs/dataset_design.md](docs/dataset_design.md) for what each file means and which data is authored, planned, synthetic, or collected.

## Extract brand mentions and citations

Reads the stored answers (Gemini answers are never sent to Gemini again) and writes `observation_extractions`, `brand_mentions` and `citations`, all tagged with `extractor_version`. Needs `DATABASE_URL` and `OPENAI_API_KEY`.

1. Rules first, in Python: brand aliases (whole words, longest match, exclusions such as "Zoho Desk"), citations from the provider's metadata (OpenAI `url_citations`, Gemini grounding chunks), and URLs written in the answer, stored separately as `response_text`.
2. Then one GPT-5.6 Luna call per answer that has something to classify: sentiment and recommendation per brand, and whether ambiguous words ("Close", "Monday", "Freshworks") are the brand. The output must match `schemas/brand_classification.schema.json`, and every quote must appear in the answer, or it is retried and finally marked `failed`.


Check the pilot by hand before the full run, e.g. in Supabase:

```sql
select m.observation_id, m.brand_id, m.sentiment, m.recommendation, m.classification_evidence
from brand_mentions m
where m.extractor_version = 'rules-llm-v3'
order by m.observation_id, m.first_mention_rank;
```

Settings are in `configs/extraction.yaml`:

- `extractor_version`: bump it whenever rules, prompt or schema change. Each version's results are kept side by side; rerunning the same version replaces its rows and never duplicates them.
- `strategy`: `rules_only` or `rules_and_llm` per source provider.
- `llm`: model (override with `EXTRACTION_MODEL`), reasoning effort, `prompt_version` (prompts are kept in `visibility_extractor/llm.py`), concurrency. The run stops if the model is not available; it never switches to another one.
- `retry`: attempts per answer; rate limits and timeouts back off, honouring `Retry-After`.
- `pricing`: USD per 1M tokens, used for `classification_estimated_cost_usd`. Taken from third-party listings; check them on OpenAI's pricing page.

### Data quality (rules-llm-v3)

From the 16-answer pilot, checked by hand, and the full run:

- Ambiguous names: all 30 pilot decisions were plausible; the LLM accepted 96% of ambiguous occurrences overall. "monday.com" used for the platform is counted as monday CRM.
- Sentiment agrees across runs on 98–100% of mentions.
- Recommendation: between pilot runs and prompt versions, "recommended or not" stayed the same for 93–100% of mentions, the exact `primary` / `conditional` / `none` label for only 84–96%. Prefer "recommended or not" (`recommendation <> 'none'`, `target_brand_recommended`) for analysis.
- 4 of 458 answers failed: the model turned table rows into headings, joined table cells, gave an empty quote, or quoted more than 300 characters. They have no mentions or citations stored.
- Gemini returns only a domain per citation, not a URL path, so citations on shared domains (zoho.com, freshworks.com, monday.com) are left without a brand. Compare per-brand citation counts between providers with that in mind.
- `classification_estimated_cost_usd` depends on unverified prices.

## Tests

```bash
.venv/bin/python -m pytest                          # mocked: no API calls, no database
RUN_DB_TESTS=1 .venv/bin/python -m pytest -m db     # real database, every test rolled back
```

There are no automated live API tests: `scripts/check_extraction_model.py --probe` makes one real call on demand.
