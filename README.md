# ai-search-visibility-engine

Portfolio project for analyzing AI search visibility: compare how OpenAI and Google Gemini answer the same CRM-buying questions, measure brand mentions, rankings, citations, and competitor presence, and eventually recommend actions to improve a brand's visibility in AI-generated answers. The first industry is B2B SaaS CRM, with HubSpot as the primary tracked brand.

## Status

Stage 1 (done): authored brand and query datasets, the collection plan, and synthetic recommendation seeds. No API collection, models, or dashboard yet.

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

Run the tests:

```bash
.venv/bin/python -m pytest
```

## Layout

```
configs/dataset.yaml            counts, providers, repetitions, intent distribution, quality guards
data/authored/*.yaml            hand-written brands, queries, query families, synthetic candidates
data/processed/*.csv            generated outputs (do not edit by hand)
scripts/generate_initial_dataset.py
tests/test_initial_dataset.py
docs/dataset_design.md          column definitions, design rationale, limitations
```

See [docs/dataset_design.md](docs/dataset_design.md) for what each file means and which data is authored, planned, synthetic, or collected.
