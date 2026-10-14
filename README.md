# Framing Analysis of Insurance Trade Publications

Did language in insurance trade publications shift between client-oriented and self-interest framing over time, and does it track market events?

## Data

- **Rough Notes:** ~8,800 articles, 2000-2026, from ProQuest.
- **The Standard:** 296 scanned PDF issues with no text layer, extracted by the pipeline below.

Raw data is not included in this repo.

## Pipeline

PDF (input) -> JSON (per-issue, then combined) -> xlsx (classified output). Runs on BU's SCC via SGE (`qsub`).

### 1. Extraction
`pipeline.py`, `batch_pipeline.py`, `run_batch_pipeline.sh`

Claude Sonnet via the Anthropic Batches API, in two stages:
1. Identify article titles and page numbers from the full issue PDF.
2. Build a per-article mini-PDF (pypdf) and extract clean text from that article only.

Articles are processed in isolation so one failure can't cascade. Output is one JSON per issue: `filename`, `publication`, `date`, `issue`, `article_title`, `text`.

### 2. Combining
`combine_articles.py`, `merge_split_halves.py`, `make_placeholder_articles.py`, `fill_placeholder.py`

Merges per-issue JSON into one file and handles split-half issues. PDFs that fail extraction (over the 32MB API limit, or empty/malformed responses) get blank stubs; `fill_placeholder.py` writes manually pasted text back as valid JSON.

### 3. Classification
`classification.py`

- Random sample from the combined JSON.
- Haiku relevance filter, then Sonnet classification, plus client-education and relationship-quality refinement passes.
- Processed articles are logged in persistent JSON files, so nothing is reprocessed across runs.
- Output xlsx columns: `title, url, date, publication, article_key, topic, framing, relevant_text, reason, other_theme`. One row per snippet.

## Usage

```bash
qsub run_batch_pipeline.sh
```
