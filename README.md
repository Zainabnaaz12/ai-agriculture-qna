# Project Samarth — AI-Powered Agricultural Q&A

![build-status](https://img.shields.io/badge/build-manual-lightgrey) ![license](https://img.shields.io/badge/license-MIT-blue)

## What this project does

This project is an open, developer-focused research prototype that provides an interactive, data-driven Q&A system for Indian agriculture. It ingests official datasets (crop production, rainfall, mandi prices) from data.gov.in, preprocesses and caches them, and uses a Retrieval-Augmented Generation (RAG) approach with the Groq API (Llama models) to answer developer and analyst queries with citations to source datasets.

Key capabilities
- Load and preprocess crop production, rainfall, and mandi (market) price datasets.
- Provide both a Streamlit web UI (`app.py`) and a CLI interactive mode (`main.py`).
- Data-aware query routing and summarization (see `query_router.py` and `rag_system.py`).
- Tools for collecting data from data.gov.in and creating demo/sample datasets (`data_collection.py`).

## Why it's useful
- Enables fast, evidence-backed answers about crop production, rainfall trends, and market prices.
- Helpful for analysts, students, and developers building agriculture-focused NLP applications or demos.
- Built to be pragmatic: includes sample data generators so demos work even without API keys.

## Quick repo map
- `app.py` — Streamlit web interface (web Q&A + data explorer).
- `main.py` — CLI setup/test runner and interactive mode.
- `rag_system.py` — Core RAG logic, data loading, Groq API integration, prompt construction.
- `query_router.py` — Lightweight intent detection and direct data queries.
- `data_collection.py` — Enhanced data collection from data.gov.in and utilities to create sample data.
- `data_preprocessing.py` — Cleaning and standardizing raw datasets into `data/processed/`.
- `requirements.txt` — Python dependencies.
- `data/` — Example raw and processed datasets (not all datasets may be present).
- `sources_metadata.json` — Mapping of data.gov.in resource IDs used for fetching.

## Quick contract (inputs / outputs / success criteria)
- Inputs: raw CSV data under `data/raw/`, environment variables for API keys, and user queries.
- Outputs: cleaned CSVs in `data/processed/`, optional vector store (`vector_db/`), and generated answers in the UI/CLI.
- Success criteria: system initializes without missing critical env vars, processed CSVs are present, and `SamarthRAG.answer_query()` returns answers.

## Requirements
Python 3.9+ recommended. Install dependencies:

```powershell
pip install -r requirements.txt
```

`requirements.txt` highlights:
- pandas, numpy, requests
- streamlit (for `app.py`)
- groq (Groq API client)
- python-dotenv

## Environment variables
Create a `.env` file at the repo root with at least:

```text
GROQ_API_KEY=your_groq_api_key_here
DATA_GOV_API_KEY=optional_data_gov_api_key
# Optional keys used in some scripts
ANTHROPIC_API_KEY=optional_claude_key
```

Notes:
- `GROQ_API_KEY` is required for the RAG system (`rag_system.py`).
- `DATA_GOV_API_KEY` is optional but needed to fetch datasets programmatically from data.gov.in.
- The code includes safe fallbacks (sample data) when public API keys are not present.

## Setup & typical workflows

1) Prepare environment and install packages

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) (Optional) Fetch datasets from data.gov.in
- Run the data collector to download datasets or create sample data if you don't have API access:

```powershell
python data_collection.py
# or for the improved interactive collector
python data_collection.py
```

3) Preprocess data

```powershell
python data_preprocessing.py
```

Processed files are written to `data/processed/` (e.g., `crop_production_clean.csv`, `rainfall_clean.csv`).

4) Start the system
- CLI interactive + tests:

```powershell
python main.py
```

- Streamlit web UI (recommended for quick demos):

```powershell
streamlit run app.py
```

5) Using the system
- In the Streamlit app: click "Initialize System" in the sidebar, ask questions in the UI.
- In CLI: choose the interactive mode from the `main.py` menu or run test queries.

## Examples
- Ask: "Which are the top 5 rice producing states in India?"
- Ask: "Compare rainfall trends in Maharashtra and Karnataka"

The system will return answers that reference the dataset used (Crop Production / Rainfall / Mandi Price).

## Testing / verification
- `main.py` includes `check_environment()` to detect missing API keys and processed data.
- After preprocessing, run `python main.py` and choose "Run test queries".

## Contribution & support
- Issues and PRs are welcome. Please follow repository contribution guidelines.
- If present, see `docs/CONTRIBUTING.md` or `CONTRIBUTING.md` at the repo root for details.

## Maintainers
- Primary repo owner: `Zainabnaaz12` (GitHub user)
- See repository settings / `CODEOWNERS` for the authoritative maintainer list.

## Security & data handling
- Do not commit API keys or secrets. Use `.env` and add it to `.gitignore`.
- Datasets from data.gov.in are public government datasets; respect any data use policies from upstream sources.
