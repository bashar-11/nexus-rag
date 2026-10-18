# Contributing to Nexus RAG

Thanks for taking the time to contribute. This document covers the local
setup and the conventions the project follows.

## Getting started

```bash
git clone https://github.com/bashar-11/nexus-rag.git
cd nexus-rag

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`, `QDRANT_URL` and
`QDRANT_API_KEY`. Never commit `.env`.

Run the app:

```bash
streamlit run app.py
```

## Before you open a pull request

```bash
ruff check src app.py tests     # lint
ruff format src app.py tests    # formatting
pytest tests -q                 # tests
```

All three run automatically in CI on every pull request.

## Tests

The suite is written to run with **no API keys and no vector database**.
Anything that needs the network belongs behind a guard, not in a test.

When you touch retrieval or chunking, please add a case that would have
caught the bug. The chunking tests in particular assert Arabic sentence
boundaries, because the app is bilingual and it is easy to break Arabic
handling without noticing.

## Pull requests

1. Branch off `main` with a descriptive name.
2. Keep the change focused; unrelated cleanups belong in a separate PR.
3. Add a line to the description explaining *why*, not *what*.
4. Make sure CI is green.

## Reporting bugs

Open an issue with the Python version, the exact error, and the steps to
reproduce. If retrieval returned the wrong passage, include the query and the
document — that is the case that is hardest to debug and the most useful to
report.