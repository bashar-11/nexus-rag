<div align="center">

# Nexus RAG

**A bilingual (Arabic / English) retrieval-augmented generation knowledge hub.**
Hybrid BM25 + dense vector search, fused with Reciprocal Rank Fusion, served through a Streamlit UI.

[![CI](https://github.com/bashar-11/nexus-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/bashar-11/nexus-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

</div>

---

## The problem this solves

A vector store alone gets you *semantic* matches and misses exact keywords —
product codes, names, dates, error messages. Pure keyword search gets you
exact matches and misses paraphrases. Nexus RAG runs **both** retrievers over
the same document set and merges them, so a question works whether the user
asks in natural language or searches for a literal term.

## How it works

```
       Upload PDF / TXT
              |
      clean + normalise          (_clean_text: collapse whitespace, drop empty pages)
              |
      recursive chunking         800 chars, 200 overlap, sentence-aware
              |                 (Arabic punctuation ، ؛ ؟ recognised)
      +-------+-------+
      |               |
  BM25 (lexical)   Qdrant (dense)   paraphrase-multilingual-MiniLM-L12-v2, 384-d
      |               |
      +-------+-------+
              |
   Reciprocal Rank Fusion        each doc scores 1/(k + rank) per list, scores summed
              |
         top-k passages  --------->  Groq Llama-3.3-70B  --------->  cited answer
```

**Why RRF and not score blending?** Dense and lexical retrievers produce
scores on completely different scales, so averaging them requires per-corpus
calibration that breaks as the corpus changes. RRF fuses on **rank position
only**, so there is nothing to calibrate. A document that both retrievers rank
highly wins, which is exactly the behaviour you want.

Every answer is returned with the source file and page, so a claim can be
checked against the original document.

## Bilingual by design

The app answers in **the language of the question**. Three places matter:

- **Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2`, which handles
  Arabic and English in the same vector space, so one collection serves both.
- **Chunking** — separators include the Arabic comma `،`, semicolon `؛` and
  question mark `؟`, so chunks break on real sentence boundaries instead of
  mid-word.
- **Query classification** — summary detection matches Arabic keywords as well
  as English ones, so "لخص لي المستند" is treated as a summary request rather
  than a lookup.

## Features

- Hybrid retrieval: BM25 lexical search + dense semantic search
- Reciprocal Rank Fusion to merge the two result sets
- Multi-document indexing with per-file delete and automatic re-indexing
- PDF (with page-level citations) and TXT support
- Bilingual query handling and answer generation
- Dark-mode Streamlit UI
- Answer cites the source file and page for every passage used

## Setup

**Prerequisites:** Python 3.10+ and access to a [Groq](https://console.groq.com/keys)
API key and a [Qdrant](https://cloud.qdrant.io/) cluster (a local Qdrant
instance also works).

```bash
git clone https://github.com/bashar-11/nexus-rag.git
cd nexus-rag

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:

```bash
GROQ_API_KEY=gsk_...
QDRANT_URL=https://your-cluster-id.eu-central.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

> Every value can also be set under **Settings → Secrets** on Streamlit Cloud.
> Resolution order is `.env` first, then Streamlit secrets.

Run it:

```bash
streamlit run app.py
```

## Deployment to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Go to **share.streamlit.io** → *Deploy an app*.
3. Select the repo, then add `GROQ_API_KEY`, `QDRANT_URL` and
   `QDRANT_API_KEY` under *Advanced settings → Secrets*.

Community Cloud has an ephemeral filesystem, which is why the vector store is
external — uploaded files live for the session only, but the index persists in
Qdrant for the lifetime of the collection.

## Project structure

```
nexus-rag/
├── app.py                    # Streamlit UI, styling, session state
├── src/
│   ├── config.py             # Settings, resolved from .env or Streamlit secrets
│   ├── document_loader.py    # PDF/TXT parsing, cleaning, chunking
│   ├── embeddings.py         # Cached embedding model
│   ├── vector_store.py       # Qdrant store, BM25, Reciprocal Rank Fusion
│   └── rag_engine.py         # Prompting, query classification, answer generation
├── tests/                    # pytest suite (no API keys or database needed)
├── .devcontainer/            # Dev container definition
├── .github/workflows/ci.yml  # Lint, test, security scan
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── CONTRIBUTING.md
```

## Development

```bash
pip install -r requirements-dev.txt

ruff check src app.py tests     # lint
ruff format src app.py tests    # format
pytest tests -q                 # tests
```

CI runs lint, tests and a dependency vulnerability scan on every push and pull
request. The test suite is written to run **without API keys or a vector
database** — see [CONTRIBUTING.md](CONTRIBUTING.md) for the conventions.

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Answer generation | Groq — Llama 3.3 70B Versatile |
| Dense vector store | Qdrant |
| Lexical retrieval | BM25 (`rank-bm25` via LangChain) |
| Embeddings | Hugging Face — `paraphrase-multilingual-MiniLM-L12-v2` |

## Known limitations

- The BM25 index is rebuilt in memory on each query from the chunks held in
  session state. Fine for hundreds of chunks, slow beyond that. Persisting a
  prebuilt BM25 index would be the fix.
- PDFs must contain an embedded text layer. Scanned pages are skipped.
- A local Qdrant instance on Streamlit Community Cloud does not persist across
  restarts; use Qdrant Cloud for a durable index.

## License

[MIT](LICENSE) © 2026 Bashar Sayed