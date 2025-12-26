# Sentinel POC (FastAPI + Tiny UI)

This is a small proof-of-concept that demonstrates:

- A FastAPI backend with a single `/api/analyze` endpoint
- A very small heuristic-based scorer that returns a `Narrative Integrity Index`-style score, a small list of flags, and a short trend
- A tiny HTML frontend to paste text and view the JSON output

Quick start (macOS / zsh):

1. Create a virtualenv and activate it

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r poc/requirements.txt
```

Optional: set your OpenAI key in a local `.env` file (ignored by git):

```bash
echo 'OPENAI_API_KEY=sk-...' > .env
```

3. Run the app

```bash
uvicorn poc.app.main:app --reload --port 8000
```

4. Open http://127.0.0.1:8000/ in your browser and paste text to test.

Notes:
- This is intentionally minimal and uses simple keyword heuristics. It is designed as a starting point for integrating stronger NLP (e.g., spaCy, transformers) and document ingestion pipelines (PDF/10-K parsing).
- To test with multiple historical scores, POST JSON with a `history` array: `{ "text": "...", "history": [85,78,72] }`.

Upload endpoint:

- The UI supports uploading `pdf`, `xls`, `xlsx`, and plain `txt` files. The server extracts text from the uploaded file and runs the same analyzer.

Example using `curl`:

```bash
curl -F "file=@/path/to/10-K.pdf" http://127.0.0.1:8000/api/upload
```
