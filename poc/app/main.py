from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .scorer import analyze_text
from .extractor import extract_text_from_bytes

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Sentinel POC")


class AnalyzeRequest(BaseModel):
    text: str
    history: Optional[List[int]] = None
    # mode: 'heuristic' (default), 'llm'
    mode: Optional[str] = "heuristic"


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    mode = (req.mode or "heuristic").lower()
    if mode == "llm":
        # LLM-only analysis
        try:
            from .scorer import analyze_text_llm

            result = analyze_text_llm(req.text)
        except Exception as e:
            return JSONResponse({"error": f"LLM analysis failed: {e}"}, status_code=500)
    else:
        result = analyze_text(req.text, history=req.history)
    return JSONResponse(result)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), mode: str = "heuristic"):
    """Accept a file upload (pdf, xls, xlsx, txt) and run analysis on extracted text."""
    data = await file.read()
    mime_hint, text, numeric_features = extract_text_from_bytes(file.filename or file.content_type or 'file', data)
    if not text:
        return JSONResponse({"error": "could not extract text from file", "filename": file.filename}, status_code=400)

    # Optionally, send back a short snippet used for flags to keep responses compact
    snippet = text[:2000]
    if (mode or "heuristic").lower() == "llm":
        try:
            from .scorer import analyze_text_llm

            result = analyze_text_llm(text)
        except Exception as e:
            return JSONResponse({"error": f"LLM analysis failed: {e}"}, status_code=500)
    else:
        result = analyze_text(text, numeric_features=numeric_features)
    return JSONResponse({"filename": file.filename, "type": mime_hint, "snippet": snippet, "numeric_features": numeric_features, "result": result})


@app.get("/")
async def index():
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Sentinel POC</h1><p>Frontend not found.</p>")
