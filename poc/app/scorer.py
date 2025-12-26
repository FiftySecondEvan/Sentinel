import re
from typing import List, Optional, Dict, Any

KEYWORDS = [
    "restate",
    "material weakness",
    "going concern",
    "uncertain",
    "one-time",
    "adjusted",
    "non-gaap",
    "pro forma",
    "fraud",
    "error",
    "restatement",
    "irregularity",
    "limitation",
]


def _find_sentence_with_keyword(text: str, keyword: str) -> Optional[str]:
    pattern = r"([^.?!]*\b" + re.escape(keyword) + r"\b[^.?!]*)[.?!]"
    m = re.search(pattern, text, flags=re.I)
    if m:
        return m.group(1).strip()
    return None


def _extract_section(text: str, headings: List[str]) -> str:
    """Naive section extractor: find first heading that matches any of `headings` and return up to next all-caps heading or end.

    This is a heuristic but works for many 10-K/10-Q text extractions where sections start with headings.
    """
    # Normalize line breaks
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        for h in headings:
            if h in low:
                start_idx = i
                break
        if start_idx is not None:
            break
    if start_idx is None:
        return ""

    # Collect until we hit another likely section heading: a line in ALL CAPS or a line long and short words
    out_lines = []
    for line in lines[start_idx + 1 :]:
        # break heuristics
        if line.strip() and (line.strip().isupper() and len(line.strip()) < 100):
            break
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def analyze_text(text: str, history: Optional[List[float]] = None, numeric_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Heuristic analyzer enhanced to focus on MD&A/Risk Factors and simple numeric checks.

    numeric_features: optional dict from extractor with metrics and percent changes.
    """
    if not text:
        return {"score": 100, "flags": [], "trend": history or [100], "explanation": "no text provided", "numeric_features": numeric_features or {}}

    # Extract MD&A and Risk Factors sections
    mdna = _extract_section(text, ["management's discussion", "managements discussion", "management discussion and analysis", "md&a", "md&a (" ])
    risk = _extract_section(text, ["risk factors", "risks related to"]) 

    # Count keywords in sections with higher weight in MD&A and Risk
    counts = {k: 0 for k in KEYWORDS}
    total_hits = 0

    for k in KEYWORDS:
        # weight hits in MD&A and Risk
        hits_mdna = len(re.findall(r"\b" + re.escape(k) + r"\b", mdna, flags=re.I))
        hits_risk = len(re.findall(r"\b" + re.escape(k) + r"\b", risk, flags=re.I))
        hits_all = len(re.findall(r"\b" + re.escape(k) + r"\b", text, flags=re.I))
        # weight: mdna*3 + risk*4 + rest*1
        weighted = hits_mdna * 3 + hits_risk * 4 + (hits_all - hits_mdna - hits_risk)
        counts[k] = weighted
        total_hits += weighted

    # Base score: penalize by total hits
    penalty = min(85, total_hits * 12)
    base_score = max(0, 100 - penalty)

    # Extra penalty for strong words
    strong_hits = sum(counts.get(w, 0) for w in ("fraud", "restatement", "material weakness"))
    base_score = max(0, base_score - strong_hits * 8)

    # Numeric heuristics: if numeric_features provided, check for large negative pct_changes
    numeric_flags = []
    if numeric_features:
        for m, v in numeric_features.items():
            pct = v.get("pct_change")
            if pct is None:
                continue
            if pct <= -30:
                base_score = max(0, base_score - 25)
                numeric_flags.append({"metric": m, "pct_change": pct, "severity": "high"})
            elif pct <= -10:
                base_score = max(0, base_score - 10)
                numeric_flags.append({"metric": m, "pct_change": pct, "severity": "medium"})

    # Build flags: top 5 keywords by count (only include those with count>0)
    flags = []
    for k, c in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        if c <= 0:
            continue
        # Prefer sentences from MD&A or Risk if present
        quote = _find_sentence_with_keyword(mdna, k) or _find_sentence_with_keyword(risk, k) or _find_sentence_with_keyword(text, k) or k
        flags.append({"label": k, "count": c, "quote": quote})

    # Trend
    if history:
        trend = history + [base_score]
    else:
        trend = [min(100, base_score + 8), base_score]

    explanation = f"Detected {total_hits} weighted keyword hits; penalty {penalty} applied. Strong hits: {strong_hits}."

    # Optional spaCy-based entity extraction if available
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = None
    except Exception:
        nlp = None

    nlp_extractions = None
    if nlp:
        try:
            doc = nlp(mdna or text[:20000])
            ents = [{"text": e.text, "label": e.label_} for e in doc.ents if e.label_ in ("ORG","MONEY","PERCENT","DATE")]
            nlp_extractions = ents[:20]
        except Exception:
            nlp_extractions = None

    return {
        "score": int(base_score),
        "flags": flags,
        "trend": [int(x) for x in trend],
        "explanation": explanation,
        "numeric_flags": numeric_flags,
        "numeric_features": numeric_features or {},
        "nlp_extractions": nlp_extractions,
    }


def analyze_text_llm(text: str, model: str = "gpt-4o-mini", max_snippets: int = 6) -> Dict[str, Any]:
    """LLM-only scorer: call configured LLM provider to produce a JSON score/reasons/highlights.

    This function is independent from the heuristic `analyze_text` above.
    It uses the `poc.app.llm` provider to perform the call and returns the provider output merged into a standard wrapper.
    """
    if not text:
        return {"score": None, "reasons": [], "highlights": [], "notes": "no text provided"}

    try:
        from .llm import get_default_provider

        prov = get_default_provider()
        out = prov.synthesize(text, model=model, max_snippets=max_snippets)
        # normalize output: ensure score is int or None
        if out.get("score") is not None:
            try:
                out["score"] = int(out["score"])
            except Exception:
                out["score"] = None
        return out
    except Exception as e:
        return {"score": None, "reasons": [], "highlights": [], "notes": f"llm error: {e}"}
