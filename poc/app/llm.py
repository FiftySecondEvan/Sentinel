import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROMPT_VERSION = "v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(text: str, model: str, prompt_version: str = PROMPT_VERSION) -> str:
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _read_cache(key: str) -> Optional[Dict[str, Any]]:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(key: str, payload: Dict[str, Any]):
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


class OpenAIProvider:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed; add it to requirements.txt to use LLM features")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment; set it to use OpenAI provider")
        self.client = OpenAI(api_key=self.api_key)

    def synthesize(self, text: str, model: str = "gpt-4o-mini", max_snippets: int = 6) -> Dict[str, Any]:
        """Return a structured JSON result for a document using only the provided text.

        The function will select up to `max_snippets` snippets from MD&A/Risk sections if available,
        build a compact prompt and call the OpenAI Chat API with temperature=0.
        """
        # simple cache key
        key = _cache_key(text, model)
        cached = _read_cache(key)
        if cached:
            cached["cached"] = True
            return cached

        # Build snippets: naive split by paragraphs and pick paragraphs containing common headings or keywords
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        # Prefer paragraphs that mention Management Discussion or Risk
        preferred = []
        for p in paragraphs:
            low = p.lower()
            if "management" in low and ("discussion" in low or "md&a" in low):
                preferred.append(p)
        for p in paragraphs:
            low = p.lower()
            if "risk" in low and ("factor" in low or "risk" in low):
                preferred.append(p)

        # fallback: choose paragraphs with keywords
        if len(preferred) < max_snippets:
            for p in paragraphs:
                if any(k in p.lower() for k in ("adjusted","non-gaap","restat", "fraud", "uncertain")):
                    preferred.append(p)
                if len(preferred) >= max_snippets:
                    break

        snippets = preferred[:max_snippets]
        if not snippets:
            # last resort: take the first N short paragraphs
            snippets = paragraphs[:max_snippets]

        # Construct prompt
        system = (
            "You are a careful financial analyst. Using ONLY the provided numbered snippets and numeric facts, "
            "return a JSON object EXACTLY matching the schema requested. Do not invent facts or external data."
        )

        user_lines = ["Snippets:"]
        for i, s in enumerate(snippets, start=1):
            safe = s.replace("\n", " ")
            user_lines.append(f"{i}) {safe}")
        user_lines.append("\nTask: Provide a JSON object with the following schema:\n{")
        user_lines.append('  "score": int,          // 0-100 narrative integrity score')
        user_lines.append('  "reasons": [ {"text": str, "snippet_ids": [int], "severity": "low|medium|high"} ],')
        user_lines.append('  "highlights": [ {"text": str, "snippet_id": int} ],')
        user_lines.append('  "notes": str')
        user_lines.append('}')
        user_lines.append("Return JSON and NOTHING else.")

        prompt = "\n".join(user_lines)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Call OpenAI
        resp_text = None
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            # prefer message content
            resp_text = resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")

        # Attempt to parse JSON from response
        parsed = None
        try:
            # tolerant parse: find first '{' and last '}'
            start = resp_text.find("{")
            end = resp_text.rfind("}")
            json_text = resp_text[start:end+1] if start != -1 and end != -1 else resp_text
            parsed = json.loads(json_text)
        except Exception:
            # fallback: return raw text
            parsed = {"score": None, "reasons": [], "highlights": [], "notes": "failed to parse LLM JSON; see raw_llm", "raw_llm": resp_text}

        out = {
            "score": int(parsed.get("score")) if parsed.get("score") is not None else None,
            "reasons": parsed.get("reasons", []),
            "highlights": parsed.get("highlights", []),
            "notes": parsed.get("notes", ""),
            "raw_llm": resp_text,
            "model": model,
            "cached": False,
        }

        try:
            _write_cache(key, out)
        except Exception:
            pass

        return out


def get_default_provider() -> OpenAIProvider:
    # Allow selecting a provider via LLM_PROVIDER env var. If set to 'mock', return a deterministic MockProvider.
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "mock":
        return MockProvider()
    # If openai isn't installed or API key missing, fall back to mock provider for local testing
    if OpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        return MockProvider()
    return OpenAIProvider()


class MockProvider:
    """A deterministic mock provider for local testing when OpenAI is not available.

    It uses simple keyword counts to produce a repeatable score and returns JSON matching the expected schema.
    """
    def __init__(self):
        self.keywords = [
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

    def synthesize(self, text: str, model: str = "mock", max_snippets: int = 6) -> Dict[str, Any]:
        low = text.lower()
        score = 100
        reasons = []
        highlights = []
        # penalize for occurrences of strong keywords
        hits = 0
        for k in self.keywords:
            c = low.count(k)
            if c:
                hits += c
                reasons.append({"text": f"Found keyword '{k}' {c} times.", "snippet_ids": [], "severity": "medium" if c < 3 else "high"})
                if len(highlights) < 4:
                    highlights.append({"text": k, "snippet_id": 0})
        score = max(0, 100 - min(85, hits * 10))
        out = {
            "score": int(score),
            "reasons": reasons[:5],
            "highlights": highlights,
            "notes": "mock provider: heuristic count",
            "raw_llm": None,
            "model": model,
            "cached": False,
        }
        return out
