import io
from typing import Tuple, Dict, Any

from pdfminer.high_level import extract_text
import pandas as pd


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using pdfminer."""
    with io.BytesIO(data) as fh:
        try:
            text = extract_text(fh)
        except Exception:
            text = ""
    return text or ""


def extract_numeric_features_from_excel_bytes(data: bytes) -> Dict[str, Any]:
    """Attempt to find common financial metric rows and compute simple percent changes.

    Heuristic: look for rows where a cell contains metric keywords (revenue, net income, total assets, cash)
    and the row contains at least two numeric values — compute percent change between last two numeric values.
    Returns a dict mapping metric_name -> {last, prev, pct_change}.
    """
    features = {}
    with io.BytesIO(data) as fh:
        try:
            df_dict = pd.read_excel(fh, sheet_name=None)
        except Exception:
            try:
                fh.seek(0)
                df_dict = pd.read_excel(fh, sheet_name=None, engine="xlrd")
            except Exception:
                return features

    keywords = ["revenue", "net income", "total assets", "cash", "operating income", "income from operations"]
    for sheet_name, df in df_dict.items():
        # Ensure we have a DataFrame
        if df is None or df.empty:
            continue
        # Convert all to string for header scanning
        for idx, row in df.iterrows():
            # look for a string cell matching metric keyword
            row_strings = [str(x).lower() for x in row.tolist()]
            joined = " ".join(row_strings)
            metric_found = None
            for kw in keywords:
                if kw in joined:
                    metric_found = kw
                    break
            if not metric_found:
                continue

            # collect numeric values from row
            numeric_vals = []
            for v in row.tolist():
                try:
                    num = float(v)
                    if not (num != num):  # filter NaN
                        numeric_vals.append(num)
                except Exception:
                    continue
            if len(numeric_vals) >= 2:
                last = numeric_vals[-1]
                prev = numeric_vals[-2]
                pct = None
                try:
                    pct = ((last - prev) / abs(prev)) * 100 if prev != 0 else None
                except Exception:
                    pct = None
                features[metric_found] = {"last": last, "prev": prev, "pct_change": pct, "sheet": sheet_name}

    return features


def extract_text_from_excel_bytes(data: bytes, filename: str = "") -> str:
    """Read Excel bytes and concatenate cell contents into a text blob."""
    with io.BytesIO(data) as fh:
        text_parts = []
        # Try reading with pandas; try both engines if needed
        try:
            df_dict = pd.read_excel(fh, sheet_name=None)
        except Exception:
            try:
                fh.seek(0)
                df_dict = pd.read_excel(fh, sheet_name=None, engine="xlrd")
            except Exception:
                return ""

        for sheet_name, df in df_dict.items():
            # flatten all cells into strings
            for col in df.columns:
                for v in df[col].astype(str).tolist():
                    if v and v != 'nan':
                        text_parts.append(v)

        return "\n".join(text_parts)


def extract_text_from_bytes(filename: str, data: bytes) -> Tuple[str, str, Dict[str, Any]]:
    """Detect by filename extension and extract text and simple numeric features.

    Returns (mime_hint, text, numeric_features).
    """
    name = filename.lower()
    if name.endswith('.pdf'):
        return 'pdf', extract_text_from_pdf_bytes(data), {}
    if name.endswith(('.xls', '.xlsx')):
        nums = extract_numeric_features_from_excel_bytes(data)
        text = extract_text_from_excel_bytes(data, filename=name)
        return 'excel', text, nums
    # fallback: try decode as text
    try:
        text = data.decode('utf-8')
    except Exception:
        try:
            text = data.decode('latin-1')
        except Exception:
            text = ""
    return 'text', text, {}
