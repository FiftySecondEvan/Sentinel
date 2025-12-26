from poc.app.scorer import analyze_text_llm
import os

def run_sample():
    text = """
Management's Discussion: The company reported adjusted, non-GAAP measures and one-time items.
Net income declined compared with prior period. Management noted uncertain macro conditions.

Risk Factors: Market volatility and potential restatement risk due to accounting changes.
"""
    print("LLM_PROVIDER:", os.environ.get("LLM_PROVIDER"))
    res = analyze_text_llm(text)
    import json

    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    run_sample()
