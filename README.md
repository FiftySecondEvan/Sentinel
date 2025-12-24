# Sentinel
Fraud Sentinel Proof of Concept
**User:** Retail Investors and Short-Sellers
**Decision:** Which public companies are misrepresenting their financial positions and future prospects, and are potentially fraudulent.
**Output:** A product that can ingest materials relating to the health and future outlook of a company: SEC filings, earnings call transcripts, investor relations presentations. Ingested materials are then analyzes to produce a “narrative integrity index” that examines the relationship between the accounting reports and CXO claims. 
•	A 0–100 Narrative Integrity Index
•	Top 5 flags (each with a quote + numbers), but does not necessarily need to output five if a company appears low-risk for fraud
•	Score trend over time (last 8 quarters)
**Non-goal:** This product does not declare fraud. It only flags inconsistencies / elevated risk.
Success criteria: what would make you say “this works”?
•	Scores and explanations provide a useful starting point for users to further investigate fraud risks
•	Explanations feel credible to a finance reader
•	On 2–3 known blowups/restatements, the score worsens beforehand
•	On 5 stable firms, it doesn’t scream every quarter

<img width="468" height="378" alt="image" src="https://github.com/user-attachments/assets/89842d61-4aec-442a-923a-ec7f7b6599ca" />
