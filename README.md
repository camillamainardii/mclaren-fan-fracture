# How a championship fractured a fandom

> Measuring McLaren's fan polarization through Reddit, 2024-2025.

Web & Social Media Analytics — Final project · Politecnico di Milano Graduate School of Management

**Team:** Camilla Mainardi · Ramna Jalil · Bilge Yalçın

---

## Summary

This project investigates whether McLaren F1's in-season management decisions during the 2024-2025 championship measurably polarized its online fanbase. Combining sentiment analysis, topic modeling, and community detection on **5,213 posts and 143,527 comments** scraped from r/formula1, we map a three-layered polarization signal — *temporal*, *topical*, and *structural* — anchored to ten flashpoint races.

## Repository contents

| Path | What it is |
|---|---|
| `dashboard.py` | Interactive Streamlit dashboard (the prototype) |
| `presentation.html` | 10-slide HTML deck (Reveal.js) |
| `presentation.pdf` | Exported PDF deck for submission |
| `collect_data.py` | Reddit scraper (public JSON API) |
| `analyze_sentiment.py` | VADER aspect-oriented sentiment |
| `analyze_sentiment_transformer.py` | Transformer (RoBERTa) sentiment |
| `validate_sentiment.py` | VADER vs transformer comparison |
| `analyze_topics.py` | LDA topic modeling per flashpoint |
| `analyze_communities.py` | Louvain community detection on commenter graph |
| `rerender_charts.py` | Brand-palette chart rendering |
| `export_pdf.py` | HTML deck → 16:9 PDF |
| `capture_dashboard.py` | Dashboard screenshot for slide 9 |
| `data/analysis/` | Pre-computed aggregates the dashboard reads |
| `images/` | Brand photos and chart outputs used by the deck |

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open <http://localhost:8501>.

## Live demo

Coming soon — deploy in progress on Streamlit Community Cloud.

## Methods at a glance

| Analysis | Technique | Library |
|---|---|---|
| Aspect-oriented sentiment | Transformer (cardiffnlp/twitter-roberta) validated against VADER | `transformers` · `vaderSentiment` |
| Topic modeling per event | LDA, K=5 per ±7-day window | `scikit-learn` |
| Community detection | Louvain on user co-engagement graph (8,329 users, 937,864 edges) | `networkx` · `python-louvain` |

## Key findings

1. **Hungary 2024 = patient zero.** McLaren-team sentiment crashes to −0.44 in the week of the swap.
2. **Late 2025 is when leadership loses the room.** Team-management sentiment hits −0.44 at Italian GP 2025 ('slow pit stops') and stays negative through the season finale.
3. **Verstappen as control validates findings.** His sentiment stays near zero across McLaren flashpoints — proving the negativity is McLaren-specific, not general F1 mood.
4. **'Zak Brown' enters LDA topics only at US GP Sprint 2025** — the moment fans named and blamed team leadership.
5. **Polarization is real but mild in network structure.** 4 commenter communities detected (modularity 0.267); smallest (n=643) is mildly Norris-leaning AND holds the most negative team sentiment.
