"""
Validate VADER sentiment scores against a transformer-based model.

Method:
  - Load the per-comment VADER scores from analyze_sentiment.py
  - Take a stratified sample (~5,000 comments) covering all aspects and
    flashpoint event windows
  - Score each sampled comment with cardiffnlp/twitter-roberta-base-sentiment-latest
    (a RoBERTa model fine-tuned for social media sentiment)
  - Compute:
      * Pearson + Spearman correlation between VADER compound and Transformer score
      * 3-class label agreement (positive/neutral/negative)
      * Confusion matrix
      * Disagreement examples

This validates whether the VADER trajectories we saw are robust. If the two
methods agree, our findings are defensible; if they don't, we know where.
"""

import json
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import pearsonr, spearmanr

ANALYSIS_DIR = Path("data/analysis")
COMMENTS_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments.csv"
EVENTS_FILE = Path("data/flashpoint_events.json")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SAMPLE_SIZE = 5000          # Stratified sample across aspects + events
MAX_LEN = 256               # Token cap (avg comment is ~70 tokens)
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

random.seed(42)
np.random.seed(42)


def label_from_vader(compound):
    """Map VADER compound score to 3-class label (standard thresholds)."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def label_from_transformer_scores(scores):
    """Cardiff model emits [negative, neutral, positive] probabilities."""
    return ["negative", "neutral", "positive"][int(np.argmax(scores))]


def transformer_compound(scores):
    """Map [neg, neu, pos] probs to a single compound score in [-1, 1]
    for direct comparison with VADER."""
    return float(scores[2] - scores[0])  # pos - neg


def stratified_sample(df, events, sample_size=SAMPLE_SIZE):
    """Sample so that we cover (a) all 5 aspects and (b) all event windows."""
    pieces = []
    per_bucket = max(50, sample_size // (5 * len(events)))  # rough split

    for e in events:
        edate = datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc)
        window_start = edate.timestamp() - 7 * 86400
        window_end = edate.timestamp() + 7 * 86400
        for aspect in ["Norris", "Piastri", "McLaren", "Verstappen", "Team mgmt"]:
            bucket = df[
                (df["created_utc"] >= window_start)
                & (df["created_utc"] <= window_end)
                & (df[f"mentions_{aspect}"])
            ]
            if len(bucket) == 0:
                continue
            n = min(per_bucket, len(bucket))
            pieces.append(bucket.sample(n=n, random_state=42))

    sample = pd.concat(pieces, ignore_index=True).drop_duplicates(subset="comment_id")
    # Fill the rest with random comments to hit sample_size
    if len(sample) < sample_size:
        remaining = sample_size - len(sample)
        leftover = df[~df["comment_id"].isin(sample["comment_id"])]
        if len(leftover) > 0:
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=42)
            sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.head(sample_size).reset_index(drop=True)
    return sample


def score_with_transformer(texts, tokenizer, model):
    """Batch-score a list of texts. Returns array of [neg, neu, pos] probs."""
    all_probs = []
    model.eval()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Truncate long comments — the model has a 512-token limit
        batch = [t[:2000] for t in batch]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"   batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}  "
                  f"({i + len(batch)}/{len(texts)} comments)",
                  flush=True)
    return np.vstack(all_probs)


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading VADER results from {COMMENTS_CSV} ...")
    df = pd.read_csv(COMMENTS_CSV)
    print(f"   loaded {len(df):,} per-comment rows")

    # We need the original body text, which isn't in the CSV.
    # Reload from comment files keyed by comment_id.
    print("Re-loading comment bodies (needed for transformer) ...")
    body_by_id = {}
    for f in Path("data/comments").glob("*.json"):
        try:
            for c in json.loads(f.read_text()):
                if c.get("body") and c.get("id"):
                    body_by_id[c["id"]] = c["body"]
        except Exception:
            continue
    print(f"   loaded {len(body_by_id):,} comment bodies")

    df = df[df["comment_id"].isin(body_by_id.keys())].reset_index(drop=True)
    df["body"] = df["comment_id"].map(body_by_id)

    print(f"\nStratified sampling {SAMPLE_SIZE} comments across aspects × events ...")
    events = json.loads(EVENTS_FILE.read_text())
    sample = stratified_sample(df, events, sample_size=SAMPLE_SIZE)
    print(f"   sample size: {len(sample):,}")

    print(f"\nLoading transformer model {MODEL_NAME} (first time may take ~1 min) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"   model loaded in {time.time()-t0:.0f}s")

    print(f"\nScoring {len(sample)} comments with transformer "
          f"(batch={BATCH_SIZE}) ...")
    t0 = time.time()
    probs = score_with_transformer(sample["body"].tolist(), tokenizer, model)
    elapsed = time.time() - t0
    print(f"   done in {elapsed:.0f}s "
          f"(~{1000*elapsed/len(sample):.0f}ms/comment)")

    sample["tx_neg"] = probs[:, 0]
    sample["tx_neu"] = probs[:, 1]
    sample["tx_pos"] = probs[:, 2]
    sample["tx_compound"] = sample["tx_pos"] - sample["tx_neg"]
    sample["tx_label"] = [label_from_transformer_scores(p) for p in probs]
    sample["vader_label"] = sample["compound"].apply(label_from_vader)

    sample.to_csv(ANALYSIS_DIR / "sentiment_validation_sample.csv", index=False)

    # ── Comparison stats ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  VALIDATION RESULTS")
    print("=" * 75)

    pr, pp = pearsonr(sample["compound"], sample["tx_compound"])
    sr, sp = spearmanr(sample["compound"], sample["tx_compound"])
    print(f"\nCorrelation (VADER compound  vs  Transformer compound):")
    print(f"  Pearson  r = {pr:+.4f}  (p = {pp:.2e})")
    print(f"  Spearman ρ = {sr:+.4f}  (p = {sp:.2e})")

    agree = (sample["vader_label"] == sample["tx_label"]).mean()
    print(f"\n3-class label agreement (pos/neu/neg):  {agree:.1%}")

    # Confusion matrix
    print("\nConfusion matrix (rows = VADER, cols = Transformer):")
    labels = ["negative", "neutral", "positive"]
    cm = pd.crosstab(sample["vader_label"], sample["tx_label"],
                     rownames=["VADER"], colnames=["Transformer"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    print(cm.to_string())

    # Per-method label distributions
    print("\nLabel distribution:")
    vader_dist = sample["vader_label"].value_counts(normalize=True).reindex(labels, fill_value=0)
    tx_dist = sample["tx_label"].value_counts(normalize=True).reindex(labels, fill_value=0)
    print(f"  {'label':<10s}  {'VADER':>8s}  {'Transformer':>12s}")
    for L in labels:
        print(f"  {L:<10s}  {vader_dist[L]:>8.1%}  {tx_dist[L]:>12.1%}")

    # Per-aspect agreement
    print("\nLabel agreement broken down by aspect:")
    for aspect in ["Norris", "Piastri", "McLaren", "Verstappen", "Team mgmt"]:
        sub = sample[sample[f"mentions_{aspect}"]]
        if len(sub) == 0:
            continue
        a = (sub["vader_label"] == sub["tx_label"]).mean()
        print(f"  {aspect:<12s}  {len(sub):5d} comments   agreement {a:.1%}")

    # Show 5 examples where they sharply disagree
    sample["disagreement"] = (sample["tx_compound"] - sample["compound"]).abs()
    biggest_diff = sample.nlargest(5, "disagreement")[
        ["compound", "tx_compound", "vader_label", "tx_label", "body"]
    ]
    print("\nTop 5 disagreement cases (most likely VADER false-positives/negatives):")
    for _, r in biggest_diff.iterrows():
        body_short = (r["body"] or "")[:120].replace("\n", " ")
        print(f"  VADER={r['compound']:+.2f}({r['vader_label']:<8s})  "
              f"TX={r['tx_compound']:+.2f}({r['tx_label']:<8s})  | {body_short}")


if __name__ == "__main__":
    main()
