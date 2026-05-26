"""
Analysis 2: Topic modeling around flashpoint events.

For each of the 10 flashpoint races, we:
  1. Pull all comments within ±7 days of the event
  2. Preprocess (lowercase, remove urls, stopwords, F1-noise terms)
  3. Build a TF-IDF matrix
  4. Fit LDA with K topics (default 5)
  5. Extract top terms per topic + representative comments

Outputs:
  data/analysis/topics_per_event.json    — all topics + top terms + sample comments
  data/analysis/topics_per_event.txt     — human-readable summary
  data/analysis/chart_topics_per_event.png  — heatmap-style overview
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"
ANALYSIS_DIR = DATA_DIR / "analysis"
EVENTS_FILE = DATA_DIR / "flashpoint_events.json"
COMMENTS_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments.csv"

WINDOW_DAYS = 7
K_TOPICS = 5            # topics per event
TOP_TERMS = 12          # top terms shown per topic
TOP_DOCS = 3            # top representative comments per topic
MIN_COMMENT_LEN = 25    # chars; filters very short noise comments
MAX_FEATURES = 3000     # vocab size cap for LDA

# Stopwords — sklearn's English + F1-domain noise
SKLEARN_STOPWORDS = "english"
DOMAIN_STOPWORDS = set("""
gp grand prix race racing f1 formula formula1 lap laps season
team teams driver drivers car cars yes no get got like just one
think really lol yeah idk imo tbh actually probably maybe even
people don dont didnt cant couldnt wouldnt shouldnt isnt arent
make made gonna wanna would could should will shall might must
much many lot good bad great better best worse worst
way thing things ago today right wrong same other than
oh ok hey wow nice cool great fine sure huh wait
guy guys man dude lol lmao haha way
""".split())

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")


def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = text.lower()
    # keep apostrophes inside words (don't, he's), strip everything else
    text = re.sub(r"[^a-z\s']", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_comments_with_bodies():
    """Load the per-comment metadata CSV + comment bodies from disk."""
    df = pd.read_csv(COMMENTS_CSV)
    body_by_id = {}
    for f in COMMENTS_DIR.glob("*.json"):
        try:
            for c in json.loads(f.read_text()):
                if c.get("body") and c.get("id"):
                    body_by_id[c["id"]] = c["body"]
        except Exception:
            continue
    df["body"] = df["comment_id"].map(body_by_id)
    df = df[df["body"].notna()].reset_index(drop=True)
    df["body_clean"] = df["body"].apply(preprocess)
    df = df[df["body_clean"].str.len() >= MIN_COMMENT_LEN].reset_index(drop=True)
    return df


def topics_for_event(df, event, k=K_TOPICS):
    """Run LDA on the ±WINDOW_DAYS window around `event`. Returns dict."""
    edate = datetime.fromisoformat(event["date"]).replace(tzinfo=timezone.utc)
    ws = edate.timestamp() - WINDOW_DAYS * 86400
    we = edate.timestamp() + WINDOW_DAYS * 86400
    window = df[(df["created_utc"] >= ws) & (df["created_utc"] <= we)].copy()

    if len(window) < 100:
        return {"event": event, "n_comments": len(window), "topics": [], "note": "too few comments"}

    # TF-IDF with domain stopwords
    stop_set = list(DOMAIN_STOPWORDS)
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        stop_words=SKLEARN_STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
    )
    X = vectorizer.fit_transform(window["body_clean"])
    vocab = vectorizer.get_feature_names_out()

    # Drop domain stopwords post-vectorization (some sneak through)
    keep_mask = np.array([not any(w in DOMAIN_STOPWORDS for w in term.split())
                          for term in vocab])
    X = X[:, keep_mask]
    vocab = vocab[keep_mask]

    if X.shape[1] < 50:
        return {"event": event, "n_comments": len(window), "topics": [], "note": "vocab too small"}

    # LDA
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="online",
        max_iter=20,
        n_jobs=-1,
    )
    lda.fit(X)

    # Doc-topic distribution for picking representative comments
    doc_topic = lda.transform(X)

    topics = []
    for k_idx in range(k):
        # Top terms
        term_weights = lda.components_[k_idx]
        top_term_indices = term_weights.argsort()[::-1][:TOP_TERMS]
        top_terms = [vocab[i] for i in top_term_indices]

        # Top representative comments for this topic
        doc_scores = doc_topic[:, k_idx]
        top_doc_indices = doc_scores.argsort()[::-1][:TOP_DOCS]
        top_docs = []
        for di in top_doc_indices:
            row = window.iloc[di]
            top_docs.append({
                "comment_id": row["comment_id"],
                "score": float(doc_scores[di]),
                "body": (row["body"] or "")[:300],
                "vader_compound": float(row["compound"]),
            })

        # Average sentiment of comments dominated by this topic
        dominant_mask = doc_topic.argmax(axis=1) == k_idx
        n_dominant = int(dominant_mask.sum())
        mean_sentiment = float(window[dominant_mask]["compound"].mean()) if n_dominant else 0.0

        topics.append({
            "topic_id": k_idx,
            "top_terms": top_terms,
            "n_dominant_docs": n_dominant,
            "mean_sentiment": mean_sentiment,
            "top_comments": top_docs,
        })

    return {
        "event": event,
        "n_comments": int(len(window)),
        "n_unique_authors": int(window["author"].nunique()),
        "topics": topics,
    }


def main():
    print("Loading comments + bodies ...")
    df = load_comments_with_bodies()
    print(f"   {len(df):,} comments (with bodies, len ≥ {MIN_COMMENT_LEN})")

    events = json.loads(EVENTS_FILE.read_text())
    print(f"\nRunning LDA (K={K_TOPICS}) for {len(events)} flashpoint events ...")

    all_results = []
    for e in events:
        print(f"\n[{e['date']}] {e['race']}")
        result = topics_for_event(df, e)
        all_results.append(result)
        if not result["topics"]:
            print(f"   skipped: {result.get('note', 'no topics')}")
            continue
        print(f"   {result['n_comments']:,} comments, {result['n_unique_authors']:,} authors")
        for t in result["topics"]:
            terms = ", ".join(t["top_terms"][:8])
            print(f"     T{t['topic_id']}  sent={t['mean_sentiment']:+.2f}  "
                  f"n={t['n_dominant_docs']:4d}   {terms}")

    # Save full JSON
    out_json = ANALYSIS_DIR / "topics_per_event.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved → {out_json}")

    # Write human-readable text summary
    out_txt = ANALYSIS_DIR / "topics_per_event.txt"
    with out_txt.open("w") as f:
        f.write("TOPIC MODELING PER FLASHPOINT EVENT\n")
        f.write("=" * 75 + "\n")
        for r in all_results:
            e = r["event"]
            f.write(f"\n{'='*75}\n{e['date']}  {e['race']}\n")
            f.write(f"{e['note']}\n")
            if not r.get("topics"):
                f.write(f"  [skipped: {r.get('note', 'no data')}]\n")
                continue
            f.write(f"  Comments in window: {r['n_comments']:,}\n")
            f.write(f"  Unique authors:     {r['n_unique_authors']:,}\n")
            for t in r["topics"]:
                f.write(f"\n  Topic {t['topic_id']}  (n={t['n_dominant_docs']}, sentiment={t['mean_sentiment']:+.3f})\n")
                f.write(f"    Top terms: {', '.join(t['top_terms'])}\n")
                f.write(f"    Representative comments:\n")
                for c in t["top_comments"]:
                    body = c["body"].replace("\n", " ")[:200]
                    f.write(f"      • [VADER {c['vader_compound']:+.2f}]  {body}\n")
    print(f"Saved → {out_txt}")

    # Build a heatmap-ish chart: rows = events, cols = topics
    plot_topics_chart(all_results, ANALYSIS_DIR / "chart_topics_per_event.png")


def plot_topics_chart(results, out_path):
    """One panel per event: bar chart of topics with their mean sentiment + top 3 terms as label."""
    plottable = [r for r in results if r.get("topics")]
    n = len(plottable)
    if n == 0:
        return
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), squeeze=False)

    for i, r in enumerate(plottable):
        ax = axes[i // cols, i % cols]
        topics = r["topics"]
        labels = [", ".join(t["top_terms"][:3]) for t in topics]
        sents = [t["mean_sentiment"] for t in topics]
        ns = [t["n_dominant_docs"] for t in topics]
        colors = ["#D32F2F" if s < -0.05 else "#1E88E5" if s < 0.15 else "#43A047" for s in sents]

        bars = ax.barh(range(len(topics)), ns, color=colors, alpha=0.85)
        ax.set_yticks(range(len(topics)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        for bar, s in zip(bars, sents):
            ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                    f"sent {s:+.2f}", va="center", fontsize=8, color="#444")
        ax.set_title(f"{r['event']['date']}  {r['event']['race']}",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("# comments dominated by topic", fontsize=8)
        ax.grid(alpha=0.2, axis="x")

    # Hide unused axes
    for j in range(len(plottable), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Topics around each flashpoint event\n"
                 "Bar = #comments dominated by topic   |   Color = mean sentiment "
                 "(red: very neg · blue: mild · green: positive)",
                 fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart → {out_path}")


if __name__ == "__main__":
    main()
