"""
Re-score all 110k comments with a transformer model (cardiffnlp/twitter-roberta).
Regenerate the trajectory chart with these higher-quality sentiment scores.

Reuses the aspect detection + flashpoint events from the VADER run.

Outputs:
  data/analysis/sentiment_per_aspect_comments_TX.csv  — per-comment transformer scores
  data/analysis/sentiment_per_aspect_weekly_TX.csv    — weekly aggregates (TX)
  data/analysis/chart_sentiment_trajectory_TX.png     — the publishable chart
  data/analysis/chart_sentiment_comparison.png        — VADER vs Transformer side-by-side

Resumable: skips already-scored comments stored in the cache file.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"
ANALYSIS_DIR = DATA_DIR / "analysis"
EVENTS_FILE = DATA_DIR / "flashpoint_events.json"
VADER_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments.csv"
TX_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments_TX.csv"
TX_WEEKLY_CSV = ANALYSIS_DIR / "sentiment_per_aspect_weekly_TX.csv"
TX_CHART = ANALYSIS_DIR / "chart_sentiment_trajectory_TX.png"
COMPARE_CHART = ANALYSIS_DIR / "chart_sentiment_comparison.png"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16               # moderate batch — balance speed vs heat
MAX_LEN = 256
SLEEP_BETWEEN_BATCHES = 0.15  # brief cool-down between batches
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

ASPECTS = ["Norris", "Piastri", "McLaren", "Verstappen", "Team mgmt"]


def load_comment_bodies():
    """Load all comment bodies from disk, keyed by comment_id."""
    body_by_id = {}
    for f in COMMENTS_DIR.glob("*.json"):
        try:
            for c in json.loads(f.read_text()):
                if c.get("body") and c.get("id"):
                    body_by_id[c["id"]] = c["body"]
        except Exception:
            continue
    return body_by_id


def score_in_batches(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Score a list of texts; yield batches of probabilities for incremental save.
    Includes brief cool-down sleeps between batches to reduce thermal load."""
    model.eval()
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [t[:2000] for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        yield i, probs
        time.sleep(SLEEP_BETWEEN_BATCHES)


def main():
    print(f"Device: {DEVICE}")
    print("Loading VADER per-comment table (for aspect tags & metadata) ...")
    df = pd.read_csv(VADER_CSV)
    print(f"   {len(df):,} rows")

    # Resume support
    if TX_CSV.exists():
        existing = pd.read_csv(TX_CSV)
        already_scored = set(existing["comment_id"].astype(str))
        print(f"   resuming: {len(already_scored):,} comments already scored")
        df_remaining = df[~df["comment_id"].astype(str).isin(already_scored)].reset_index(drop=True)
    else:
        existing = None
        df_remaining = df.copy()

    print(f"   to score: {len(df_remaining):,} comments")
    if len(df_remaining) == 0:
        print("   nothing to do — proceeding to aggregation.")
        scored_df = existing
    else:
        print("Re-loading comment bodies from disk ...")
        body_by_id = load_comment_bodies()
        print(f"   loaded {len(body_by_id):,} bodies")
        df_remaining["body"] = df_remaining["comment_id"].map(body_by_id)
        df_remaining = df_remaining[df_remaining["body"].notna()].reset_index(drop=True)
        print(f"   {len(df_remaining):,} comments have bodies available")

        print(f"\nLoading transformer {MODEL_NAME} ...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        print(f"   loaded in {time.time()-t0:.0f}s")

        # Score, with periodic checkpoints
        texts = df_remaining["body"].tolist()
        print(f"\nScoring {len(texts):,} comments  (batch={BATCH_SIZE})  ...")
        t0 = time.time()
        all_probs = np.zeros((len(texts), 3), dtype=np.float32)
        last_checkpoint = 0
        CHECKPOINT_EVERY = 5000

        for i, probs in score_in_batches(texts, tokenizer, model):
            all_probs[i : i + len(probs)] = probs
            done = i + len(probs)
            if done - last_checkpoint >= CHECKPOINT_EVERY or done == len(texts):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(texts) - done) / rate if rate > 0 else 0
                print(f"   {done:6,d}/{len(texts):,}  "
                      f"({100*done/len(texts):5.1f}%)  "
                      f"elapsed={elapsed:5.0f}s  rate={rate:5.1f}/s  eta={eta:5.0f}s",
                      flush=True)

                # Incremental save (so we can resume if killed)
                partial = df_remaining.iloc[:done].copy()
                partial["tx_neg"] = all_probs[:done, 0]
                partial["tx_neu"] = all_probs[:done, 1]
                partial["tx_pos"] = all_probs[:done, 2]
                partial["tx_compound"] = partial["tx_pos"] - partial["tx_neg"]
                partial_to_save = partial.drop(columns=["body"], errors="ignore")
                if existing is not None:
                    partial_to_save = pd.concat([existing, partial_to_save], ignore_index=True)
                partial_to_save.to_csv(TX_CSV, index=False)
                last_checkpoint = done

        print(f"\n   total scoring time: {time.time()-t0:.0f}s")
        scored_df = pd.read_csv(TX_CSV)

    # ── Aggregate weekly ─────────────────────────────────────────────────
    print("\nAggregating weekly transformer sentiment per aspect ...")
    scored_df["week_start"] = pd.to_datetime(scored_df["date"]).dt.to_period("W").dt.start_time

    out_rows = []
    for aspect in ASPECTS:
        col = f"mentions_{aspect}"
        if col not in scored_df.columns:
            continue
        sub = scored_df[scored_df[col].astype(bool)]
        weekly = sub.groupby("week_start").agg(
            n_comments=("tx_compound", "count"),
            mean_compound=("tx_compound", "mean"),
            median_compound=("tx_compound", "median"),
            mean_pos=("tx_pos", "mean"),
            mean_neg=("tx_neg", "mean"),
        ).reset_index()
        weekly["aspect"] = aspect
        out_rows.append(weekly)

    weekly_df = pd.concat(out_rows, ignore_index=True)
    weekly_df.to_csv(TX_WEEKLY_CSV, index=False)
    print(f"   saved → {TX_WEEKLY_CSV}")

    # ── Plot the TX trajectory chart ─────────────────────────────────────
    print("\nPlotting transformer-based trajectory ...")
    events = json.loads(EVENTS_FILE.read_text())
    plot_trajectory(weekly_df, events, TX_CHART,
                    title="How a championship fractured a fandom",
                    subtitle="Per-aspect sentiment in r/formula1 discussion, 2024-2025  ·  transformer-validated")

    # ── Plot VADER vs Transformer comparison ────────────────────────────
    vader_weekly = pd.read_csv(ANALYSIS_DIR / "sentiment_per_aspect_weekly.csv")
    vader_weekly["week_start"] = pd.to_datetime(vader_weekly["week_start"])
    weekly_df["week_start"] = pd.to_datetime(weekly_df["week_start"])
    plot_comparison(vader_weekly, weekly_df, events, COMPARE_CHART)

    # ── Print summary ────────────────────────────────────────────────────
    print_summary(scored_df, events)


# ─── Plotting helpers ────────────────────────────────────────────────────────

def plot_trajectory(weekly_df, events, out_path, title, subtitle):
    colors = {
        "Norris":     "#FF8000",
        "Piastri":    "#1E88E5",
        "McLaren":    "#222222",
        "Verstappen": "#D32F2F",
        "Team mgmt":  "#888888",
    }
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for aspect in ASPECTS:
        sub = weekly_df[weekly_df["aspect"] == aspect].sort_values("week_start")
        sub = sub[sub["n_comments"] >= 10]
        if len(sub) < 2:
            continue
        sub = sub.copy()
        sub["smoothed"] = sub["mean_compound"].rolling(3, min_periods=1, center=True).mean()
        ax1.plot(sub["week_start"], sub["smoothed"],
                 label=aspect, color=colors[aspect], linewidth=2.2, alpha=0.9)
        ax1.scatter(sub["week_start"], sub["mean_compound"],
                    color=colors[aspect], s=8, alpha=0.25)

    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Weekly mean transformer sentiment\n(–1 = very neg · +1 = very pos)", fontsize=10)
    ax1.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", loc="left", pad=15)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10, ncol=5)
    ax1.grid(alpha=0.2)

    # Volume
    volume_all = weekly_df.groupby("week_start")["n_comments"].sum().reset_index()
    ax2.fill_between(volume_all["week_start"], 0, volume_all["n_comments"],
                     color="#666666", alpha=0.4, linewidth=0)
    ax2.set_ylabel("Comments / week", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(alpha=0.2)

    # Stagger event labels so they don't overlap
    for i, e in enumerate(events):
        edate = pd.to_datetime(e["date"])
        for ax in (ax1, ax2):
            ax.axvline(edate, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        short = e["race"].replace(" Grand Prix", " GP").replace(" 2024", " '24").replace(" 2025", " '25")
        y_offset = 8 + (i % 2) * 14  # stagger every other label
        ax1.annotate(
            short,
            xy=(edate, ax1.get_ylim()[1]),
            xytext=(0, y_offset), textcoords="offset points",
            rotation=30, ha="left", va="bottom",
            fontsize=8, color="#444444",
        )

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   saved chart → {out_path}")


def plot_comparison(vader_df, tx_df, events, out_path):
    """Side-by-side: VADER vs Transformer for the same aspects."""
    colors = {
        "Norris":     "#FF8000",
        "Piastri":    "#1E88E5",
        "McLaren":    "#222222",
        "Verstappen": "#D32F2F",
        "Team mgmt":  "#888888",
    }
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax, src_df, label in zip(axes, [vader_df, tx_df], ["VADER (lexicon)", "Transformer (RoBERTa)"]):
        for aspect in ASPECTS:
            sub = src_df[src_df["aspect"] == aspect].sort_values("week_start")
            sub = sub[sub["n_comments"] >= 10]
            if len(sub) < 2:
                continue
            sub = sub.copy()
            sub["smoothed"] = sub["mean_compound"].rolling(3, min_periods=1, center=True).mean()
            ax.plot(sub["week_start"], sub["smoothed"], label=aspect, color=colors[aspect], linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(f"{label}\nweekly mean sentiment", fontsize=10)
        ax.legend(loc="upper left", ncol=5, fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
        for e in events:
            ax.axvline(pd.to_datetime(e["date"]), color="black", linestyle=":", linewidth=0.6, alpha=0.3)

    axes[0].set_title(
        "Method comparison: lexicon-based vs transformer-based sentiment\n"
        "Same comments, same aggregation, two methods",
        fontsize=12, fontweight="bold", loc="left",
    )
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   saved comparison chart → {out_path}")


def print_summary(df, events):
    print("\n" + "=" * 75)
    print("  TRANSFORMER-BASED SENTIMENT — per-event summary")
    print("=" * 75)
    print(f"  {'date':<12} {'event':<30} " + " ".join(f"{a:>10s}" for a in ASPECTS))
    for e in events:
        edate = datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc)
        ws = edate.timestamp() - 7 * 86400
        we = edate.timestamp() + 7 * 86400
        sub = df[(df["created_utc"] >= ws) & (df["created_utc"] <= we)]
        cells = []
        for aspect in ASPECTS:
            col = f"mentions_{aspect}"
            asub = sub[sub[col].astype(bool)] if col in sub.columns else pd.DataFrame()
            if len(asub) >= 5:
                cells.append(f"{asub['tx_compound'].mean():+.3f}")
            else:
                cells.append("    —")
        race_short = e["race"][:28]
        print(f"  {e['date']:<12} {race_short:<30} " + " ".join(f"{c:>10s}" for c in cells))


if __name__ == "__main__":
    main()
