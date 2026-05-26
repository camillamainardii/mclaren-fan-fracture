"""
Analysis 1: Aspect-oriented sentiment trajectory.

For each comment in 2024-2025, we:
  1. Compute VADER sentiment (compound score in [-1, 1])
  2. Detect which "aspect" entities are mentioned in the comment text
     (Norris, Piastri, McLaren, Verstappen as control, McLaren leadership)
  3. Aggregate sentiment per (aspect × week) into a time series
  4. Plot the trajectory with the 10 flashpoint events annotated

Outputs:
  data/analysis/sentiment_per_aspect_weekly.csv      — the aggregated data
  data/analysis/sentiment_per_aspect_comments.csv    — per-comment sentiment + aspects
  data/analysis/chart_sentiment_trajectory.png       — the headline chart
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)
POSTS_FILE = DATA_DIR / "posts.json"
EVENTS_FILE = DATA_DIR / "flashpoint_events.json"

# ─── Aspect definitions ──────────────────────────────────────────────────────
# Map each aspect to a regex that matches its mentions (case-insensitive).
# Order matters: we check each comment for each aspect independently, so a
# single comment can be tagged with multiple aspects.

ASPECTS = {
    "Norris":     re.compile(r"\b(lando|norris|landos)\b", re.IGNORECASE),
    "Piastri":    re.compile(r"\b(oscar|piastri|piastris)\b", re.IGNORECASE),
    "McLaren":    re.compile(r"\b(mclaren|papaya|mcl3[89])\b", re.IGNORECASE),
    "Verstappen": re.compile(r"\b(max|verstappen|verstappens)\b", re.IGNORECASE),
    "Team mgmt":  re.compile(r"\b(stella|zak brown|brown|team principal)\b", re.IGNORECASE),
}

# ─── Load data ───────────────────────────────────────────────────────────────

def load_posts_by_id():
    posts = json.loads(POSTS_FILE.read_text())
    return {p["id"]: p for p in posts}


def iter_comments():
    """Yield (post_id, comment_dict) tuples across all stored comment files."""
    for f in COMMENTS_DIR.glob("*.json"):
        pid = f.stem
        try:
            for c in json.loads(f.read_text()):
                yield pid, c
        except Exception:
            continue


# ─── Sentiment + aspect tagging ──────────────────────────────────────────────

def compute_per_comment():
    """Build a DataFrame: one row per comment, with VADER score + per-aspect flags."""
    sia = SentimentIntensityAnalyzer()
    rows = []
    posts = load_posts_by_id()

    skipped_no_text = 0
    skipped_deleted = 0

    for pid, c in iter_comments():
        body = c.get("body") or ""
        author = c.get("author") or ""
        ts = c.get("created_utc")
        if not body or not ts:
            skipped_no_text += 1
            continue
        if body in ("[deleted]", "[removed]") or author in ("[deleted]", "AutoModerator"):
            skipped_deleted += 1
            continue

        # Filter to 2024-2025 only
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.year not in (2024, 2025):
            continue

        # VADER sentiment
        sentiment = sia.polarity_scores(body)
        compound = sentiment["compound"]  # in [-1, 1]

        # Aspect detection
        aspect_flags = {a: bool(rx.search(body)) for a, rx in ASPECTS.items()}

        rows.append({
            "post_id": pid,
            "comment_id": c.get("id"),
            "author": author,
            "created_utc": ts,
            "date": dt.date(),
            "year_week": f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}",
            "compound": compound,
            "vader_pos": sentiment["pos"],
            "vader_neg": sentiment["neg"],
            "vader_neu": sentiment["neu"],
            "comment_score": c.get("score") or 0,
            "length": len(body),
            **{f"mentions_{a}": v for a, v in aspect_flags.items()},
        })

    print(f"   Skipped (no text): {skipped_no_text}")
    print(f"   Skipped (deleted): {skipped_deleted}")
    print(f"   Kept (2024-2025):  {len(rows)}")
    return pd.DataFrame(rows)


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate_weekly(df):
    """For each aspect, compute weekly mean sentiment and volume.
    Returns long-format DataFrame: (aspect, year_week, n_comments, mean_compound)."""
    df["week_start"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time

    out_rows = []
    for aspect in ASPECTS:
        mask = df[f"mentions_{aspect}"]
        sub = df[mask]
        weekly = sub.groupby("week_start").agg(
            n_comments=("compound", "count"),
            mean_compound=("compound", "mean"),
            median_compound=("compound", "median"),
            std_compound=("compound", "std"),
        ).reset_index()
        weekly["aspect"] = aspect
        out_rows.append(weekly)
    weekly_df = pd.concat(out_rows, ignore_index=True)
    return weekly_df


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_trajectory(weekly_df, events, out_path):
    """The headline chart: weekly sentiment per aspect, with events annotated."""
    aspects = list(ASPECTS.keys())
    colors = {
        "Norris":     "#FF8000",  # papaya orange — Norris
        "Piastri":    "#1E88E5",  # blue — Piastri (his helmet)
        "McLaren":    "#222222",  # near-black — the team
        "Verstappen": "#D32F2F",  # red — Verstappen (control)
        "Team mgmt":  "#888888",  # gray — leadership
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: sentiment trajectories
    for aspect in aspects:
        sub = weekly_df[weekly_df["aspect"] == aspect].sort_values("week_start")
        # Only plot weeks with enough comments to be meaningful
        sub = sub[sub["n_comments"] >= 10]
        if len(sub) < 2:
            continue
        # Smooth slightly with rolling mean for readability
        sub = sub.copy()
        sub["smoothed"] = sub["mean_compound"].rolling(3, min_periods=1, center=True).mean()
        ax1.plot(sub["week_start"], sub["smoothed"],
                 label=aspect, color=colors[aspect], linewidth=2.2, alpha=0.9)
        ax1.scatter(sub["week_start"], sub["mean_compound"],
                    color=colors[aspect], s=8, alpha=0.3)

    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Weekly mean VADER compound sentiment\n(–1 = very negative · +1 = very positive)",
                   fontsize=10)
    ax1.set_title(
        "How a championship fractured a fandom\n"
        "Per-aspect sentiment in r/formula1 discussion, 2024-2025",
        fontsize=14, fontweight="bold", loc="left", pad=15,
    )
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10, ncol=5)
    ax1.grid(alpha=0.2)
    ax1.set_ylim(-0.5, 0.5)

    # Bottom: discussion volume (all aspects combined for context)
    volume_all = weekly_df.groupby("week_start")["n_comments"].sum().reset_index()
    ax2.fill_between(volume_all["week_start"], 0, volume_all["n_comments"],
                     color="#666666", alpha=0.4, linewidth=0)
    ax2.set_ylabel("Comments / week\n(sum across all aspects)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(alpha=0.2)

    # Event annotations on both axes
    for e in events:
        edate = pd.to_datetime(e["date"])
        for ax in (ax1, ax2):
            ax.axvline(edate, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        # Place label above the chart, rotated
        short = e["race"].replace(" Grand Prix", " GP").replace(" 2024", " '24").replace(" 2025", " '25")
        ax1.annotate(
            short,
            xy=(edate, ax1.get_ylim()[1]),
            xytext=(0, 4), textcoords="offset points",
            rotation=45, ha="left", va="bottom",
            fontsize=7.5, color="#555555",
        )

    # Format x-axis dates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   Saved chart → {out_path}")


# ─── Summary stats for the slides ────────────────────────────────────────────

def print_summary(df, weekly_df, events):
    print("\n" + "=" * 75)
    print("  SUMMARY STATISTICS")
    print("=" * 75)

    print(f"\nTotal comments analyzed:    {len(df):,}")
    print(f"Unique commenters:          {df['author'].nunique():,}")
    print(f"Date range:                 {df['date'].min()} → {df['date'].max()}")

    print("\nComments per aspect (overlap allowed):")
    for aspect in ASPECTS:
        n = df[f"mentions_{aspect}"].sum()
        mean = df.loc[df[f"mentions_{aspect}"], "compound"].mean()
        print(f"  {aspect:12s}  {n:7,d} comments   mean sentiment: {mean:+.3f}")

    # Sentiment around each event
    print("\nMean sentiment per aspect within ±7 days of each flashpoint:")
    print(f"  {'date':<12} {'event':<30} " + " ".join(f"{a:>10s}" for a in ASPECTS))
    for e in events:
        edate = datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc)
        window_start = edate.timestamp() - 7 * 86400
        window_end = edate.timestamp() + 7 * 86400
        window_df = df[(df["created_utc"] >= window_start) & (df["created_utc"] <= window_end)]
        means = []
        for aspect in ASPECTS:
            sub = window_df[window_df[f"mentions_{aspect}"]]
            if len(sub) >= 5:
                means.append(f"{sub['compound'].mean():+.3f}")
            else:
                means.append("    —")
        race_short = e["race"][:28]
        print(f"  {e['date']:<12} {race_short:<30} " + " ".join(f"{m:>10s}" for m in means))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading data and computing per-comment sentiment ...")
    df = compute_per_comment()
    if len(df) == 0:
        print("No comments to analyze!")
        return

    df.to_csv(ANALYSIS_DIR / "sentiment_per_aspect_comments.csv", index=False)
    print(f"   Saved {len(df):,} per-comment rows → sentiment_per_aspect_comments.csv")

    print("\nAggregating weekly sentiment per aspect ...")
    weekly_df = aggregate_weekly(df)
    weekly_df.to_csv(ANALYSIS_DIR / "sentiment_per_aspect_weekly.csv", index=False)
    print(f"   Saved weekly aggregates → sentiment_per_aspect_weekly.csv")

    print("\nPlotting trajectory chart ...")
    events = json.loads(EVENTS_FILE.read_text())
    plot_trajectory(weekly_df, events, ANALYSIS_DIR / "chart_sentiment_trajectory.png")

    print_summary(df, weekly_df, events)


if __name__ == "__main__":
    main()
