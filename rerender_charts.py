"""
Re-render the analysis charts in our papaya/dark brand palette.

Outputs:
  data/analysis/chart_sentiment_trajectory_DARK.png     — sentiment chart on dark bg
  data/analysis/chart_community_bias_BRAND.png          — community bias bars, brand palette
  data/analysis/chart_community_network_DARK.png        — network on dark bg
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
from pathlib import Path

# ─── Brand palette ───────────────────────────────────────────────────────────
BLACK = "#0A0A0A"
CARD = "#1A1A1A"
PAPAYA = "#FF8000"
WHITE = "#F5F1EA"
GRAY = "#888888"
GRID = "#333333"
ASPECT_COLORS = {
    "Norris":     "#FF8000",  # papaya
    "Piastri":    "#1E88E5",  # blue
    "McLaren":    "#F5F1EA",  # off-white (was black — flipped for dark bg)
    "Verstappen": "#D32F2F",  # red
    "Team mgmt":  "#AAAAAA",  # lighter gray for dark bg
}

ANALYSIS_DIR = Path("data/analysis")
EVENTS_FILE = Path("data/flashpoint_events.json")

# ─── Sentiment trajectory — dark theme ───────────────────────────────────────

def render_sentiment_dark():
    weekly = pd.read_csv(ANALYSIS_DIR / "sentiment_per_aspect_weekly_TX.csv")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    events = json.loads(EVENTS_FILE.read_text())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
        facecolor=BLACK,
    )

    for aspect in ASPECT_COLORS:
        sub = weekly[(weekly["aspect"] == aspect) & (weekly["n_comments"] >= 10)].sort_values("week_start")
        if len(sub) < 2:
            continue
        sub = sub.copy()
        sub["smoothed"] = sub["mean_compound"].rolling(3, min_periods=1, center=True).mean()
        ax1.plot(sub["week_start"], sub["smoothed"],
                 label=aspect, color=ASPECT_COLORS[aspect], linewidth=2.5, alpha=0.95)

    ax1.set_facecolor(BLACK)
    ax1.axhline(0, color=WHITE, linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Weekly mean sentiment\n(–1 = very negative · +1 = very positive)",
                   fontsize=11, color=WHITE)
    ax1.legend(loc="upper right", framealpha=0.0, fontsize=11, ncol=5,
               labelcolor=WHITE)
    ax1.grid(alpha=0.15, color=WHITE)
    ax1.spines["bottom"].set_color(GRID)
    ax1.spines["left"].set_color(GRID)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(colors=GRAY, labelcolor=WHITE)

    # Volume
    vol = weekly.groupby("week_start")["n_comments"].sum().reset_index()
    ax2.set_facecolor(BLACK)
    ax2.fill_between(vol["week_start"], 0, vol["n_comments"],
                     color=PAPAYA, alpha=0.4, linewidth=0)
    ax2.set_ylabel("Comments / week", fontsize=10, color=WHITE)
    ax2.set_xlabel("Date", fontsize=10, color=WHITE)
    ax2.grid(alpha=0.15, color=WHITE)
    ax2.spines["bottom"].set_color(GRID)
    ax2.spines["left"].set_color(GRID)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(colors=GRAY, labelcolor=WHITE)

    # Event markers
    for i, e in enumerate(events):
        edate = pd.to_datetime(e["date"])
        for ax in (ax1, ax2):
            ax.axvline(edate, color=PAPAYA, linestyle=":", linewidth=0.9, alpha=0.5)
        short = e["race"].replace(" Grand Prix", " GP").replace(" 2024", " '24").replace(" 2025", " '25")
        ax1.annotate(
            short,
            xy=(edate, ax1.get_ylim()[1]),
            xytext=(0, 6 + (i % 2) * 14), textcoords="offset points",
            rotation=28, ha="left", va="bottom",
            fontsize=8, color=WHITE,
        )

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10, hspace=0.05)
    out = ANALYSIS_DIR / "chart_sentiment_trajectory_DARK.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.4, facecolor=BLACK)
    plt.close()
    print(f"saved → {out}")


# ─── Community bias chart — brand palette ────────────────────────────────────

def render_community_bias():
    summary = json.loads((ANALYSIS_DIR / "communities_summary.json").read_text())
    comms = sorted(summary["communities"], key=lambda c: c["n_users"], reverse=True)[:10]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), facecolor=WHITE,
                                    gridspec_kw={"height_ratios": [1.3, 1]})

    # Left: per-aspect mean sentiment per community
    x = np.arange(len(comms))
    w = 0.2
    labels = ["Norris", "Piastri", "McLaren", "Verstappen"]
    series_colors = [ASPECT_COLORS["Norris"], ASPECT_COLORS["Piastri"], "#333333", ASPECT_COLORS["Verstappen"]]
    for i, (label, color) in enumerate(zip(labels, series_colors)):
        vals = [c[f"mean_sent_{label}"] for c in comms]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=label, color=color, edgecolor="none")

    ax1.set_facecolor(WHITE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c['community_id']}\nn={c['n_users']}" for c in comms], fontsize=9)
    ax1.set_ylabel("Mean transformer sentiment", fontsize=11)
    ax1.set_title("Per-community mean sentiment by aspect",
                  fontsize=12, fontweight="bold", loc="left", color=BLACK)
    ax1.axhline(0, color=BLACK, linewidth=0.5, alpha=0.5)
    ax1.legend(ncol=4, fontsize=9, framealpha=0)
    ax1.grid(alpha=0.2, axis="y", color=GRAY)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # Right: Norris-vs-Piastri bias
    biases = [c["norris_minus_piastri"] for c in comms]
    colors = [PAPAYA if v > 0 else "#1E88E5" for v in biases]
    ax2.bar(range(len(comms)), biases, color=colors, edgecolor="none")
    ax2.set_facecolor(WHITE)
    ax2.set_xticks(range(len(comms)))
    ax2.set_xticklabels([f"C{c['community_id']}" for c in comms], fontsize=10)
    ax2.axhline(0, color=BLACK, linewidth=0.5)
    ax2.set_ylabel("← Piastri-leaning   |   Norris-leaning →", fontsize=10)
    ax2.set_title("Community polarization\nNorris sentiment – Piastri sentiment",
                  fontsize=12, fontweight="bold", loc="left", color=BLACK)
    ax2.grid(alpha=0.2, axis="y", color=GRAY)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    plt.tight_layout()
    out = ANALYSIS_DIR / "chart_community_bias_BRAND.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE, transparent=False)
    plt.close()
    # Re-save as RGB (no alpha) so the file has no transparency anywhere
    from PIL import Image as _PILImage
    _im = _PILImage.open(out)
    _bg = _PILImage.new("RGBA", _im.size, (255, 255, 255, 255))
    _PILImage.alpha_composite(_bg, _im.convert("RGBA")).convert("RGB").save(out, "PNG")
    print(f"saved → {out}")


# ─── Community network — dark theme ──────────────────────────────────────────

def render_community_network_dark():
    """Smaller, cleaner network with brand colors and dark background.
    Reuses the user→community mapping from users_communities.csv."""
    users_df = pd.read_csv(ANALYSIS_DIR / "users_communities.csv")
    # Just generate a fresh visualization using positions baked into the existing image
    # would require re-running graph build. Instead, take the existing chart and
    # just add a darkened backdrop overlay via the HTML.
    # We'll leave this one as is for now — the existing network chart works.
    pass


if __name__ == "__main__":
    render_sentiment_dark()
    render_community_bias()
    print("done")
