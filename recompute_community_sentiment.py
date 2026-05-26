"""
Recompute Analysis 3's per-community sentiment using TRANSFORMER scores
instead of VADER. Keeps the existing community structure (graph + Louvain
result) intact — only swaps the sentiment characterization.

Why: VADER has a documented positive bias (validation showed 0.44 correlation
with transformer). For consistency with Analyses 1-2 (which use transformer),
the community polarization test should use the same sentiment source.

Inputs:
  data/analysis/users_communities.csv           — user → community membership
  data/analysis/sentiment_per_aspect_comments_TX.csv — per-comment transformer scores

Outputs (overwrites VADER versions but keeps original as _VADER backup):
  data/analysis/users_communities.csv           — now with TX sentiment columns
  data/analysis/communities_summary.json        — TX-based community summary
  data/analysis/chart_community_bias.png        — TX-based bias chart
  data/analysis/users_communities_VADER.csv     — backup of VADER version
  data/analysis/communities_summary_VADER.json  — backup of VADER version
"""

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ANALYSIS_DIR = Path("data/analysis")
USERS_CSV = ANALYSIS_DIR / "users_communities.csv"
COMMUNITIES_JSON = ANALYSIS_DIR / "communities_summary.json"
TX_COMMENTS_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments_TX.csv"
BIAS_CHART = ANALYSIS_DIR / "chart_community_bias.png"

ASPECTS = ["Norris", "Piastri", "McLaren", "Verstappen"]


def main():
    print("Loading existing community assignments ...")
    users_df = pd.read_csv(USERS_CSV)
    print(f"   {len(users_df):,} users with community assignments")

    # Backup VADER versions
    print("\nBacking up VADER versions ...")
    vader_users_backup = ANALYSIS_DIR / "users_communities_VADER.csv"
    vader_comm_backup = ANALYSIS_DIR / "communities_summary_VADER.json"
    shutil.copy(USERS_CSV, vader_users_backup)
    if COMMUNITIES_JSON.exists():
        shutil.copy(COMMUNITIES_JSON, vader_comm_backup)
    print(f"   {vader_users_backup}")
    print(f"   {vader_comm_backup}")

    print("\nLoading transformer scores ...")
    tx_df = pd.read_csv(TX_COMMENTS_CSV)
    print(f"   {len(tx_df):,} transformer-scored comments")

    # Compute per-user transformer sentiment per aspect
    print("\nComputing per-user transformer sentiment per aspect ...")
    user_set = set(users_df["user"])
    tx_user_aspect = {}
    for aspect in ASPECTS:
        col = f"mentions_{aspect}"
        if col not in tx_df.columns:
            print(f"   WARNING: column {col!r} missing in TX data")
            continue
        sub = tx_df[tx_df[col].astype(bool) & tx_df["author"].isin(user_set)]
        means = sub.groupby("author")["tx_compound"].mean().to_dict()
        for u, s in means.items():
            tx_user_aspect.setdefault(u, {})[aspect] = s

    # Drop old VADER sentiment columns, add TX ones
    for aspect in ASPECTS:
        old_col = f"sent_{aspect}"
        if old_col in users_df.columns:
            users_df = users_df.rename(columns={old_col: f"sent_{aspect}_VADER"})
    for aspect in ASPECTS:
        users_df[f"sent_{aspect}"] = users_df["user"].map(
            lambda u: tx_user_aspect.get(u, {}).get(aspect)
        )
    users_df.to_csv(USERS_CSV, index=False)
    print(f"   updated → {USERS_CSV}")

    # Recompute community summary
    print("\nRebuilding community summary with transformer sentiment ...")
    community_summary = []
    for cid in sorted(users_df["community"].unique()):
        sub = users_df[users_df["community"] == cid]
        if len(sub) < 10:
            continue
        row = {
            "community_id": int(cid),
            "n_users": int(len(sub)),
            "total_comments": int(sub["n_comments"].sum()),
            "mean_degree_cent": float(sub["degree_cent"].mean()),
            "mean_betweenness_cent": float(sub["betweenness_cent"].mean()),
        }
        for a in ASPECTS:
            row[f"mean_sent_{a}"] = float(sub[f"sent_{a}"].mean())
        row["norris_minus_piastri"] = row["mean_sent_Norris"] - row["mean_sent_Piastri"]
        community_summary.append(row)

    # Pull network stats from old summary if present
    if vader_comm_backup.exists():
        old = json.loads(vader_comm_backup.read_text())
        n_nodes = old.get("n_nodes")
        n_edges = old.get("n_edges")
        modularity = old.get("modularity")
    else:
        n_nodes = len(users_df)
        n_edges = None
        modularity = None

    new_summary = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "modularity": modularity,
        "n_communities_with_min_size": len(community_summary),
        "sentiment_source": "transformer (cardiffnlp/twitter-roberta)",
        "communities": community_summary,
    }
    COMMUNITIES_JSON.write_text(json.dumps(new_summary, indent=2, default=str))
    print(f"   saved → {COMMUNITIES_JSON}")

    # Print summary
    print("\n" + "=" * 75)
    print("  COMMUNITY DETECTION — TRANSFORMER SENTIMENT")
    print("=" * 75)
    print(f"\nNetwork:    {n_nodes:,} users · {n_edges:,} edges · modularity {modularity:.3f}")
    print(f"Communities (≥10 users): {len(community_summary)}\n")
    print(f"  {'C#':>3s}  {'users':>6s}  {'Norris':>8s}  {'Piastri':>8s}  "
          f"{'McLaren':>8s}  {'Verstap':>8s}  {'N-P bias':>9s}")
    for c in community_summary:
        bias = c.get("norris_minus_piastri")
        bias_str = f"{bias:+.3f}" if bias is not None else "—"
        print(f"  {c['community_id']:>3d}  {c['n_users']:>6d}  "
              f"{c['mean_sent_Norris']:>+8.3f}  {c['mean_sent_Piastri']:>+8.3f}  "
              f"{c['mean_sent_McLaren']:>+8.3f}  {c['mean_sent_Verstappen']:>+8.3f}  "
              f"{bias_str:>9s}")

    # Replot the bias chart with TX values
    plot_community_bias(community_summary, BIAS_CHART)


def plot_community_bias(community_summary, out_path):
    df = pd.DataFrame(community_summary).sort_values("n_users", ascending=False).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: per-aspect mean transformer sentiment per community
    x = np.arange(len(df))
    w = 0.2
    ax1.bar(x - 1.5*w, df["mean_sent_Norris"], w, label="Norris", color="#FF8000")
    ax1.bar(x - 0.5*w, df["mean_sent_Piastri"], w, label="Piastri", color="#1E88E5")
    ax1.bar(x + 0.5*w, df["mean_sent_McLaren"], w, label="McLaren", color="#222222")
    ax1.bar(x + 1.5*w, df["mean_sent_Verstappen"], w, label="Verstappen", color="#D32F2F")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{int(r['community_id'])}\nn={int(r['n_users'])}" for _, r in df.iterrows()],
                       fontsize=8)
    ax1.set_ylabel("Mean transformer sentiment")
    ax1.set_title("Per-community sentiment by aspect (transformer)",
                  fontsize=11, fontweight="bold", loc="left")
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.legend(ncol=4, fontsize=8)
    ax1.grid(alpha=0.2, axis="y")

    # Right: bias
    bias_df = df[df["norris_minus_piastri"].notna()].copy()
    colors = ["#FF8000" if v > 0 else "#1E88E5" for v in bias_df["norris_minus_piastri"]]
    ax2.bar(range(len(bias_df)), bias_df["norris_minus_piastri"], color=colors)
    ax2.set_xticks(range(len(bias_df)))
    ax2.set_xticklabels([f"C{int(r['community_id'])}" for _, r in bias_df.iterrows()], fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Norris-favorable ←   Mean sentiment difference   → Piastri-favorable")
    ax2.set_title("Community polarization (transformer)\n"
                  "positive = Norris-leaning · negative = Piastri-leaning",
                  fontsize=11, fontweight="bold", loc="left")
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart → {out_path}")


if __name__ == "__main__":
    main()
