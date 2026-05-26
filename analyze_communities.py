"""
Analysis 3: Commenter community detection.

Tests the core "fanbase fracture" hypothesis by asking:
  - Do commenters cluster into distinct communities based on whom they engage with?
  - Do those communities differ in their sentiment toward Norris vs Piastri?

Method:
  1. Build a bipartite graph: users ↔ posts they commented on (2024-2025)
  2. Filter to "active" users (engaged with ≥ MIN_USER_ACTIVITY posts)
     and "selective" posts (≤ MAX_USERS_PER_POST commenters) — megathreads
     with 5000+ commenters carry weaker signal of who's "really" engaged
     with what
  3. Project bipartite → user-user co-engagement graph (edge weight = # of
     posts co-engaged with)
  4. Filter edges (weight ≥ MIN_EDGE_WEIGHT)
  5. Run Louvain community detection (greedy modularity optimization)
  6. Compute centrality measures (degree, betweenness)
     → betweenness identifies "bridge users" between communities
  7. Cross-validate: per community, aggregate VADER sentiment per aspect
     → if communities differ in Norris-vs-Piastri sentiment, the
       polarization hypothesis is supported

Outputs:
  data/analysis/users_communities.csv        — per-user community + sentiment
  data/analysis/communities_summary.json     — per-community stats
  data/analysis/chart_community_network.png  — colored network graph
  data/analysis/chart_community_bias.png     — Norris-vs-Piastri bias by community

Install:
  pip3 install networkx python-louvain
"""

import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import community as community_louvain  # python-louvain
from itertools import combinations

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ANALYSIS_DIR = DATA_DIR / "analysis"
COMMENTS_CSV = ANALYSIS_DIR / "sentiment_per_aspect_comments.csv"

MIN_USER_ACTIVITY = 3        # user must have commented on ≥ N posts
MAX_USERS_PER_POST = 300     # ignore megathreads with more commenters than this
                             # (their co-engagement signal is too noisy)
MIN_EDGE_WEIGHT = 2          # two users must co-engage on ≥ N posts
TOP_NODES_FOR_VIZ = 1500     # cap nodes shown in network plot for readability
BETWEENNESS_SAMPLE = 500     # Brandes' approximation sample size (full betweenness is O(N×E))

ASPECTS = ["Norris", "Piastri", "McLaren", "Verstappen"]

# ─── Load and filter ─────────────────────────────────────────────────────────

def load_comments():
    df = pd.read_csv(COMMENTS_CSV)
    df = df[df["author"].notna()]
    df = df[~df["author"].isin(["[deleted]", "AutoModerator"])]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} comments from {df['author'].nunique():,} unique users "
          f"across {df['post_id'].nunique():,} posts")
    return df


# ─── Bipartite → unipartite projection ───────────────────────────────────────

def build_user_user_graph(df):
    """Build user co-engagement graph with the filtering rules above."""
    # 1. user → set of posts they commented on
    user_posts = df.groupby("author")["post_id"].apply(set).to_dict()

    # 2. Filter to active users
    active_users = {u for u, ps in user_posts.items() if len(ps) >= MIN_USER_ACTIVITY}
    print(f"Active users (≥{MIN_USER_ACTIVITY} posts): {len(active_users):,}")

    # 3. post → set of active users who commented on it
    post_to_users = defaultdict(set)
    for u in active_users:
        for p in user_posts[u]:
            post_to_users[p].add(u)

    # 4. Filter out megathreads (low signal-to-noise for co-engagement)
    selective_posts = {p: us for p, us in post_to_users.items()
                       if 2 <= len(us) <= MAX_USERS_PER_POST}
    print(f"Selective posts (2 ≤ active users ≤ {MAX_USERS_PER_POST}): "
          f"{len(selective_posts):,} of {len(post_to_users):,}")

    # 5. For each selective post, add edges between every pair of co-commenters
    edge_weights = defaultdict(int)
    t0 = time.time()
    for i, (p, users) in enumerate(selective_posts.items()):
        for u1, u2 in combinations(sorted(users), 2):
            edge_weights[(u1, u2)] += 1
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(selective_posts)} posts, "
                  f"{len(edge_weights):,} candidate edges  ({time.time()-t0:.0f}s)",
                  flush=True)

    # 6. Build graph with weight filter
    G = nx.Graph()
    G.add_nodes_from(active_users)
    edges_kept = 0
    for (u1, u2), w in edge_weights.items():
        if w >= MIN_EDGE_WEIGHT:
            G.add_edge(u1, u2, weight=w)
            edges_kept += 1

    print(f"Graph built: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges  (filter: weight ≥ {MIN_EDGE_WEIGHT})")

    # Keep only the largest connected component for cleaner community detection
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)
        largest = components[0]
        print(f"Largest connected component: {len(largest):,} nodes  "
              f"({100*len(largest)/G.number_of_nodes():.1f}% of total)")
        G = G.subgraph(largest).copy()

    return G


# ─── Community detection + centrality ────────────────────────────────────────

def detect_communities(G):
    print("\nRunning Louvain community detection ...")
    t0 = time.time()
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    modularity = community_louvain.modularity(partition, G, weight="weight")
    n_comms = len(set(partition.values()))
    print(f"  {n_comms} communities found, modularity = {modularity:.3f}  "
          f"({time.time()-t0:.0f}s)")
    return partition, modularity


def compute_centrality(G):
    print("\nComputing centrality measures ...")
    t0 = time.time()
    degree_cent = nx.degree_centrality(G)
    print(f"  degree centrality done  ({time.time()-t0:.0f}s)")

    # Betweenness with sampling (full is O(N×E), too slow for big graphs)
    k = min(BETWEENNESS_SAMPLE, G.number_of_nodes())
    betweenness_cent = nx.betweenness_centrality(G, k=k, weight="weight", seed=42)
    print(f"  betweenness centrality done (k={k} sample)  ({time.time()-t0:.0f}s)")
    return degree_cent, betweenness_cent


# ─── Cross-validate with sentiment ───────────────────────────────────────────

def per_user_aspect_sentiment(df, users):
    """Per user: mean VADER sentiment for each comment where they mention each aspect."""
    out = {}
    for aspect in ASPECTS:
        col = f"mentions_{aspect}"
        sub = df[df[col].astype(bool) & df["author"].isin(users)]
        means = sub.groupby("author")["compound"].mean().to_dict()
        for u, s in means.items():
            out.setdefault(u, {})[aspect] = s
    return out


# ─── Outputs ─────────────────────────────────────────────────────────────────

def build_users_csv(G, partition, degree_cent, betweenness_cent, user_sent, df):
    activity = df.groupby("author").size().to_dict()
    rows = []
    for u in G.nodes():
        sents = user_sent.get(u, {})
        rows.append({
            "user": u,
            "community": partition[u],
            "degree_cent": degree_cent.get(u, 0),
            "betweenness_cent": betweenness_cent.get(u, 0),
            "n_comments": activity.get(u, 0),
            **{f"sent_{a}": sents.get(a) for a in ASPECTS},
        })
    return pd.DataFrame(rows)


def build_community_summary(users_df, min_size=10):
    rows = []
    for cid in sorted(users_df["community"].unique()):
        sub = users_df[users_df["community"] == cid]
        if len(sub) < min_size:
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
        row["norris_minus_piastri"] = (
            row["mean_sent_Norris"] - row["mean_sent_Piastri"]
        ) if row["mean_sent_Norris"] is not None and row["mean_sent_Piastri"] is not None else None
        rows.append(row)
    return rows


def find_bridge_users(users_df, top_n=20):
    """Users with highest betweenness — they connect communities."""
    return users_df.nlargest(top_n, "betweenness_cent")[
        ["user", "community", "betweenness_cent", "n_comments", *[f"sent_{a}" for a in ASPECTS]]
    ]


def plot_network(G, partition, out_path):
    print("\nDrawing network ...")
    # Sample to top-degree nodes for readability
    if G.number_of_nodes() > TOP_NODES_FOR_VIZ:
        top = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:TOP_NODES_FOR_VIZ]
        Gs = G.subgraph(top).copy()
    else:
        Gs = G

    # Color by community
    cmap = plt.colormaps["tab20"]
    colors = [cmap(partition.get(n, 0) % 20) for n in Gs.nodes()]
    sizes = [3 + 0.05 * Gs.degree(n) for n in Gs.nodes()]

    print(f"  spring layout for {Gs.number_of_nodes():,} nodes ...")
    t0 = time.time()
    pos = nx.spring_layout(Gs, k=0.4, iterations=40, seed=42)
    print(f"  layout done  ({time.time()-t0:.0f}s)")

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_edges(Gs, pos, alpha=0.04, width=0.3, edge_color="#888888")
    nx.draw_networkx_nodes(Gs, pos, node_size=sizes, node_color=colors, alpha=0.85, linewidths=0)
    plt.title(
        f"Commenter co-engagement network\n"
        f"Top {Gs.number_of_nodes():,} of {G.number_of_nodes():,} users by degree, "
        f"colored by Louvain community",
        fontsize=13, loc="left",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved → {out_path}")


def plot_community_bias(community_summary, out_path):
    """The key validation chart: per-community Norris-vs-Piastri sentiment bias."""
    if not community_summary:
        return
    df = pd.DataFrame(community_summary).sort_values("n_users", ascending=False).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: per-aspect mean sentiment per community
    x = np.arange(len(df))
    w = 0.2
    ax1.bar(x - 1.5*w, df["mean_sent_Norris"], w, label="Norris", color="#FF8000")
    ax1.bar(x - 0.5*w, df["mean_sent_Piastri"], w, label="Piastri", color="#1E88E5")
    ax1.bar(x + 0.5*w, df["mean_sent_McLaren"], w, label="McLaren", color="#222222")
    ax1.bar(x + 1.5*w, df["mean_sent_Verstappen"], w, label="Verstappen", color="#D32F2F")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{int(r['community_id'])}\nn={int(r['n_users'])}" for _, r in df.iterrows()],
                       fontsize=8)
    ax1.set_ylabel("Mean VADER sentiment")
    ax1.set_title("Per-community mean sentiment by aspect", fontsize=11, fontweight="bold", loc="left")
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.legend(ncol=4, fontsize=8)
    ax1.grid(alpha=0.2, axis="y")

    # Right: Norris-vs-Piastri bias per community — the polarization smoking gun
    bias_df = df[df["norris_minus_piastri"].notna()].copy()
    if not bias_df.empty:
        colors = ["#FF8000" if v > 0 else "#1E88E5" for v in bias_df["norris_minus_piastri"]]
        ax2.bar(range(len(bias_df)), bias_df["norris_minus_piastri"], color=colors)
        ax2.set_xticks(range(len(bias_df)))
        ax2.set_xticklabels([f"C{int(r['community_id'])}" for _, r in bias_df.iterrows()], fontsize=8)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Norris-favorable ←   Mean sentiment difference   → Piastri-favorable")
        ax2.set_title("Community polarization\n(positive = Norris-leaning, negative = Piastri-leaning)",
                     fontsize=11, fontweight="bold", loc="left")
        ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading comment data ...")
    df = load_comments()

    print("\nBuilding user co-engagement graph ...")
    G = build_user_user_graph(df)

    partition, modularity = detect_communities(G)
    degree_cent, betweenness_cent = compute_centrality(G)

    print("\nComputing per-user aspect sentiment ...")
    user_sent = per_user_aspect_sentiment(df, set(G.nodes()))

    users_df = build_users_csv(G, partition, degree_cent, betweenness_cent, user_sent, df)
    users_df.to_csv(ANALYSIS_DIR / "users_communities.csv", index=False)
    print(f"\nSaved → users_communities.csv")

    community_summary = build_community_summary(users_df)
    (ANALYSIS_DIR / "communities_summary.json").write_text(
        json.dumps({
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "modularity": modularity,
            "n_communities_with_min_size": len(community_summary),
            "communities": community_summary,
        }, indent=2, default=str)
    )
    print(f"Saved → communities_summary.json")

    print("\n" + "=" * 75)
    print(f"  COMMUNITY DETECTION SUMMARY")
    print("=" * 75)
    print(f"\nNetwork: {G.number_of_nodes():,} users × {G.number_of_edges():,} edges")
    print(f"Modularity: {modularity:.3f}  (>0.3 = meaningful structure; >0.5 = strong)")
    print(f"Communities with ≥10 users: {len(community_summary)}\n")

    print(f"  {'C#':>3s}  {'users':>6s}  {'comments':>9s}  "
          f"{'Norris':>8s}  {'Piastri':>8s}  {'McLaren':>8s}  {'Verstap':>8s}  "
          f"{'N-P bias':>9s}")
    for c in community_summary[:15]:
        bias = c.get("norris_minus_piastri")
        bias_str = f"{bias:+.3f}" if bias is not None else "—"
        print(f"  {c['community_id']:>3d}  {c['n_users']:>6d}  {c['total_comments']:>9d}  "
              f"{c['mean_sent_Norris']:>+8.3f}  {c['mean_sent_Piastri']:>+8.3f}  "
              f"{c['mean_sent_McLaren']:>+8.3f}  {c['mean_sent_Verstappen']:>+8.3f}  "
              f"{bias_str:>9s}")

    print("\nBridge users (top 10 by betweenness centrality):")
    bridges = find_bridge_users(users_df, top_n=10)
    print(bridges.to_string(index=False))

    plot_network(G, partition, ANALYSIS_DIR / "chart_community_network.png")
    plot_community_bias(community_summary, ANALYSIS_DIR / "chart_community_bias.png")


if __name__ == "__main__":
    main()
