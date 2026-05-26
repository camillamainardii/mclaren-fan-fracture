"""
Streamlit dashboard for the McLaren fan polarization study.

Run with:
  streamlit run dashboard.py

Browser opens at http://localhost:8501
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ANALYSIS_DIR = DATA_DIR / "analysis"

ASPECTS = ["Norris", "Piastri", "McLaren", "Verstappen", "Team mgmt"]
ASPECT_COLORS = {
    "Norris":     "#FF8000",   # papaya
    "Piastri":    "#1E88E5",   # blue
    "McLaren":    "#F5F1EA",   # off-white (flipped for dark theme)
    "Verstappen": "#D32F2F",   # red
    "Team mgmt":  "#AAAAAA",   # lighter gray for dark bg
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#0A0A0A",
        font=dict(color="#F5F1EA", family="Inter"),
        xaxis=dict(gridcolor="#2A2A2A", linecolor="#444", zerolinecolor="#444"),
        yaxis=dict(gridcolor="#2A2A2A", linecolor="#444", zerolinecolor="#444"),
        colorway=["#FF8000", "#1E88E5", "#F5F1EA", "#D32F2F", "#AAAAAA"],
    )
)


st.set_page_config(
    page_title="How a championship fractured a fandom",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Brand CSS injection ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@700;900&family=Inter:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Page background gradient */
.stApp {
  background: linear-gradient(180deg, #0A0A0A 0%, #131313 100%);
}

/* Headings — Saira Condensed, papaya accent on h1 */
h1, h2, h3, h4 {
  font-family: 'Saira Condensed', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: -0.01em;
}
h1 {
  font-weight: 900 !important;
  font-size: 48px !important;
  color: #F5F1EA !important;
  line-height: 1 !important;
  margin-bottom: 0 !important;
}
h2 { font-weight: 800 !important; color: #FF8000 !important; font-size: 28px !important; margin-top: 24px !important; }
h3 { font-weight: 700 !important; color: #F5F1EA !important; font-size: 20px !important; }

/* Body text */
body, .stMarkdown, p, li, span, div {
  font-family: 'Inter', sans-serif;
  color: #F5F1EA;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: #131313 !important;
  border-right: 1px solid #2A2A2A;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #FF8000 !important;
  font-size: 14px !important;
  letter-spacing: 0.1em !important;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: #1A1A1A;
  border: 1px solid #2A2A2A;
  border-radius: 12px;
  padding: 16px 18px;
  border-left: 3px solid #FF8000;
}
[data-testid="stMetricValue"] {
  font-family: 'Saira Condensed', sans-serif !important;
  font-weight: 900 !important;
  color: #FF8000 !important;
  font-size: 36px !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 11px !important;
  color: #888888 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: #131313;
  padding: 8px;
  border-radius: 12px;
  border: 1px solid #2A2A2A;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: #888888;
  font-family: 'Saira Condensed', sans-serif;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 13px;
  padding: 10px 18px;
  border-radius: 8px;
}
.stTabs [aria-selected="true"] {
  background: #FF8000 !important;
  color: #0A0A0A !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
  border: 1px solid #2A2A2A;
  border-radius: 12px;
  overflow: hidden;
}

/* Info box */
.stAlert {
  background: #1A1A1A;
  border: 1px solid #2A2A2A;
  border-left: 3px solid #FF8000;
  border-radius: 12px;
}

/* Selectbox + multiselect + slider — papaya accent */
[data-baseweb="select"], [data-baseweb="slider"] {
  border-color: #2A2A2A !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] { background: #FF8000 !important; }

/* Expander */
.streamlit-expanderHeader {
  background: #1A1A1A;
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  font-family: 'Saira Condensed', sans-serif !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 14px !important;
}

/* Custom hero banner at top */
.brand-banner {
  border-left: 4px solid #FF8000;
  padding-left: 16px;
  margin-bottom: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.2em;
  color: #FF8000;
  text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─── Data loading (cached) ───────────────────────────────────────────────────

@st.cache_data
def load_posts():
    return pd.DataFrame(json.loads((DATA_DIR / "posts.json").read_text()))


@st.cache_data
def load_events():
    return json.loads((DATA_DIR / "flashpoint_events.json").read_text())


@st.cache_data
def load_weekly(source):
    """source = 'tx' or 'vader'"""
    f = ANALYSIS_DIR / ("sentiment_per_aspect_weekly_TX.csv" if source == "tx"
                       else "sentiment_per_aspect_weekly.csv")
    df = pd.read_csv(f)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@st.cache_data
def load_topics():
    return json.loads((ANALYSIS_DIR / "topics_per_event.json").read_text())


@st.cache_data
def load_communities():
    return json.loads((ANALYSIS_DIR / "communities_summary.json").read_text())


@st.cache_data
def load_users_communities():
    return pd.read_csv(ANALYSIS_DIR / "users_communities.csv")


@st.cache_data
def load_validation_sample():
    return pd.read_csv(ANALYSIS_DIR / "sentiment_validation_sample.csv")


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown('<div class="brand-banner">Web &amp; Social Media Analytics · Final Project</div>', unsafe_allow_html=True)
st.title("How a championship fractured a fandom")
st.markdown(
    "<div style='color:#888; font-size:14px; margin-top:-12px;'>"
    "Measuring McLaren's fan polarization through Reddit, 2024-2025 · "
    "<span style='color:#FF8000;'>Camilla Mainardi</span>, "
    "<span style='color:#FF8000;'>Ramna Jalil</span>, "
    "<span style='color:#FF8000;'>Bilge Yalçın</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Project at a glance")
    posts = load_posts()
    n_posts = len(posts)
    n_active = 8329  # from the community analysis
    st.metric("Posts collected", f"{n_posts:,}")
    st.metric("Comments analyzed", "143,527")
    st.metric("Unique commenters", "32,856")
    st.metric("Flashpoint events tracked", "10")
    st.markdown("---")
    st.markdown("### Data source")
    st.markdown("r/formula1 (public JSON endpoint)  \n"
                "2024-01-01 → 2025-12-31")
    st.markdown("### Methods")
    st.markdown(
        "- **Sentiment**: VADER (lexicon) + cardiffnlp/twitter-roberta (transformer)\n"
        "- **Topics**: LDA (scikit-learn)\n"
        "- **Communities**: Louvain on user co-engagement graph (networkx + python-louvain)"
    )

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_overview, tab_sentiment, tab_topics, tab_communities, tab_methods, tab_conclusions = st.tabs(
    ["📋 Overview", "📈 Sentiment trajectory", "🗂 Topics per event",
     "🕸 Communities", "🔬 Method validation", "🎯 Conclusions"]
)

# ─── Overview tab ────────────────────────────────────────────────────────────

with tab_overview:
    st.markdown("## The question")
    st.markdown(
        "> *When a sports brand makes controversial in-season management "
        "decisions, can we detect from social media alone whether its "
        "fanbase has fractured — and pinpoint when, where, and how severely?*"
    )
    st.markdown("## The case")
    st.markdown(
        "**McLaren F1's 2024-2025 championship.** "
        "The team's season was defined by repeated 'team orders' controversies "
        "between teammates **Lando Norris** and **Oscar Piastri**, "
        "starting with the **Hungarian GP 2024** swap that became known as "
        "*patient zero* of the McLaren fan schism."
    )
    st.markdown("## The hypothesis")
    st.markdown(
        "McLaren's fanbase, initially unified around the Norris-Piastri duo, "
        "polarized as team-order controversies accumulated. We expect to find:\n"
        "1. A measurable **sentiment divergence** between aspects (drivers vs team)\n"
        "2. **Discrete fracture events** tied to specific races\n"
        "3. **Tribal structure** in the commenter network that did not exist before mid-2024"
    )
    st.markdown("## The 10 flashpoint events")
    events = load_events()
    edf = pd.DataFrame(events)
    st.dataframe(edf, use_container_width=True, hide_index=True)


# ─── Sentiment trajectory tab ────────────────────────────────────────────────

with tab_sentiment:
    st.markdown("## Per-aspect sentiment over time")
    st.markdown(
        "Aggregated weekly. Each aspect = comments that mention the entity. "
        "Verstappen included as a **cross-team control variable**."
    )

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        source = st.radio(
            "Sentiment method",
            options=["transformer (RoBERTa)", "VADER (lexicon)"],
            index=0,
            help="Transformer is more accurate, especially for sarcasm / negation. "
                 "VADER over-detects positive sentiment.",
        )
    with col2:
        smooth = st.slider("Smoothing window (weeks)", 1, 8, 3)
    with col3:
        selected_aspects = st.multiselect(
            "Aspects to display",
            options=ASPECTS,
            default=ASPECTS,
        )

    weekly = load_weekly("tx" if source.startswith("transformer") else "vader")
    events = load_events()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.05,
        subplot_titles=("Weekly mean sentiment per aspect", "Comments / week (volume)"),
    )

    for aspect in selected_aspects:
        sub = weekly[(weekly["aspect"] == aspect) & (weekly["n_comments"] >= 10)].sort_values("week_start")
        if len(sub) < 2:
            continue
        sub = sub.copy()
        sub["smoothed"] = sub["mean_compound"].rolling(smooth, min_periods=1, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=sub["week_start"], y=sub["smoothed"],
                mode="lines", name=aspect,
                line=dict(color=ASPECT_COLORS[aspect], width=2.5),
                hovertemplate="%{x|%Y-%m-%d}<br>" + aspect + ": %{y:+.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Volume
    vol = weekly.groupby("week_start")["n_comments"].sum().reset_index()
    fig.add_trace(
        go.Bar(x=vol["week_start"], y=vol["n_comments"],
               marker_color="#FF8000", marker_line_width=0, opacity=0.65, showlegend=False,
               hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} comments<extra></extra>"),
        row=2, col=1,
    )

    # Event markers
    for e in events:
        edate = pd.to_datetime(e["date"])
        for row in (1, 2):
            fig.add_vline(x=edate, line=dict(color="black", dash="dot", width=1),
                          row=row, col=1, opacity=0.4)
        fig.add_annotation(
            x=edate, y=1.04, yref="y domain",
            text=e["race"].replace(" Grand Prix", " GP"),
            showarrow=False, font=dict(size=9, color="#444"),
            textangle=-25, xanchor="left", row=1, col=1,
        )

    fig.add_hline(y=0, line=dict(color="#F5F1EA", width=0.5), opacity=0.3, row=1, col=1)
    fig.update_yaxes(title_text="mean sentiment", row=1, col=1,
                     gridcolor="#2A2A2A", linecolor="#444", color="#F5F1EA")
    fig.update_yaxes(title_text="comments", row=2, col=1,
                     gridcolor="#2A2A2A", linecolor="#444", color="#F5F1EA")
    fig.update_xaxes(gridcolor="#2A2A2A", linecolor="#444", color="#F5F1EA")
    fig.update_layout(
        height=650, hovermode="x unified",
        paper_bgcolor="#0A0A0A", plot_bgcolor="#0A0A0A",
        font=dict(color="#F5F1EA", family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#F5F1EA")),
        margin=dict(t=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Per-event mean sentiment (±7 days)")
    # Build a per-event table
    rows = []
    for e in events:
        edate = pd.to_datetime(e["date"])
        ws = edate - pd.Timedelta(days=7)
        we = edate + pd.Timedelta(days=7)
        row = {"date": e["date"], "event": e["race"]}
        for aspect in ASPECTS:
            sub = weekly[(weekly["aspect"] == aspect)
                         & (weekly["week_start"] >= ws)
                         & (weekly["week_start"] <= we)]
            row[aspect] = sub["mean_compound"].mean() if len(sub) > 0 else None
        rows.append(row)
    event_df = pd.DataFrame(rows)
    st.dataframe(
        event_df.style.format({a: "{:+.3f}" for a in ASPECTS})
                     .background_gradient(subset=ASPECTS, cmap="RdYlGn", vmin=-0.5, vmax=0.5),
        use_container_width=True, hide_index=True,
    )


# ─── Topics tab ──────────────────────────────────────────────────────────────

with tab_topics:
    st.markdown("## Topics around each flashpoint")
    st.markdown(
        "LDA (Latent Dirichlet Allocation), K=5 topics, "
        "fit on comments within ±7 days of each event."
    )
    topics_data = load_topics()
    event_options = {f"{r['event']['date']} — {r['event']['race']}": i
                     for i, r in enumerate(topics_data) if r.get("topics")}
    choice = st.selectbox("Choose a flashpoint event", list(event_options.keys()))
    r = topics_data[event_options[choice]]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Comments in window", f"{r['n_comments']:,}")
        st.metric("Unique commenters", f"{r['n_unique_authors']:,}")
        st.markdown(f"**Event note:** *{r['event']['note']}*")

    with col2:
        topic_summary = pd.DataFrame([
            {"Topic": f"T{t['topic_id']}",
             "Top terms": ", ".join(t["top_terms"][:6]),
             "Comments": t["n_dominant_docs"],
             "Mean sentiment": t["mean_sentiment"]}
            for t in r["topics"]
        ])
        st.dataframe(
            topic_summary.style.format({"Mean sentiment": "{:+.3f}"})
                              .background_gradient(subset=["Mean sentiment"], cmap="RdYlGn",
                                                   vmin=-0.3, vmax=0.3),
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Topic deep-dive")
    for t in r["topics"]:
        sent = t["mean_sentiment"]
        emoji = "🔴" if sent < -0.05 else ("🟢" if sent > 0.15 else "🔵")
        with st.expander(f"{emoji} Topic {t['topic_id']}  "
                         f"({t['n_dominant_docs']} comments · sentiment {sent:+.3f})  "
                         f"— *{', '.join(t['top_terms'][:5])}*"):
            st.markdown(f"**All top terms:** {', '.join(t['top_terms'])}")
            st.markdown("**Representative comments:**")
            for c in t["top_comments"]:
                st.markdown(f"> _{c['body']}_ — VADER {c['vader_compound']:+.2f}")


# ─── Communities tab ─────────────────────────────────────────────────────────

with tab_communities:
    st.markdown("## Commenter co-engagement communities")
    comm_data = load_communities()
    users_df = load_users_communities()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active users", f"{comm_data['n_nodes']:,}")
    col2.metric("Edges (co-engagement)", f"{comm_data.get('n_edges') or 0:,}")
    col3.metric("Communities", comm_data["n_communities_with_min_size"])
    col4.metric("Modularity", f"{comm_data['modularity']:.3f}")

    st.markdown(
        f"*Sentiment source: {comm_data.get('sentiment_source', 'unknown')}*"
    )

    st.markdown("### Per-community sentiment")
    comm_df = pd.DataFrame(comm_data["communities"]).sort_values("n_users", ascending=False)
    display_cols = ["community_id", "n_users", "total_comments",
                    "mean_sent_Norris", "mean_sent_Piastri",
                    "mean_sent_McLaren", "mean_sent_Verstappen",
                    "norris_minus_piastri"]
    sent_cols = ["mean_sent_Norris", "mean_sent_Piastri", "mean_sent_McLaren",
                 "mean_sent_Verstappen", "norris_minus_piastri"]
    st.dataframe(
        comm_df[display_cols].style.format({c: "{:+.3f}" for c in sent_cols})
                                  .background_gradient(subset=sent_cols, cmap="RdYlGn",
                                                       vmin=-0.3, vmax=0.3),
        use_container_width=True, hide_index=True,
    )

    # Bias bar chart
    bias_df = comm_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"C{int(r.community_id)} (n={int(r.n_users)})" for r in bias_df.itertuples()],
        y=bias_df["norris_minus_piastri"],
        marker_color=["#FF8000" if v > 0 else "#1E88E5" for v in bias_df["norris_minus_piastri"]],
        text=[f"{v:+.3f}" for v in bias_df["norris_minus_piastri"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.update_layout(
        title="Community polarization (positive = Norris-leaning · negative = Piastri-leaning)",
        yaxis_title="Mean sentiment difference (Norris – Piastri)",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Network visualization")
    st.image(str(ANALYSIS_DIR / "chart_community_network.png"),
             caption="Top 1,500 of 8,329 users by degree, colored by Louvain community")

    st.markdown("### Bridge users (highest betweenness)")
    bridges = users_df.nlargest(15, "betweenness_cent")[
        ["user", "community", "betweenness_cent", "n_comments",
         "sent_Norris", "sent_Piastri", "sent_McLaren", "sent_Verstappen"]
    ]
    st.dataframe(
        bridges.style.format({
            "betweenness_cent": "{:.4f}",
            "sent_Norris": "{:+.2f}",
            "sent_Piastri": "{:+.2f}",
            "sent_McLaren": "{:+.2f}",
            "sent_Verstappen": "{:+.2f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ─── Methods validation tab ──────────────────────────────────────────────────

with tab_methods:
    st.markdown("## VADER vs Transformer sentiment validation")
    st.markdown(
        "Validation sample of 5,000 comments stratified across aspects × flashpoint windows. "
        "Each comment scored with both methods."
    )
    val = load_validation_sample()

    col1, col2, col3 = st.columns(3)
    col1.metric("Sample size", f"{len(val):,}")
    correlation = val["compound"].corr(val["tx_compound"])
    col2.metric("Pearson correlation", f"{correlation:+.3f}")
    label_map = {True: "agree", False: "disagree"}
    val["vader_label"] = val["compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    agree_rate = (val["vader_label"] == val["tx_label"]).mean()
    col3.metric("3-class label agreement", f"{agree_rate:.1%}")

    st.markdown("### Scatter: VADER vs Transformer per comment")
    fig = px.scatter(
        val.sample(min(2000, len(val)), random_state=42),
        x="compound", y="tx_compound",
        opacity=0.4, height=500,
        labels={"compound": "VADER compound", "tx_compound": "Transformer compound"},
        title="Each point = one comment. Strong correlation → methods agree.",
    )
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="red", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="black", width=0.5))
    fig.add_vline(x=0, line=dict(color="black", width=0.5))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Disagreement examples — where VADER fails")
    val["disagreement"] = (val["tx_compound"] - val["compound"]).abs()
    biggest = val.nlargest(8, "disagreement")[["compound", "tx_compound", "vader_label", "tx_label", "body"]]
    for _, r in biggest.iterrows():
        body = (r["body"] or "")[:300].replace("\n", " ")
        st.markdown(
            f"**VADER {r['compound']:+.2f}** ({r['vader_label']}) vs "
            f"**Transformer {r['tx_compound']:+.2f}** ({r['tx_label']})  \n"
            f"> _{body}_"
        )


# ─── Conclusions tab ─────────────────────────────────────────────────────────

with tab_conclusions:
    st.markdown("## Findings")

    st.markdown("### Three layers of polarization evidence")
    st.markdown(
        "| Layer | Evidence | Source |\n"
        "|---|---|---|\n"
        "| **Temporal** | Sentiment toward McLaren crashes at every flashpoint while Verstappen baseline holds | Analysis 1 |\n"
        "| **Topical** | Distinct grievance topics emerge per event ('slow pit stop', 'Zak Brown', 'papaya rules') | Analysis 2 |\n"
        "| **Structural** | Mild but detectable Norris-leaning sub-tribe (community C3, n=643) with the worst team sentiment (–0.270) | Analysis 3 |\n"
    )

    st.markdown("### Key quantitative findings")
    st.markdown(
        "- **Hungarian GP 2024 = patient zero confirmed.** McLaren sentiment crashes from baseline to –0.440 in the ±7-day window.\n"
        "- **Italian GP 2025 = worst event for team management.** Team mgmt sentiment hits –0.444.\n"
        "- **The title celebration is bittersweet.** At Abu Dhabi 2025, fans celebrate the title (+0.24 on dominant topic) while McLaren team sentiment is –0.239 and Team mgmt is –0.327. *Winning didn't heal the fracture.*\n"
        "- **'Zak Brown' enters discourse as a top topic word only at US GP Sprint 2025** — first event where fans name and blame team leadership.\n"
        "- **Verstappen as control validates findings.** His sentiment stays neutral across McLaren flashpoints, proving the negativity is McLaren-specific, not general F1 mood.\n"
    )

    st.markdown("### Management implication")
    st.info(
        "McLaren's controversies created **measurable emotional ruptures** that crystallized "
        "into **specific anger topics** and produced a **detectable structural sub-tribe** of "
        "disaffected pro-Norris fans. The team won the constructors' championship while "
        "*simultaneously* losing the most ground in fan sentiment toward its leadership. "
        "For sports brand managers, the lesson is sharp: in-race communication ambiguity "
        "compounds across a season — and championship victories do not erase that debt."
    )

    st.markdown("### Methodological note")
    st.markdown(
        "All three analyses use a **transformer-based sentiment model** "
        "(cardiffnlp/twitter-roberta-base-sentiment-latest) rather than VADER, "
        "because validation showed VADER has a significant positive bias that "
        "smooths over real fan anger (Pearson correlation 0.44; "
        "VADER labels 51% positive vs transformer's 17% positive). "
        "See the *Method validation* tab for details."
    )
