"""
First-pass exploratory analysis of the McLaren/Papaya Rules Reddit dataset.

What it answers:
  - How much data do we have? (posts, comments, unique commenters, date range)
  - When are people posting? (timeline by month + event anchors)
  - What words dominate the discussion? (raw frequencies, post titles + comment bodies)
  - Which are the highest-engagement posts? (for spot-checking on-topic-ness)
  - How "tribal" does the commenter network look? (commenter overlap across posts)
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")
POSTS_FILE = DATA_DIR / "posts.json"
COMMENTS_DIR = DATA_DIR / "comments"
EVENTS_FILE = DATA_DIR / "flashpoint_events.json"

# Lightweight stop-word list. We don't pull NLTK here — for an exploratory pass,
# this is enough to clean the top-N word list.
STOPWORDS = set("""
the a an and or but of to in for on at by with from is are was were be been being
this that these those it its it's he she they we you i me my your his her their our
not no don dont can cant won wont just like get got go going one two three
have has had do does did would could should will shall may might must
about more some any all such only own same other than then so if as up out down
when where why how what who which there here now also even still really very
much many lot good bad great better best worse worst right wrong way thing things
"""
.split())

# F1-specific noise we want to ignore in top words (mentions of these don't add signal)
F1_NOISE = set("""
gp grand prix race racing f1 formula formula1 lap laps season seasons
team teams driver drivers car cars yes no get u
""".split())

STOPWORDS |= F1_NOISE

WORD_RE = re.compile(r"[A-Za-z']{3,}")  # 3+ letter words only


def load_posts():
    if not POSTS_FILE.exists():
        return []
    return json.loads(POSTS_FILE.read_text())


def iter_comments():
    """Yield (post_id, comment_dict) tuples across all stored comment files."""
    for f in COMMENTS_DIR.glob("*.json"):
        pid = f.stem
        try:
            for c in json.loads(f.read_text()):
                yield pid, c
        except Exception:
            continue


def date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).date() if ts else None


def hdr(s):
    print()
    print("=" * 75)
    print(f"  {s}")
    print("=" * 75)


def main():
    posts = load_posts()
    hdr(f"DATASET OVERVIEW")
    print(f"Posts collected:       {len(posts):,}")
    n_comment_files = sum(1 for _ in COMMENTS_DIR.glob("*.json"))
    print(f"Comment files written: {n_comment_files:,}")

    # ── Post date distribution ───────────────────────────────────────────
    by_year_month = Counter()
    by_year = Counter()
    for p in posts:
        d = date(p.get("created_utc"))
        if d:
            by_year_month[(d.year, d.month)] += 1
            by_year[d.year] += 1

    hdr("POST VOLUME BY YEAR")
    for y in sorted(by_year):
        bar = "█" * min(60, by_year[y] // 5)
        print(f"  {y}  {by_year[y]:5d}  {bar}")

    hdr("POST VOLUME BY MONTH (2024-2025 only)")
    for (y, m), n in sorted(by_year_month.items()):
        if y < 2024:
            continue
        bar = "█" * min(60, n // 2)
        print(f"  {y}-{m:02d}  {n:4d}  {bar}")

    # ── Comment-level stats ──────────────────────────────────────────────
    comment_count = 0
    unique_authors = set()
    author_post_counts = defaultdict(set)  # author → set of post_ids they commented on
    for pid, c in iter_comments():
        comment_count += 1
        a = c.get("author")
        if a and a not in ("[deleted]", "AutoModerator"):
            unique_authors.add(a)
            author_post_counts[a].add(pid)

    hdr("COMMENT-LEVEL STATS")
    print(f"Total comments fetched:   {comment_count:,}")
    print(f"Unique commenter authors: {len(unique_authors):,}")
    if unique_authors:
        # Heavy-tail check: how many of the top commenters drive the conversation?
        top = sorted(author_post_counts.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"\nTop 15 commenters by # of posts they appear on (proxy for engagement):")
        for a, posts_set in top[:15]:
            print(f"  {a:30s}  {len(posts_set):4d} posts")

    # ── Top posts by engagement ──────────────────────────────────────────
    hdr("TOP 25 POSTS BY COMMENT COUNT (spot-check on-topic-ness)")
    top_posts = sorted(posts, key=lambda p: p.get("num_comments", 0) or 0, reverse=True)[:25]
    for p in top_posts:
        d = date(p.get("created_utc"))
        title = (p.get("title") or "").replace("\n", " ")[:75]
        flair = p.get("link_flair_text") or ""
        print(f"  [{p.get('num_comments') or 0:5d}c]  {d}  {flair[:12]:12s}  {title}")

    # ── Top words across post TITLES ─────────────────────────────────────
    hdr("TOP 50 WORDS IN POST TITLES")
    title_words = Counter()
    for p in posts:
        for w in WORD_RE.findall((p.get("title") or "").lower()):
            if w not in STOPWORDS:
                title_words[w] += 1
    for w, n in title_words.most_common(50):
        bar = "█" * min(40, n // 5)
        print(f"  {w:20s}  {n:4d}  {bar}")

    # ── Top words across COMMENT BODIES ──────────────────────────────────
    hdr("TOP 50 WORDS IN COMMENT BODIES")
    body_words = Counter()
    for _, c in iter_comments():
        body = (c.get("body") or "").lower()
        for w in WORD_RE.findall(body):
            if w not in STOPWORDS:
                body_words[w] += 1
    for w, n in body_words.most_common(50):
        bar = "█" * min(40, n // 200)
        print(f"  {w:20s}  {n:6d}  {bar}")

    # ── Posts per flashpoint event ───────────────────────────────────────
    if EVENTS_FILE.exists():
        events = json.loads(EVENTS_FILE.read_text())
        hdr("POSTS WITHIN ±7 DAYS OF EACH FLASHPOINT EVENT")
        for e in events:
            event_date = datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc)
            window_start = event_date.timestamp() - 7 * 86400
            window_end = event_date.timestamp() + 7 * 86400
            n_posts = sum(
                1 for p in posts
                if p.get("created_utc") and window_start <= p["created_utc"] <= window_end
            )
            print(f"  {e['date']}  {e['race']:35s}  {n_posts:4d} posts  | {e['note'][:50]}")


if __name__ == "__main__":
    main()
