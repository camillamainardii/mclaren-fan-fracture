"""
McLaren / Papaya Rules / Norris-Piastri data collection from r/formula1.

Uses Reddit's public JSON endpoint (no auth needed).
Saves posts + full comment trees to local JSON for offline analysis.
Resumable: skips posts whose comments have already been fetched.

Outputs:
  data/posts.json          — all unique posts with metadata
  data/comments/{id}.json  — one file per post with its comment tree
  data/keywords_log.json   — per-keyword fetch counts (for debugging)
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

# ─── Config ──────────────────────────────────────────────────────────────────

SUBREDDIT = "formula1"
HEADERS = {"User-Agent": "web-analytics-exam-mclaren/0.2 (research; one-shot)"}
DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"
POSTS_FILE = DATA_DIR / "posts.json"
KEYWORDS_LOG = DATA_DIR / "keywords_log.json"

# Reddit search pagination caps. 100/page × ~10 pages = 1000 max per query.
PAGE_LIMIT = 100
MAX_PAGES = 10
SLEEP_BETWEEN_REQUESTS = 2.0  # polite

# Comment fetching: Reddit's comments endpoint returns top-level + nested
# replies in one call (truncated at ~500 by default). For very popular posts
# we'd need to follow "MoreComments" tokens, but for the initial dataset
# the first ~500 comments per post is plenty.
COMMENTS_LIMIT_PER_POST = 500


# ─── Keywords organized by category ──────────────────────────────────────────
# Sourced from F1-expert friend's research notes.

KEYWORDS = {
    "drivers": [
        "Norris", "Lando", "Piastri", "Oscar",
    ],
    "team_people": [
        "Andrea Stella", "Zak Brown",
    ],
    "controversy_core": [
        "papaya rules", "papaya drama", "papaya saga",
        "team orders", "swap positions", "let him through",
        "hold position", "team radio", "undercut",
    ],
    "iconic_quotes": [
        "slow pit stop", "part of racing",
        "control the controllables", "not proud about it",
        "papaya rules are over",
    ],
    "labels_nicknames": [
        "number 2 driver", "Lando bias",
        "McLaren favourite", "McLaren favorite",
        "silent assassin", "Oscar villain arc",
    ],
    "team_context": [
        "McLaren team orders", "McLaren championship",
        "Norris Piastri", "Lando Oscar",
        "MCL38", "MCL39", "WDC", "WCC",
    ],
    "race_megathreads_2024": [
        "Hungarian Grand Prix 2024",
        "Italian Grand Prix 2024",
        "Brazil Grand Prix 2024", "Brazilian Grand Prix 2024",
        "Qatar Grand Prix 2024",
    ],
    "race_megathreads_2025": [
        "Australian Grand Prix 2025",
        "Italian Grand Prix 2025",
        "Singapore Grand Prix 2025",
        "Qatar Grand Prix 2025",
        "Abu Dhabi Grand Prix 2025",
        "United States Grand Prix 2025",
        "Dutch Grand Prix 2025",
    ],
    "context_broad": [
        "McLaren",  # very broad — last so it just fills in gaps
    ],
}

# ─── Flashpoint events (for downstream event-anchored analysis) ──────────────
# Saved alongside the data so the analysis step can reference them.

FLASHPOINT_EVENTS = [
    {"date": "2024-07-21", "race": "Hungarian GP 2024",
     "note": "Patient zero. Norris-Piastri swap controversy. Fanbase split begins."},
    {"date": "2024-09-01", "race": "Italian GP 2024",
     "note": "Piastri lap-1 overtake on Norris. 'Papaya rules' enters public discourse."},
    {"date": "2024-11-02", "race": "Brazil GP Sprint 2024",
     "note": "Piastri asked to let Norris through. 'Not proud about it' quote."},
    {"date": "2024-12-01", "race": "Qatar GP 2024",
     "note": "Norris held behind Piastri in final stint."},
    {"date": "2025-03-16", "race": "Australian GP 2025",
     "note": "Papaya rules applied race 1 of new season."},
    {"date": "2025-09-07", "race": "Italian GP 2025",
     "note": "Major flashpoint. 'Slow pit stops are part of racing' quote."},
    {"date": "2025-10-05", "race": "Singapore GP 2025",
     "note": "Piastri radio disconnect. 'Oscar's villain arc' meme. 'Control the controllables'."},
    {"date": "2025-10-18", "race": "US GP Sprint 2025 (COTA)",
     "note": "Norris-Piastri collision. 'Papaya rules are finally over' fan response."},
    {"date": "2025-11-30", "race": "Qatar GP 2025",
     "note": "McLaren strategy failure. Title fight goes to final race."},
    {"date": "2025-12-07", "race": "Abu Dhabi GP 2025",
     "note": "Brown openly endorses team orders. Norris wins title. Constructors back-to-back."},
]


# ─── Data fetching ───────────────────────────────────────────────────────────

def search_subreddit(query, limit=PAGE_LIMIT, max_pages=MAX_PAGES):
    """Yield post dicts from r/formula1 search for a query, paginated."""
    after = None
    for _ in range(max_pages):
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": "all",
            "limit": limit,
        }
        if after:
            params["after"] = after
        url = f"https://www.reddit.com/r/{SUBREDDIT}/search.json"
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"   ! HTTP {r.status_code} on query={query!r}")
                return
            data = r.json()
        except Exception as e:
            print(f"   ! error on query={query!r}: {e}")
            return
        children = data.get("data", {}).get("children", [])
        if not children:
            return
        for c in children:
            yield c["data"]
        after = data.get("data", {}).get("after")
        if not after:
            return
        time.sleep(SLEEP_BETWEEN_REQUESTS)


def fetch_comments(post_id):
    """Fetch the comment tree for a post. Returns the raw JSON 'data' tree."""
    url = f"https://www.reddit.com/comments/{post_id}.json"
    params = {"limit": COMMENTS_LIMIT_PER_POST, "depth": 10, "sort": "top"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"   ! comment fetch error for {post_id}: {e}")
        return None


def flatten_comments(comment_listing):
    """Recursively flatten a Reddit comment tree into a list of comment dicts.
    Strips the heavy fields, keeps what we need for analysis."""
    out = []
    if not comment_listing or "data" not in comment_listing:
        return out
    children = comment_listing.get("data", {}).get("children", [])
    for child in children:
        if child.get("kind") != "t1":  # t1 = comment; t3 = post; "more" = MoreComments
            continue
        d = child["data"]
        out.append({
            "id": d.get("id"),
            "author": d.get("author"),
            "body": d.get("body"),
            "score": d.get("score"),
            "created_utc": d.get("created_utc"),
            "parent_id": d.get("parent_id"),
            "permalink": d.get("permalink"),
        })
        # Recurse into replies
        replies = d.get("replies")
        if isinstance(replies, dict):
            out.extend(flatten_comments(replies))
    return out


def slim_post(post):
    """Strip a Reddit post dict to the fields we care about."""
    return {
        "id": post.get("id"),
        "title": post.get("title"),
        "selftext": post.get("selftext"),
        "author": post.get("author"),
        "score": post.get("score"),
        "upvote_ratio": post.get("upvote_ratio"),
        "num_comments": post.get("num_comments"),
        "created_utc": post.get("created_utc"),
        "permalink": post.get("permalink"),
        "url": post.get("url"),
        "link_flair_text": post.get("link_flair_text"),
        "matched_keywords": [],  # filled in below
    }


# ─── Main pipeline ───────────────────────────────────────────────────────────

def collect_posts():
    """Phase 1: search for posts across all keywords, deduplicate."""
    posts_by_id = {}
    if POSTS_FILE.exists():
        print(f"Resuming from existing {POSTS_FILE} ...")
        posts_by_id = {p["id"]: p for p in json.loads(POSTS_FILE.read_text())}
        print(f"   loaded {len(posts_by_id)} existing posts")

    keyword_counts = {}
    total_keywords = sum(len(v) for v in KEYWORDS.values())
    i = 0
    for category, queries in KEYWORDS.items():
        print(f"\n[{category}]")
        for q in queries:
            i += 1
            before = len(posts_by_id)
            for post in search_subreddit(q):
                pid = post.get("id")
                if not pid:
                    continue
                if pid not in posts_by_id:
                    posts_by_id[pid] = slim_post(post)
                if q not in posts_by_id[pid]["matched_keywords"]:
                    posts_by_id[pid]["matched_keywords"].append(q)
            added = len(posts_by_id) - before
            keyword_counts[q] = added
            print(f"  ({i:2d}/{total_keywords}) {q!r:45s}  +{added:4d}  total={len(posts_by_id)}")
            # Save incrementally so we don't lose progress
            POSTS_FILE.write_text(json.dumps(list(posts_by_id.values()), indent=2))
            KEYWORDS_LOG.write_text(json.dumps(keyword_counts, indent=2))
            time.sleep(1)

    print(f"\nPhase 1 done. {len(posts_by_id)} unique posts collected.")
    return list(posts_by_id.values())


def collect_comments(posts):
    """Phase 2: fetch comment trees for each post. Resumable."""
    print(f"\nPhase 2: fetching comments for {len(posts)} posts ...")
    # Sort by comment count desc so we get the high-value posts first
    posts_sorted = sorted(posts, key=lambda p: p.get("num_comments", 0) or 0, reverse=True)
    total_comments_fetched = 0
    for i, post in enumerate(posts_sorted, 1):
        pid = post["id"]
        out_file = COMMENTS_DIR / f"{pid}.json"
        if out_file.exists():
            continue  # resume: skip already-fetched
        raw = fetch_comments(pid)
        if raw is None:
            print(f"  ({i:4d}/{len(posts_sorted)}) {pid}  FAILED")
            continue
        # raw is [post_listing, comments_listing]
        comments = flatten_comments(raw[1]) if len(raw) >= 2 else []
        out_file.write_text(json.dumps(comments, indent=2))
        total_comments_fetched += len(comments)
        title = (post.get("title") or "")[:60]
        print(f"  ({i:4d}/{len(posts_sorted)}) {pid}  {len(comments):4d} comments  | {title}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    print(f"\nPhase 2 done. {total_comments_fetched} new comments fetched across all posts.")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    COMMENTS_DIR.mkdir(exist_ok=True)
    # Save the flashpoint events file for downstream analysis
    (DATA_DIR / "flashpoint_events.json").write_text(
        json.dumps(FLASHPOINT_EVENTS, indent=2)
    )
    print("=" * 70)
    print("McLaren / Papaya Rules data collection")
    print(f"Subreddit: r/{SUBREDDIT}")
    print(f"Keywords:  {sum(len(v) for v in KEYWORDS.values())} across {len(KEYWORDS)} categories")
    print(f"Events:    {len(FLASHPOINT_EVENTS)} flashpoint races logged")
    print("=" * 70)
    posts = collect_posts()
    collect_comments(posts)
    print("\nALL DONE.")
    print(f"   posts:    {POSTS_FILE}")
    print(f"   comments: {COMMENTS_DIR}/")


if __name__ == "__main__":
    main()
