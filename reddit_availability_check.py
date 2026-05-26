"""
Quick data-availability check for the Web Analytics exam project.
Uses Reddit's public search.json endpoint — no API credentials needed.

For each candidate topic, queries r/formula1 with a few keyword variants,
deduplicates posts by ID, then reports:
  - number of unique posts found
  - total comments across those posts
  - oldest / newest post date in the result set
"""

import time
import requests
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": "web-analytics-exam-availability-check/0.1 (research; one-shot script)"
}

SUBREDDIT = "formula1"
# Reddit search.json caps at ~100/page, ~1000 total via pagination.
# For an availability *estimate* we go to 250 per query (3 pages).
PAGE_LIMIT = 100
MAX_PAGES = 3


def search_subreddit(query: str, sort: str = "relevance", time_filter: str = "all"):
    """Yield post dicts from r/formula1 search for `query`."""
    after = None
    for _ in range(MAX_PAGES):
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": sort,
            "t": time_filter,
            "limit": PAGE_LIMIT,
        }
        if after:
            params["after"] = after
        url = f"https://www.reddit.com/r/{SUBREDDIT}/search.json"
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"   ! HTTP {resp.status_code} for query={query!r}")
                return
            data = resp.json()
        except Exception as e:
            print(f"   ! request failed for query={query!r}: {e}")
            return
        children = data.get("data", {}).get("children", [])
        if not children:
            return
        for c in children:
            yield c["data"]
        after = data.get("data", {}).get("after")
        if not after:
            return
        time.sleep(2)  # be polite to the public endpoint


def summarize(topic_label: str, queries: list[str]):
    print(f"\n=== {topic_label} ===")
    posts_by_id = {}
    for q in queries:
        before = len(posts_by_id)
        for post in search_subreddit(q):
            posts_by_id[post["id"]] = post
        added = len(posts_by_id) - before
        print(f"   query {q!r:50s}  +{added:4d} new posts  (total unique: {len(posts_by_id)})")
        time.sleep(1)

    if not posts_by_id:
        print("   no posts found")
        return

    total_comments = sum(p.get("num_comments", 0) for p in posts_by_id.values())
    timestamps = [p.get("created_utc", 0) for p in posts_by_id.values() if p.get("created_utc")]
    oldest = datetime.fromtimestamp(min(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")
    newest = datetime.fromtimestamp(max(timestamps), tz=timezone.utc).strftime("%Y-%m-%d")
    avg_comments = total_comments / len(posts_by_id)
    top = sorted(posts_by_id.values(), key=lambda p: p.get("num_comments", 0), reverse=True)[:3]

    print(f"   --- SUMMARY ---")
    print(f"   unique posts:       {len(posts_by_id)}")
    print(f"   total comments:     {total_comments:,}  (avg {avg_comments:.0f} per post)")
    print(f"   date range:         {oldest}  →  {newest}")
    print(f"   top 3 posts by comment count:")
    for p in top:
        title = p["title"][:90].replace("\n", " ")
        print(f"     [{p['num_comments']:5d} comments] {title}")


TOPICS = {
    "TOPIC 1 — Hamilton at Ferrari": [
        "Hamilton Ferrari",
        "Lewis Ferrari",
        "Hamilton SF-25",
        "Hamilton Maranello",
    ],
    "TOPIC 2 — 2026 regulations": [
        "2026 regulations",
        "2026 regs",
        "2026 rules",
        "2026 engine",
        "active aero",
    ],
    "TOPIC 3 — Norris championship + Papaya Rules": [
        "Papaya Rules",
        "Norris championship",
        "McLaren team orders",
        "Piastri Norris",
    ],
}


if __name__ == "__main__":
    print(f"Querying r/{SUBREDDIT} via public search.json (no auth)")
    print(f"Per query: up to {PAGE_LIMIT * MAX_PAGES} posts")
    for label, queries in TOPICS.items():
        summarize(label, queries)
