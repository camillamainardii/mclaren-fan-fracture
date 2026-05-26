"""
Smart comment collector — prioritized, time-capped, resumable.

Strategy:
  1. Load all posts from posts.json
  2. Filter to 2024-2025 only (our research window)
  3. Sort by num_comments DESC (megathreads first)
  4. Skip posts whose comments are already saved on disk
  5. Fetch comments with adaptive rate-limiting (slows on 429, speeds up on 200)
  6. Cap total runtime so the foreground process always returns

Run repeatedly — each run picks up where the last left off.
"""

import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"
POSTS_FILE = DATA_DIR / "posts.json"
HEADERS = {"User-Agent": "web-analytics-exam-mclaren/0.3 (research)"}

# Tuning
MIN_COMMENTS = 50          # skip noise
MAX_RUNTIME_SECONDS = 480  # 8 minutes — safe under the 10-min foreground cap
BASE_SLEEP = 1.2           # between requests when things are fine
SLEEP_ON_RATELIMIT = 10    # back off this much on 429
COMMENTS_LIMIT = 500


def flatten_comments(listing):
    out = []
    if not listing or "data" not in listing:
        return out
    for child in listing.get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        d = child["data"]
        out.append({
            "id": d.get("id"),
            "author": d.get("author"),
            "body": d.get("body"),
            "score": d.get("score"),
            "created_utc": d.get("created_utc"),
            "parent_id": d.get("parent_id"),
        })
        replies = d.get("replies")
        if isinstance(replies, dict):
            out.extend(flatten_comments(replies))
    return out


def main():
    COMMENTS_DIR.mkdir(parents=True, exist_ok=True)
    posts = json.loads(POSTS_FILE.read_text())

    # Filter to 2024-2025
    target = []
    for p in posts:
        ts = p.get("created_utc")
        if not ts:
            continue
        y = datetime.fromtimestamp(ts, tz=timezone.utc).year
        if y in (2024, 2025) and (p.get("num_comments") or 0) >= MIN_COMMENTS:
            target.append(p)
    # Sort by comment count desc
    target.sort(key=lambda p: p.get("num_comments", 0) or 0, reverse=True)

    # How many already done?
    done = {f.stem for f in COMMENTS_DIR.glob("*.json")}
    remaining = [p for p in target if p["id"] not in done]

    print(f"Target posts (2024-2025, {MIN_COMMENTS}+ comments): {len(target):,}")
    print(f"Already fetched: {len(done & set(p['id'] for p in target)):,}")
    print(f"Remaining to fetch: {len(remaining):,}")
    print(f"Runtime cap: {MAX_RUNTIME_SECONDS} seconds")
    print("=" * 70, flush=True)

    if not remaining:
        print("Nothing to fetch. Done.")
        return

    start = time.time()
    sleep_secs = BASE_SLEEP
    fetched_this_run = 0
    failed_this_run = 0
    total_comments_this_run = 0

    for i, post in enumerate(remaining, 1):
        if time.time() - start > MAX_RUNTIME_SECONDS:
            print(f"\n⏱  Runtime cap hit. Run me again to continue.", flush=True)
            break

        pid = post["id"]
        url = f"https://www.reddit.com/comments/{pid}.json"
        params = {"limit": COMMENTS_LIMIT, "depth": 10, "sort": "top"}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                print(f"  ⚠  429 rate limit. Backing off {SLEEP_ON_RATELIMIT}s ...", flush=True)
                time.sleep(SLEEP_ON_RATELIMIT)
                sleep_secs = min(sleep_secs * 1.5, 5.0)
                continue
            if r.status_code != 200:
                failed_this_run += 1
                print(f"  ({i:4d}/{len(remaining)}) {pid}  HTTP {r.status_code}", flush=True)
                time.sleep(sleep_secs)
                continue
            data = r.json()
        except Exception as e:
            failed_this_run += 1
            print(f"  ({i:4d}/{len(remaining)}) {pid}  ERROR {e}", flush=True)
            time.sleep(sleep_secs)
            continue

        comments = flatten_comments(data[1]) if len(data) >= 2 else []
        (COMMENTS_DIR / f"{pid}.json").write_text(json.dumps(comments))
        fetched_this_run += 1
        total_comments_this_run += len(comments)
        title = (post.get("title") or "")[:55]
        elapsed = time.time() - start
        print(f"  ({i:4d}/{len(remaining)}) [{elapsed:5.0f}s] {pid} {len(comments):4d}c | {title}", flush=True)

        # Gentle speed-up on success
        sleep_secs = max(BASE_SLEEP, sleep_secs * 0.95)
        time.sleep(sleep_secs)

    print(f"\n=== Run complete ===", flush=True)
    print(f"Posts fetched this run: {fetched_this_run}", flush=True)
    print(f"Comments collected:     {total_comments_this_run:,}", flush=True)
    print(f"Failed:                 {failed_this_run}", flush=True)
    print(f"Total comment files:    {sum(1 for _ in COMMENTS_DIR.glob('*.json'))}", flush=True)


if __name__ == "__main__":
    main()
