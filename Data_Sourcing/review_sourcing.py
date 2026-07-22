"""Low-level Jikan (MyAnimeList) review fetching.

Jikan v4 is rate-limited to ~3 req/sec and ~60 req/min, so everything here
goes through a single shared session with a minimum inter-request delay and
retry/backoff on 429 / 5xx. Higher-level batching + resume lives in
fetch_reviews_api.py.
"""

import time
import requests

BASE_URL = "https://api.jikan.moe/v4"

# 60 req/min is the binding limit -> ~1 request/second sustained.
MIN_INTERVAL = 1.1
_last_request_ts = [0.0]

_session = requests.Session()
_session.headers.update({"User-Agent": "shanime-recommender/0.1"})


def _throttle():
    elapsed = time.time() - _last_request_ts[0]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_ts[0] = time.time()


def fetch_reviews(BASE_URL, params=None, value=None, media_type="anime"):
    """Fetch a single page of reviews. Kept for backwards compatibility."""
    try:
        _throttle()
        response = _session.get(
            f"{BASE_URL}/{media_type}/{value}/reviews", params=params, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews: {e}")
        return []


def _get_page(mal_id, media_type, page, max_retries=4):
    """One page with throttling + backoff. Returns parsed JSON or None."""
    url = f"{BASE_URL}/{media_type}/{mal_id}/reviews"
    params = {"page": page, "preliminary": "true", "spoilers": "true"}
    for attempt in range(max_retries):
        _throttle()
        try:
            resp = _session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None  # no reviews / bad id — don't retry
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)  # backoff and retry
            continue
        return None
    return None


def fetch_all_reviews(mal_id, media_type="anime", max_pages=2):
    """Paginate reviews for one title. Returns a list of normalized dicts.

    max_pages caps how many pages (20 reviews each) we pull per title — a
    couple of pages is plenty of signal for an emotional profile and keeps the
    full-corpus crawl from ballooning.
    """
    rows = []
    for page in range(1, max_pages + 1):
        data = _get_page(mal_id, media_type, page)
        if not data:
            break
        reviews = data.get("data", [])
        if not reviews:
            break
        for r in reviews:
            user = r.get("user") or {}
            rows.append({
                "review_id": r.get("mal_id"),
                "profile": user.get("username"),
                "score": r.get("score"),
                "text": r.get("review"),
            })
        if not data.get("pagination", {}).get("has_next_page"):
            break
    return rows
