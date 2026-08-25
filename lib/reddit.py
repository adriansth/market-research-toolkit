"""
Reddit harvesting. Read-only, application-only OAuth (client credentials).

No username/password needed: we never act as a user, we only read public
listings. PRAW calls this read-only mode and it's the default flow for a
script app given only client_id + client_secret
"""

import os
import time
import json
from dataclasses import dataclass, field
import praw
import prawcore
from pathlib import Path

# Complaint grammar. These are the shapes of *unprompted* pain, not topics.
PAIN_PHRASES = [
    "i hate when",
    "there's no way to",
    "is there a way to",
    "we just use a spreadsheet",
    "still using excel",
    "wish there was",
    "manually",
    "by hand",
    "every week i have to",
    "workaround",
    "am i the only one",
    "drives me crazy",
    "waste of time",
    "how do you all handle",
    "what do you use for",
]


@dataclass
class HarvestConfig:
    subreddits: list
    queries: list
    time_filters: list = field(default_factory=lambda: ["year"])
    sorts: list = field(default_factory=lambda: ["relevance"])
    limit_per_search: int = 250 # Reddit's practical search ceiling
    comments_per_post: int = 40
    expand_more_comments: int = 0  # each expansion costs an API call; 0 = skip


def get_client(user_agent=None):
    """Build a read-only PRAW client from env vars."""
    client_id = os.environ["REDDIT_CLIENT_ID"]
    client_secret = os.environ["REDDIT_CLIENT_SECRET"]
    user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT")

    # Map missing env variables
    missing = [
        n for n, v in [
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        ] if not v
    ]

    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    # Validate Reddit's user agent format
    if "(by /u/" not in user_agent and "(by u/" not in user_agent:
        raise ValueError(
            "REDDIT_USER_AGENT must match Reddit's format:\n"
            "  <platform>:<app id>:<version> (by /u/<username>)\n"
            "e.g. python:market-research-kit:v0.1 (by /u/yourname)\n"
            "A generic user agent is the most common cause of instant 429s."
        )

    # Generate PRAW client
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    reddit.read_only = True
    return reddit


def check_auth(reddit):
    """Cheap call to check that credentials work before a long harvest."""
    sub = reddit.subreddit("redditdev")
    subscribers = sub.subscribers      # not settable locally -> forces a fetch
    return {
        "read_only": reddit.read_only,
        "subscribers": subscribers,
        "limits": reddit.auth.limits,
    }


def rate_status(reddit):
    """Live rate-limit headers. Authoritative; don't hardcode numbers."""
    return reddit.auth.limits


def normalize_record(obj, kind):
    """Normalize a PRAW submission or comment into a flat dict."""
    if kind == "submission":
        title = obj.title
        body = obj.selftext or ""
        parent_id = None
    else:
        title = None
        body = obj.body or ""
        # link_id is already on the comment; avoids a network round trip
        parent_id = getattr(obj, "link_id", None)

    try:
        author = obj.author.name if obj.author else None
    except Exception:
        author = None

    return {
        "id": obj.fullname,
        "kind": kind,
        "subreddit": str(obj.subreddit),
        "permalink": f"https://reddit.com{obj.permalink}",
        "created_utc": int(obj.created_utc),
        "score": int(getattr(obj, "score", 0) or 0),
        "parent_id": parent_id,
        "title": title,
        "body": body,
        "author": author
    }


def search_subreddit(reddit, subreddit, query, sort="relevance", time_filter="year", limit=250):
    """One search. Returns [] rather than raising on inaccessible subs."""
    output = []
    try:
        results = reddit.subreddit(subreddit).search(
            query,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
        )
        for post in results:
            normalized_post = normalize_record(post, "submission")
            output.append(normalized_post)

    except prawcore.exceptions.Redirect:
        print(f"   ! r/{subreddit} does not exist")
    except prawcore.exceptions.Forbidden:
        print(f"   ! r/{subreddit} is private or quarantined")
    except prawcore.exceptions.NotFound:
        print(f"   ! r/{subreddit} not found or banned")
    except prawcore.exceptions.TooManyRequests:
        print("   ! rate limited; sleeping 60s")
        time.sleep(60)

    return output


def fetch_comments(reddit, submission_fullname, limit=40, expand_more=0):
    """
    Pul the comment tree for one post.

    This is where the real pain lives. Posts are performative; comments are 
    where someone explains the workflow they actually hate.

    expand_more=0 drops the 'load more comments' stubs entirely (free).
    Anything higher costs onne API call per expansion.
    """
    sub_id = submission_fullname.split("_", 1)[-1]
    output = []

    try:
        submission = reddit.submission(id=sub_id)
        submission.comments.replace_more(limit=expand_more)
        for comment in submission.comments.list()[:limit]:
            normalized_comment = normalize_record(comment, "comment")
            output.append(normalized_comment)

    except prawcore.exceptions.TooManyRequests:
        print("   ! rate limited; sleeping 60s")
        time.sleep(60)
    except Exception as e:
        print(f"   ! comment fetch failed for {submission_fullname}: {str(e)}")

    return output


def harvest(reddit, cfg: HarvestConfig, verbose=True):
    """
    Run the full search matrix, deduped. Returns (submissions, comments).

    Reddit caps search at roughly 250 results per query, so we widen coverage
    by varying the *query phrasing*, *sort*, and *time window* rather than by
    trying to paginate deeper on a single query.
    """
    seen = set()
    submissions = []

    for sub in cfg.subreddits:
        for query in cfg.queries:
            for sort in cfg.sorts:
                for tf in cfg.time_filters:
                    batch = search_subreddit(
                        reddit,
                        sub,
                        query,
                        sort=sort,
                        time_filter=tf,
                        limit=cfg.limit_per_search,
                    )
                    fresh = [r for r in batch if r["id"] not in seen]
                    for r in fresh:
                        seen.add(r["id"])
                        r["found_via"] = query
                        submissions.append(r)
                    if verbose and fresh:
                        print(f"   r/{sub} [{sort}/{tf}] '{query[:40]}' "
                              f"-> +{len(fresh)} new {len(batch)} raw")
    if verbose:
        print(f"\n{len(submissions)} unique submissions. Fetching comments...")

    comments = []
    for i, s in enumerate(submissions, 1):
        comments_result = fetch_comments(
            reddit,
            s["id"],
            limit=cfg.comments_per_post,
            expand_more=cfg.expand_more_comments
        )

        for c in comments_result:
            if c["id"] not in seen:
                seen.add(c["id"])
                c["found_via"] = s["id"]
                comments.append(c)
        if verbose and i % 25 == 0:
            print(f"   {i}/{len(submissions)} posts -> {len(comments)} comments")

    return submissions, comments


def verify_alive(reddit, item_ids, batch_size=1000):
    """
    Check which IDs still exist on Reddit. 100 per call.

    Two purposes: you must drop content deleted upstream, and you don't want
    to cite a dead thread in your own report.
    """
    alive = set()
    ids = list(item_ids)

    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        try:
            for obj in reddit.info(fullnames=chunk):
                alive.add(obj.fullname)
        except Exception as e:
            print(f"   ! verify batch failed: {str(e)}")

    return alive, set(ids) - alive


def save_json(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]