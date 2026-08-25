"""
Token accounting for the extraction stage.

The raw corpus size is a vanity metric. What you actually pay for is:
    (surviving items after prefilter) x (truncated body) + (prompt overhead per batch)
"""

import json

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def count_tokens(text):
    """Token count for a string.

    Uses tiktoken if available. That's OpenAI's tokenizer, so for Claude it
    runs roughly 10-20% low - fine for budgeting, not for hard limits. The
    fallback is the chars/4 rule of thumb.
    """
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text) // 4


def item_text(row, max_chars=None):
    """The text you'd actually send for one item: title + body, truncated."""
    parts = []
    if row.get("title"):
        parts.append(row["title"])
    if row.get("body"):
        parts.append(row["body"])
    text = "\n".join(parts)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + " [...]"
    return text


def corpus_stats(records, max_chars=None):
    """Token totals and distribution for a set of records."""
    counts = [count_tokens(item_text(r, max_chars)) for r in records]
    counts.sort()
    n = len(counts)
    if n == 0:
        return {}

    def pct(p):
        return counts[min(int(n * p), n - 1)]

    return {
        "items": n,
        "total_tokens": sum(counts),
        "mean": round(sum(counts) / n, 1),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": counts[-1],
        # what fraction of all tokens live in the largest 1% of items
        "top1pct_share": round(sum(counts[int(n * 0.99):]) / max(sum(counts), 1), 3),
    }


def estimate_cost(records, max_chars=1200, batch_size=8, prompt_overhead=400, output_per_item=150, in_per_mtok=1.0, out_per_mtok=5.0):
    """Estimate the cost of one extraction pass.

    prompt_overhead : tokens of schema + instructions repeated per batch
    output_per_item : tokens of JSON the model writes back per item
    Prices default to a mid-tier model, in USD per million tokens.
    """
    body_tokens = sum(count_tokens(item_text(r, max_chars)) for r in records)
    n_batches = (len(records) + batch_size - 1) // batch_size

    input_tokens = body_tokens + n_batches * prompt_overhead
    output_tokens = len(records) * output_per_item

    cost_in = input_tokens / 1_000_000 * in_per_mtok
    cost_out = output_tokens / 1_000_000 * out_per_mtok

    return {
        "items": len(records),
        "batches": n_batches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_input_usd": round(cost_in, 3),
        "cost_output_usd": round(cost_out, 3),
        "cost_total_usd": round(cost_in + cost_out, 3),
    }


def truncation_curve(records, caps=(400, 800, 1200, 2000, 4000, None)):
    """How much do you save by truncating, and how many items are affected?"""
    rows = []
    full = sum(count_tokens(item_text(r)) for r in records)
    for cap in caps:
        toks = sum(count_tokens(item_text(r, cap)) for r in records)
        affected = sum(1 for r in records
                       if cap is not None and len(item_text(r)) > cap)
        rows.append({
            "max_chars": cap if cap else "none",
            "total_tokens": toks,
            "pct_of_full": round(100 * toks / max(full, 1), 1),
            "items_truncated": affected,
            "pct_items_truncated": round(100 * affected / max(len(records), 1), 1),
        })
    return rows