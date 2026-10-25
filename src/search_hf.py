#!/usr/bin/env python3
"""Read-only HF Hub *search* helper for competitive landscape research.

Uses the public search endpoint to discover small models in target niches
(function-calling, tool-use, structured output, extraction, command/CLI).

Usage:
    python3 src/search_hf.py "function calling" --max-params 1.5 --limit 30
    python3 src/search_hf.py --queries "tool use,json extraction,shell command"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
USER_AGENT = "hf-model-lab-research/0.1 (+read-only metadata)"

DEFAULT_QUERIES = [
    "function calling",
    "tool use",
    "tool calling",
    "json extraction",
    "structured output",
    "shell command",
    "intent classification",
    "sql generation",
    "slot filling",
]


def fetch(url: str, timeout: int = 30) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search(query: str, limit: int = 30) -> list[dict]:
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
        "direction": -1,
        "full": "false",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    try:
        raw = fetch(url)
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] {query!r}: {e!r}", file=sys.stderr)
        return []
    out = []
    for m in raw:
        tags = m.get("tags", []) or []
        out.append({
            "id": m.get("id"),
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            "lastModified": m.get("lastModified"),
            "license": next((t.split(":", 1)[1] for t in tags
                             if t.startswith("license:")), None),
            "tags": tags,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="single search query")
    ap.add_argument("--queries", help="comma-separated list of queries")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-params", type=float, default=None,
                    help="keep only rows whose name hints at <= N billion params")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    queries = ([args.query] if args.query
               else ([q.strip() for q in args.queries.split(",")] if args.queries
                     else DEFAULT_QUERIES))

    snap = {"collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": API, "results": {}}
    for q in queries:
        rows = search(q, args.limit)
        snap["results"][q] = rows
        print(f"\n=== {q} ({len(rows)}) ===")
        for r in rows[:args.limit]:
            print(f"  {r['id']:<55} dl={r['downloads']:<10} likes={r['likes']:<5} lic={r['license']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(snap, indent=2) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())