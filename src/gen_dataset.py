"""Deterministic tool-call dataset generator (v2) for hf-model-lab.

Produces reproducible JSONL splits from hand-written templates and a fixed
tool registry. Standard library only: no network, no external datasets, no ML
deps. Every tool schema is validated by ``schema_utils.validate_schema`` before
anything is written.

Splits (all free of duplicates, and *family*-disjoint by construction):

* ``train.jsonl``        – ``family="train"``, the 9 *seen* tools
* ``test_seen.jsonl``    – ``family="seen_paraphrase"``, seen tools, NEW templates+values
* ``test_unseen.jsonl``  – ``family="unseen_schema"``, tools NEVER seen in training
* ``test_notool.jsonl``  – ``family="no_tool"``, requests matching no tool
* ``test.jsonl``         – concat of the three held-out files (main harness input)
* ``test_v1_original.jsonl`` – the pre-v2 test split, kept for comparison

Usage:
    python3 src/gen_dataset.py --out data --seed 7
    python3 src/gen_dataset.py --out data --n-seen 140 --n-unseen 60 --n-notool 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_utils import NOTOOL, ToolSchemas, validate_schema  # noqa: E402

SYSTEM_PROMPT = (
    "You are a tool router. You are given a list of available tools and a "
    "user request. Reply with exactly one JSON object of the form "
    '{"name": <tool_name>, "arguments": {<argument>: <value>}}. '
    "Use only the listed tools and only their declared arguments. "
    f'If no listed tool applies, reply {{"name": "{NOTOOL}", "arguments": {{}}}}. '
    "Output JSON only, no prose."
)

# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------
# SEEN tools (present in training) deliberately cover every JSON type:
# string, integer, number, boolean and array.
SEEN_TOOLS = [
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"type": "object",
                    "properties": {"city": {"type": "string"},
                                   "unit": {"type": "string"}},
                    "required": ["city"]}},
    {"name": "send_email", "description": "Send an email message.",
     "parameters": {"type": "object",
                    "properties": {"to": {"type": "string"},
                                   "subject": {"type": "string"},
                                   "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]}},
    {"name": "create_calendar_event", "description": "Create a calendar event.",
     "parameters": {"type": "object",
                    "properties": {"title": {"type": "string"},
                                   "start": {"type": "string"},
                                   "duration_minutes": {"type": "integer"},
                                   "all_day": {"type": "boolean"}},
                    "required": ["title", "start"]}},
    {"name": "search_web", "description": "Search the web.",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"},
                                   "limit": {"type": "integer"},
                                   "safe_search": {"type": "boolean"}},
                    "required": ["query"]}},
    {"name": "convert_currency", "description": "Convert an amount between currencies.",
     "parameters": {"type": "object",
                    "properties": {"amount": {"type": "number"},
                                   "from_currency": {"type": "string"},
                                   "to_currency": {"type": "string"}},
                    "required": ["amount", "from_currency", "to_currency"]}},
    {"name": "set_timer", "description": "Start a countdown timer.",
     "parameters": {"type": "object",
                    "properties": {"seconds": {"type": "integer"},
                                   "label": {"type": "string"}},
                    "required": ["seconds"]}},
    {"name": "translate_text", "description": "Translate text into a language.",
     "parameters": {"type": "object",
                    "properties": {"text": {"type": "string"},
                                   "target_language": {"type": "string"},
                                   "formal": {"type": "boolean"}},
                    "required": ["text", "target_language"]}},
    {"name": "filter_items", "description": "Filter a product list by category and price.",
     "parameters": {"type": "object",
                    "properties": {"categories": {"type": "array",
                                                  "items": {"type": "string"}},
                                   "min_price": {"type": "number"},
                                   "in_stock_only": {"type": "boolean"}},
                    "required": ["categories", "min_price"]}},
    {"name": "tag_note", "description": "Attach tags to a note.",
     "parameters": {"type": "object",
                    "properties": {"content": {"type": "string"},
                                   "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["content", "tags"]}},
]

# UNSEEN tools: never used in training; the only place they appear is the
# unseen-schema test split (generalization check).
UNSEEN_TOOLS = [
    {"name": "play_music", "description": "Play a track at a volume.",
     "parameters": {"type": "object",
                    "properties": {"track": {"type": "string"},
                                   "volume": {"type": "integer"}},
                    "required": ["track"]}},
    {"name": "get_stock_price", "description": "Look up a share price.",
     "parameters": {"type": "object",
                    "properties": {"ticker": {"type": "string"},
                                   "on_date": {"type": "string"}},
                    "required": ["ticker"]}},
    {"name": "book_restaurant", "description": "Book a table at a restaurant.",
     "parameters": {"type": "object",
                    "properties": {"name": {"type": "string"},
                                   "party_size": {"type": "integer"},
                                   "time": {"type": "string"}},
                    "required": ["name", "party_size"]}},
    {"name": "draw_chart", "description": "Render a simple chart from values.",
     "parameters": {"type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}},
                                   "chart_type": {"type": "string"}},
                    "required": ["data", "chart_type"]}},
    {"name": "create_reminder", "description": "Create a reminder.",
     "parameters": {"type": "object",
                    "properties": {"task": {"type": "string"},
                                   "due": {"type": "string"},
                                   "priority": {"type": "integer"},
                                   "recurring": {"type": "boolean"}},
                    "required": ["task", "due"]}},
]

ALL_TOOLS = SEEN_TOOLS + UNSEEN_TOOLS

ARG_ORDER = {
    "get_weather": ["city", "unit"],
    "send_email": ["to", "subject", "body"],
    "create_calendar_event": ["title", "start", "duration_minutes", "all_day"],
    "search_web": ["query", "limit", "safe_search"],
    "convert_currency": ["amount", "from_currency", "to_currency"],
    "set_timer": ["seconds", "label"],
    "translate_text": ["text", "target_language", "formal"],
    "filter_items": ["categories", "min_price", "in_stock_only"],
    "tag_note": ["content", "tags"],
    "play_music": ["track", "volume"],
    "get_stock_price": ["ticker", "on_date"],
    "book_restaurant": ["name", "party_size", "time"],
    "draw_chart": ["data", "chart_type"],
    "create_reminder": ["task", "due", "priority", "recurring"],
}

# --------------------------------------------------------------------------
# Slot pools. Values may be scalars or {"phrase":..., "value":...} so array
# arguments can read naturally while staying structured in the gold call.
# --------------------------------------------------------------------------
TRAIN_POOLS = {
    "city": ["Berlin", "Lisbon", "Osaka", "Toronto", "Nairobi", "Lima", "Oslo", "Bogota"],
    "unit": ["celsius", "fahrenheit"],
    "to": ["sam@example.com", "ana@example.org", "ops@example.net", "kim@example.io",
           "dev@example.com", "lee@example.net", "raj@example.org", "mia@example.io"],
    "subject": ["quarterly report", "team offsite", "invoice", "design review",
                "release notes", "budget", "onboarding", "roadmap"],
    "body": ["see attached", "let's sync tomorrow", "thanks in advance",
             "please review", "ship it", "need your input", "all done", "next steps below"],
    "title": ["design review", "standup", "retro", "planning", "1:1", "demo",
              "kickoff", "postmortem"],
    "start": ["2026-11-02T09:00", "2026-11-03T14:30", "2026-11-04T08:15", "2026-11-05T17:45",
              "2026-12-01T10:00", "2026-12-05T16:45", "2027-01-09T11:30", "2027-02-14T07:00"],
    "duration_minutes": [15, 30, 45, 60, 90, 120],
    "all_day": [True, False],
    "query": ["rust async runtimes", "best pizza dough", "eur usd rate", "sqlite json1",
              "kettle descaling", "sourdough starter", "react hooks", "gpu benchmarks",
              "train timetables", "houseplant care"],
    "limit": [3, 5, 10, 20],
    "safe_search": [True, False],
    "amount": [12.5, 99, 250, 1000, 7.25, 1500, 42, 320.75],
    "from_currency": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"],
    "to_currency": ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"],
    "seconds": [45, 90, 300, 600, 900, 1200],
    "label": ["tea", "focus", "bread", "stretch", "nap", "pasta", "review", "laundry"],
    "text": ["good morning", "thank you", "see you soon", "good night",
             "congratulations", "happy birthday", "well done", "good luck"],
    "target_language": ["French", "German", "Japanese", "Spanish", "Korean", "Italian"],
    "formal": [True, False],
    "categories": [{"phrase": "books and toys", "value": ["books", "toys"]},
                   {"phrase": "shoes and hats", "value": ["shoes", "hats"]},
                   {"phrase": "laptops", "value": ["laptops"]},
                   {"phrase": "coffee and tea", "value": ["coffee", "tea"]},
                   {"phrase": "paint and brushes", "value": ["paint", "brushes"]}],
    "min_price": [5, 20, 50, 100, 250],
    "in_stock_only": [True, False],
    "content": ["buy milk", "call the dentist", "renew passport", "pay rent",
                "book flights", "fix the printer", "water plants", "update resume"],
    "tags": [{"phrase": "urgent and work", "value": ["urgent", "work"]},
             {"phrase": "personal", "value": ["personal"]},
             {"phrase": "home and errands", "value": ["home", "errands"]},
             {"phrase": "finance", "value": ["finance"]},
             {"phrase": "health and family", "value": ["health", "family"]}],
}

# Held-out value pools: share NO value with the training pools.
TEST_POOLS = {
    "city": ["Reykjavik", "Perth", "Quito", "Hanoi", "Dakar", "Tbilisi"],
    "unit": ["kelvin"],
    "to": ["zoe@example.com", "ivan@example.org", "nina@example.net"],
    "subject": ["holiday plan", "sprint recap", "receipt"],
    "body": ["happy to help", "will follow up", "no action needed"],
    "title": ["onboarding", "guild meeting", "town hall"],
    "start": ["2027-03-10T08:30", "2027-04-21T13:15"],
    "duration_minutes": [25, 75],
    "all_day": [True],
    "query": ["go fmt verb", "bike tyre pressure", "besan chilla recipe"],
    "limit": [1, 50],
    "safe_search": [False],
    "amount": [3.5, 777, 64.25],
    "from_currency": ["SEK", "NZD"],
    "to_currency": ["SEK", "NZD"],
    "seconds": [25, 1800],
    "label": ["cooldown", "proofing"],
    "text": ["good evening", "take care"],
    "target_language": ["Dutch", "Swahili"],
    "formal": [True],
    "categories": [{"phrase": "drones", "value": ["drones"]},
                   {"phrase": "cookware and utensils", "value": ["cookware", "utensils"]}],
    "min_price": [15, 400],
    "in_stock_only": [True],
    "content": ["pick up parcel", "back up laptop"],
    "tags": [{"phrase": "later and low", "value": ["later", "low"]},
             {"phrase": "travel", "value": ["travel"]}],
}

# Unseen-tool pools (only used by the unseen-schema split).
UNSEEN_POOLS = {
    "track": ["Blue Monday", "Africa", "Take Five"],
    "volume": [4, 9],
    "ticker": ["AAPL", "MSFT", "SAP"],
    "on_date": ["2026-06-01", "2026-06-30"],
    "name": ["Trattoria Roma", "Sushi Zen", "The Olive"],
    "party_size": [2, 4, 6],
    "time": ["19:00", "20:30"],
    "data": [{"phrase": "3, 7, 11", "value": [3, 7, 11]},
             {"phrase": "1.5, 2.5", "value": [1.5, 2.5]}],
    "chart_type": ["bar", "line"],
    "task": ["renew insurance", "call mom", "review PR"],
    "due": ["2026-07-01", "2026-07-15"],
    "priority": [1, 3, 5],
    "recurring": [True, False],
}

# --------------------------------------------------------------------------
# Templates (train / held-out paraphrase / unseen-tool / no-tool)
# --------------------------------------------------------------------------
TRAIN_TEMPLATES = [
    ("What's the weather in {city}?", "get_weather"),
    ("Can you check the weather for {city} please?", "get_weather"),
    ("Tell me the forecast for {city} in {unit}.", "get_weather"),
    ("Send an email to {to} about {subject} saying {body}.", "send_email"),
    ("Email {to}, subject {subject}, body {body}.", "send_email"),
    ("Draft an email to {to} with the subject {subject} and tell them {body}.", "send_email"),
    ("Book an event called {title} starting {start} lasting {duration_minutes} minutes.", "create_calendar_event"),
    ("Put {title} on my calendar at {start}.", "create_calendar_event"),
    ("Schedule {title} at {start} for {duration_minutes} minutes, all day {all_day}.", "create_calendar_event"),
    ("Search the web for {query}.", "search_web"),
    ("Look up {query} online, limit {limit} results.", "search_web"),
    ("Find me information about {query} with safe search {safe_search}.", "search_web"),
    ("Convert {amount} {from_currency} to {to_currency}.", "convert_currency"),
    ("How much is {amount} {from_currency} in {to_currency}?", "convert_currency"),
    ("Set a timer for {seconds} seconds labelled {label}.", "set_timer"),
    ("Start a {seconds} second timer named {label}.", "set_timer"),
    ("Translate {text} into {target_language}.", "translate_text"),
    ("How do you say {text} in {target_language}?", "translate_text"),
    ("Translate {text} into {target_language}, formal {formal}.", "translate_text"),
    ("Filter items in {categories} priced above {min_price}.", "filter_items"),
    ("Show {categories} over {min_price}, in stock only {in_stock_only}.", "filter_items"),
    ("Tag the note {content} with {tags}.", "tag_note"),
    ("Add the tags {tags} to {content}.", "tag_note"),
]

TEST_SEEN_TEMPLATES = [
    ("Weather check: {city}, please.", "get_weather"),
    ("Give me conditions for {city} right now.", "get_weather"),
    ("I want the current weather for {city} in {unit}.", "get_weather"),
    ("Compose a message to {to} titled {subject} that reads {body}.", "send_email"),
    ("Reply to {to} with subject {subject} and say {body}.", "send_email"),
    ("Create an appointment named {title} beginning {start} for {duration_minutes} minutes.", "create_calendar_event"),
    ("Note {title} down for {start}.", "create_calendar_event"),
    ("Find pages about {query}.", "search_web"),
    ("Pull up {query}, at most {limit} results.", "search_web"),
    ("Exchange {amount} from {from_currency} into {to_currency}.", "convert_currency"),
    ("Count down {seconds} seconds and call it {label}.", "set_timer"),
    ("Render {text} in {target_language}.", "translate_text"),
    ("Say {text} in {target_language}, please use a formal register.", "translate_text"),
    ("Narrow the catalogue to {categories} with a floor of {min_price}.", "filter_items"),
    ("Attach {tags} to the note {content}.", "tag_note"),
]

TEST_UNSEEN_TEMPLATES = [
    ("Play {track} at volume {volume}.", "play_music"),
    ("Put on the song {track}.", "play_music"),
    ("What is the share price of {ticker} on {on_date}?", "get_stock_price"),
    ("Get me the {ticker} close for {on_date}.", "get_stock_price"),
    ("Reserve a table at {name} for {party_size} at {time}.", "book_restaurant"),
    ("Book {party_size} seats at {name}.", "book_restaurant"),
    ("Draw a {chart_type} chart of {data}.", "draw_chart"),
    ("Plot these values as a {chart_type} graph: {data}.", "draw_chart"),
    ("Remind me to {task} by {due} with priority {priority}.", "create_reminder"),
    ("Add a reminder: {task}, due {due}, recurring {recurring}.", "create_reminder"),
]

# Requests that map to no available tool -> abstain.
# TRAIN_NOTOOL requests are DISJOINT from the held-out NOTOOL_TEMPLATES below, so
# abstention generalisation is genuinely tested rather than memorised.
TRAIN_NOTOOL_TEMPLATES = [
    "What is the weather like on Mars?",
    "Please send my regards to the whole team.",
    "Remind me how photosynthesis works.",
    "Book me a flight to the moon.",
    "Translate my thoughts into a symphony.",
    "Convert my coffee into motivation.",
    "Search the web for the meaning of life?",
    "Tag this moment as unforgettable.",
    "Set a timer for the end of the universe.",
    "List the files in my imagination.",
    "Summarise the last ten years of my life.",
    "Send an email to nobody in particular.",
    "Schedule a meeting with my past self.",
    "Filter my feelings by price.",
    "Play a song that does not exist.",
    "What is the stock price of happiness?",
    "Reserve a table on the sun.",
    "Draw a chart of my dreams.",
    "Remind me to forget this.",
    "Explain why cats purr.",
    "Write a poem about databases.",
    "What colour is Wednesday?",
    "How do I bake sourdough without flour?",
    "Tell me a story about a brave teapot.",
    "What is the best way to learn guitar?",
    "Give me a recipe for invisible soup.",
    "Why do we yawn?",
    "Describe the smell of rain.",
    "What is the loudest sound in the ocean?",
    "How tall is the average cloud?",
]

NOTOOL_TEMPLATES = [
    "Tell me a joke.",
    "What is the capital of France?",
    "Explain recursion like I'm five.",
    "Who won the 2014 World Cup?",
    "Write a haiku about rain.",
    "Summarize the theory of relativity.",
    "How many planets are there?",
    "What's the meaning of life?",
    "Give me a fun fact about octopuses.",
    "Count from one to ten.",
    "What is 17 times 23?",
    "Recommend a good book.",
    "Describe the water cycle.",
    "Why is the sky blue?",
    "Tell me about photosynthesis.",
    "What is a black hole?",
    "Define entropy.",
    "Sing me a song.",
    "Say something encouraging.",
    "What's your favourite colour?",
    "Explain the difference between weather and climate.",
    "Compose a limerick about cats.",
    "What is the tallest mountain?",
    "How do tides work?",
    "Give me a synonym for happy.",
    "What language do they speak in Brazil?",
    "Explain blockchain in one sentence.",
    "What's a good name for a puppy?",
    "Tell me a riddle.",
    "Describe the taste of a mango.",
    "What is the speed of light?",
    "Why do we dream?",
    "How does a rainbow form?",
    "What's the largest ocean?",
    "Give me three ideas for a birthday gift.",
    "Explain compound interest simply.",
    "What is gravity?",
    "Tell me something surprising about bees.",
    "What's the difference between a crocodile and an alligator?",
    "Write a short poem about the sea.",
]


def _placeholders(template: str) -> list[str]:
    return re.findall(r"\{(\w+)\}", template)


def _fill(template: str, pools: dict, rng: random.Random) -> tuple[str, dict]:
    text = template
    canonical: dict[str, object] = {}
    for name in _placeholders(template):
        v = rng.choice(pools[name])
        if isinstance(v, dict):
            text = text.replace("{" + name + "}", str(v["phrase"]))
            canonical[name] = v["value"]
        else:
            text = text.replace("{" + name + "}", str(v))
            canonical[name] = v
    return text, canonical


def _ex_id(utterance: str, call: dict) -> str:
    h = hashlib.sha256((utterance + json.dumps(call, sort_keys=True)).encode())
    return h.hexdigest()[:16]


def _example(template: str, tool: str, pools: dict, rng: random.Random,
             schemas: ToolSchemas, family: str) -> dict:
    utterance, slots = _fill(template, pools, rng)
    args = {k: slots[k] for k in ARG_ORDER[tool] if k in slots}
    call = {"name": tool, "arguments": args}
    return {
        "id": _ex_id(utterance, call),
        "family": family,
        "held_out_tool": tool not in {t["name"] for t in SEEN_TOOLS},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"tools": [schemas.by_name[t] for t in _context_tools(tool)],
                 "request": utterance}, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(call)},
        ],
        "gold": call,
    }


def _context_tools(tool: str) -> list[str]:
    """Distractor tools shown alongside the target (multi-tool context)."""
    names = [t["name"] for t in ALL_TOOLS]
    i = names.index(tool)
    return [names[i]] + [n for n in names if n != tool][:2]


def _notool_example(request: str, schemas: ToolSchemas) -> dict:
    call = {"name": NOTOOL, "arguments": {}}
    return {
        "id": _ex_id(request, call),
        "family": "no_tool",
        "held_out_tool": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"tools": [schemas.by_name[t] for t in ("get_weather", "send_email",
                                                        "set_timer")],
                 "request": request}, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(call)},
        ],
        "gold": call,
    }


def _gen(templates, pools, n, rng, schemas, family) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    guard = 0
    while len(rows) < n and guard < n * 200 + 500:
        guard += 1
        tmpl, tool = rng.choice(templates)
        ex = _example(tmpl, tool, pools, rng, schemas, family)
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        rows.append(ex)
    if len(rows) < n:
        print(f"[warn] {family}: produced {len(rows)}/{n} unique examples", file=sys.stderr)
    return rows


def _gen_notool(n, rng, schemas, templates=None) -> list[dict]:
    rows = [_notool_example(t, schemas) for t in (templates or NOTOOL_TEMPLATES)]
    rng.shuffle(rows)
    if len(rows) < n:
        print(f"[warn] no_tool: only {len(rows)} templates (requested {n})", file=sys.stderr)
    return rows[:n]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-seen", type=int, default=140)
    ap.add_argument("--n-unseen", type=int, default=60)
    ap.add_argument("--n-notool", type=int, default=40)
    ap.add_argument("--n-train-notool", type=int, default=90,
                    help="no_tool examples added to TRAIN (disjoint requests)")
    args = ap.parse_args(argv)

    problems = {t["name"]: validate_schema(t) for t in ALL_TOOLS}
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        print(f"[fatal] invalid tool schemas: {bad}", file=sys.stderr)
        return 2

    schemas = ToolSchemas(ALL_TOOLS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Preserve the pre-v2 test split once, for comparison.
    old_test = out / "test.jsonl"
    if old_test.exists() and not (out / "test_v1_original.jsonl").exists():
        (out / "test_v1_original.jsonl").write_text(old_test.read_text())

    rng = random.Random(args.seed)
    train = _gen(TRAIN_TEMPLATES, TRAIN_POOLS, args.n_train, rng, schemas, "train")
    # Abstention MUST be learned during training (mission requirement), using
    # requests disjoint from the held-out no_tool set.
    train += _gen_notool(args.n_train_notool, rng, schemas, TRAIN_NOTOOL_TEMPLATES)
    rng.shuffle(train)
    seen = _gen(TEST_SEEN_TEMPLATES, TEST_POOLS, args.n_seen, rng, schemas, "seen_paraphrase")
    unseen = _gen(TEST_UNSEEN_TEMPLATES, UNSEEN_POOLS, args.n_unseen, rng, schemas, "unseen_schema")
    notool = _gen_notool(args.n_notool, rng, schemas, NOTOOL_TEMPLATES)
    heldout = seen + unseen + notool

    all_ids: dict[str, str] = {}
    dupes = []
    for split_name, rows in (("train", train), ("heldout", heldout)):
        for r in rows:
            if r["id"] in all_ids:
                dupes.append((split_name, r["id"]))
            all_ids[r["id"]] = split_name
        if any(a[1] in {x["id"] for x in rows} for a in dupes):
            pass
    train_ids = {r["id"] for r in train}
    held_ids = {r["id"] for r in heldout}
    overlap = train_ids & held_ids
    dupes = [d for d in dupes if d[1] in held_ids] or dupes

    (out / "tools.json").write_text(
        json.dumps(SEEN_TOOLS + UNSEEN_TOOLS, indent=2, sort_keys=True) + "\n")
    (out / "tools_seen.json").write_text(
        json.dumps(SEEN_TOOLS, indent=2, sort_keys=True) + "\n")

    h_tr = write_jsonl(out / "train.jsonl", train)
    h_se = write_jsonl(out / "test_seen.jsonl", seen)
    h_un = write_jsonl(out / "test_unseen.jsonl", unseen)
    h_nt = write_jsonl(out / "test_notool.jsonl", notool)
    h_ho = write_jsonl(out / "test.jsonl", heldout)

    manifest = {
        "seed": args.seed,
        "n_train": len(train),
        "n_train_notool": args.n_train_notool,
        "n_heldout": len(heldout),
        "n_seen_paraphrase": len(seen),
        "n_unseen_schema": len(unseen),
        "n_no_tool": len(notool),
        "sha256": {"train": h_tr, "test": h_ho, "test_seen": h_se,
                   "test_unseen": h_un, "test_notool": h_nt},
        "seen_tools": [t["name"] for t in SEEN_TOOLS],
        "unseen_tools": [t["name"] for t in UNSEEN_TOOLS],
        "leakage_check": "fail" if overlap else "pass",
        "duplicate_ids": dupes,
        "notes": "Held-out = held-out templates AND held-out value pools; "
                 "unseen_schema uses tools absent from training entirely.",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 1 if overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())