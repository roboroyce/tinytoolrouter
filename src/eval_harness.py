"""Benchmark harness: compare candidate tool-routers on held-out data.

Reports, separately and honestly:

* ``json_parse_rate``            – raw output yielded JSON (shape only)
* ``valid_structured_output_rate`` – JSON that *also* satisfies the schema
* ``tool_selection_accuracy``    – picked the right tool name
* ``argument_accuracy``          – right tool AND all argument values equal
* ``exact_match_accuracy``       – headline metric (tool + arguments)
* ``no_tool_precision`` / ``recall`` / ``f1`` – abstention quality
* ``unseen_schema_exact_match``  – slice over held-out tool schemas

Backends
--------
``heuristic``    Deterministic keyword router (weak floor; emits JSON with
                 empty arguments on purpose). NOT the base LLM.
``gold``         Echoes the gold call. Sanity check -> all metrics 1.0.
``transformers`` Loads a local HF causal LM and prompts it with the frozen
                 system+user messages. Requires transformers+torch and local
                 weights; raises a named error otherwise.

Usage:
    python3 src/eval_harness.py --data data/test.jsonl --backend heuristic
    python3 src/eval_harness.py --data data/test.jsonl --backend transformers \
        --model Qwen/Qwen3-0.6B
    python3 src/eval_harness.py --data "data/test.jsonl,data/test_unseen.jsonl" ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_utils import NOTOOL, ToolSchemas, score_prediction  # noqa: E402

KEYWORD_ROUTER = [
    ("get_weather", ("weather", "forecast", "conditions", "temperature")),
    ("send_email", ("email", "message", "compose", "draft")),
    ("create_calendar_event", ("calendar", "schedule", "appointment", "event", "book")),
    ("search_web", ("search", "look up", "find", "pages about")),
    ("convert_currency", ("convert", "exchange", "how much is", "currency")),
    ("set_timer", ("timer", "count down", "countdown")),
    ("translate_text", ("translate", "how do you say")),
    ("list_files", ("list the files", "directory", "files in")),
]

METRIC_KEYS = ("json_parsed", "normalized", "parse_ok", "tool_correct",
               "args_correct", "exact_match", "pred_is_notool")


def load_tools(path: Path = Path("data/tools.json")) -> ToolSchemas:
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/gen_dataset.py first")
    return ToolSchemas(json.loads(path.read_text()))


def heuristic_route(request: str, schemas: ToolSchemas) -> str:
    """Keyword route; emits valid JSON with empty args (a weak, honest floor)."""
    low = request.casefold()
    best, best_hits = None, 0
    for name, keys in KEYWORD_ROUTER:
        if name not in schemas.by_name:
            continue
        hits = sum(1 for k in keys if k in low)
        if hits > best_hits:
            best, best_hits = name, hits
    return json.dumps({"name": best or NOTOOL, "arguments": {}})


def make_transformer_backend(model_id: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"backend 'transformers' unavailable ({e!r}). Install transformers+torch "
            "and provide local weights. No weights are downloaded automatically."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_utils import render_chat, resolve_model_ref

    ref = resolve_model_ref(model_id)
    if not os.path.isdir(ref) and ref == model_id and "/" not in model_id:
        raise SystemExit(f"model path {model_id!r} does not exist")
    tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ref, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()

    def infer(row: dict) -> str:
        msgs = row["messages"][:2]  # system + user only
        prompt = render_chat(tok, msgs, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return infer


def _aggregate(rows: list[dict], infer, schemas: ToolSchemas) -> dict:
    totals = {k: 0 for k in METRIC_KEYS}
    n = 0
    tp = fp = fn = 0
    failures: list[dict] = []
    for row in rows:
        pred = infer(row)
        m = score_prediction(pred, row["gold"], schemas)
        n += 1
        for k in METRIC_KEYS:
            totals[k] += int(m[k])
        if m["pred_is_notool"] and m["gold_is_notool"]:
            tp += 1
        elif m["pred_is_notool"] and not m["gold_is_notool"]:
            fp += 1
        elif not m["pred_is_notool"] and m["gold_is_notool"]:
            fn += 1
        if not m["exact_match"] and len(failures) < 15:
            failures.append({"id": row["id"], "family": row.get("family"),
                             "pred": pred[:400], "gold": row["gold"], "scored": m})
    d = max(n, 1)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (tp or fp or fn) else None)
    return {
        "n": n,
        "json_parse_rate": round(totals["json_parsed"] / d, 4),
        "valid_structured_output_rate": round(totals["parse_ok"] / d, 4),
        "tool_selection_accuracy": round(totals["tool_correct"] / d, 4),
        "argument_accuracy": round(totals["args_correct"] / d, 4),
        "exact_match_accuracy": round(totals["exact_match"] / d, 4),
        "no_tool_precision": None if prec is None else round(prec, 4),
        "no_tool_recall": None if rec is None else round(rec, 4),
        "no_tool_f1": None if f1 is None else round(f1, 4),
        "sample_failures": failures,
    }


def evaluate(rows: list[dict], backend: str, schemas: ToolSchemas | None = None,
             model_id: str | None = None) -> dict:
    schemas = schemas or load_tools()
    if backend == "heuristic":
        infer = lambda row: heuristic_route(  # noqa: E731
            json.loads(row["messages"][1]["content"])["request"], schemas)
    elif backend == "gold":
        infer = lambda row: json.dumps(row["gold"])  # noqa: E731
    elif backend == "transformers":
        if not model_id:
            raise SystemExit("--model is required for the transformers backend")
        infer = make_transformer_backend(model_id)
    else:
        raise SystemExit(f"unknown backend {backend!r}")

    report = {"backend": backend, "model_id": model_id, "overall": _aggregate(rows, infer, schemas)}
    by_family: dict[str, dict] = {}
    for fam in sorted({r.get("family", "unknown") for r in rows}):
        sub = [r for r in rows if r.get("family", "unknown") == fam]
        agg = _aggregate(sub, infer, schemas)
        by_family[fam] = {k: agg[k] for k in agg if k != "sample_failures"}
    # Unseen-schema slice is promoted to the top level for release gating.
    unseen = by_family.get("unseen_schema", {})
    report["unseen_schema_exact_match"] = unseen.get("exact_match_accuracy")
    report["by_family"] = by_family
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl",
                    help="path, comma list, or glob of held-out JSONL files")
    ap.add_argument("--backend", default="heuristic",
                    choices=["heuristic", "gold", "transformers"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths: list[str] = []
    for part in args.data.split(","):
        part = part.strip()
        paths.extend(sorted(glob.glob(part)) or ([part] if Path(part).exists() else []))
    rows: list[dict] = []
    for p in paths:
        rows.extend(json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())
    if not rows:
        raise SystemExit(f"no rows loaded from {args.data!r}")

    report = evaluate(rows, args.backend, load_tools(), args.model)
    report["data_files"] = paths
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "overall"}
                     | {"overall": {k: v for k, v in report["overall"].items()
                                    if k != "sample_failures"}}, indent=2))
    if args.out:
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())