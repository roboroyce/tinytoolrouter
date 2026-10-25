"""Shared schema + tool-call utilities for the hf-model-lab project.

Pure standard library (json + re). No network, no ML deps.

Core ideas
----------
* A *tool schema* is a JSON-Schema-ish dict describing one callable tool:
      {"name": "get_weather",
       "description": "...",
       "parameters": {"type": "object",
                      "properties": {"city": {"type": "string"}, ...},
                      "required": ["city"]}}
* A *tool call* is {"name": <str>, "arguments": {<str>: <value>}}.
* We measure two things separately:
    - tool-selection accuracy  (did it pick the right `name`?)
    - argument accuracy        (are the argument values correct?)
  plus a *valid structured-output rate* (does the raw text parse to a
  well-formed call that satisfies the schema?).
"""
from __future__ import annotations

import json
import re
from typing import Any

JSON_TYPES = {"string": str, "number": (int, float), "integer": int,
              "boolean": bool, "array": list, "object": dict}

# Reserved sentinel the model emits when no supplied tool applies.
NOTOOL = "no_tool"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


class ToolSchemas:
    """A small, self-contained tool registry used by generator + harness."""

    def __init__(self, tools: list[dict]):
        self.tools = tools
        self.by_name = {t["name"]: t for t in tools}

    def names(self) -> list[str]:
        return [t["name"] for t in self.tools]


def validate_schema(tool: dict) -> list[str]:
    """Return a list of structural problems (empty list == valid)."""
    errs: list[str] = []
    if not isinstance(tool, dict):
        return ["tool must be an object"]
    if not isinstance(tool.get("name"), str) or not tool["name"]:
        errs.append("missing/empty 'name'")
    params = tool.get("parameters")
    if not isinstance(params, dict):
        errs.append("missing 'parameters' object")
        return errs
    if params.get("type") != "object":
        errs.append("parameters.type must be 'object'")
    props = params.get("properties")
    if not isinstance(props, dict) or not props:
        errs.append("parameters.properties must be a non-empty object")
    else:
        for pname, pspec in props.items():
            if not isinstance(pspec, dict) or pspec.get("type") not in JSON_TYPES:
                errs.append(f"property {pname!r} has unsupported type")
    required = params.get("required", [])
    if not isinstance(required, list):
        errs.append("parameters.required must be a list")
    elif isinstance(props, dict):
        for r in required:
            if r not in props:
                errs.append(f"required property {r!r} not in properties")
    return errs


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of a JSON object from model text.

    Handles raw JSON, fenced ```json blocks, and a JSON object embedded in
    prose. Returns None when nothing parses.
    """
    if not isinstance(text, str):
        return None
    candidates: list[str] = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    m = _BRACE_RE.search(text)
    if m:
        candidates.append(m.group(0).strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_call(obj: Any) -> dict | None:
    """Coerce a parsed object into {"name":..., "arguments":{...}} or None.

    Accepts the common shapes:
      {"name": "x", "arguments": {...}}
      {"name": "x", "parameters": {...}}
      {"tool": "x", "args": {...}}
      OpenAI-style {"function": {"name": ..., "arguments": <json str>}}
    """
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def validate_call(call: dict | None, schemas: ToolSchemas) -> list[str]:
    """Return structural problems of a normalized call against the registry."""
    if call is None:
        return ["could not parse a tool call"]
    errs: list[str] = []
    if call["name"] == NOTOOL:
        # Abstention: must carry no arguments, but need not match a schema.
        return [] if not call["arguments"] else ["no_tool must not carry arguments"]
    tool = schemas.by_name.get(call["name"])
    if tool is None:
        return [f"unknown tool {call['name']!r}"]
    params = tool["parameters"]
    props = params.get("properties", {})
    required = params.get("required", [])
    args = call["arguments"]
    for r in required:
        if r not in args:
            errs.append(f"missing required argument {r!r}")
    for k, v in args.items():
        if k not in props:
            errs.append(f"unexpected argument {k!r}")
            continue
        expected = props[k].get("type")
        py = JSON_TYPES.get(expected)
        if py is None:
            continue
        # bool is a subclass of int; guard integer/number checks.
        if expected in ("integer", "number") and isinstance(v, bool):
            errs.append(f"argument {k!r} must be {expected}, got boolean")
        elif expected == "integer" and isinstance(v, float) and not v.is_integer():
            errs.append(f"argument {k!r} must be an integer")
        elif not isinstance(v, py):
            errs.append(f"argument {k!r} must be {expected}")
        elif expected == "array":
            items = props[k].get("items", {}).get("type")
            ipy = JSON_TYPES.get(items)
            if ipy is not None:
                for i, elem in enumerate(v):
                    if isinstance(elem, bool) and items in ("integer", "number"):
                        errs.append(f"argument {k}[{i}] must be {items}, got boolean")
                    elif not isinstance(elem, ipy):
                        errs.append(f"argument {k}[{i}] must be {items}")
    return errs


def _canon(v: Any) -> Any:
    """Canonical comparison form: numbers -> float, strings stripped/lowercased."""
    if isinstance(v, str):
        return v.strip().casefold()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in sorted(v.items())}
    return v


def score_prediction(pred_text: str, gold: dict, schemas: ToolSchemas) -> dict:
    """Score one prediction against one gold call.

    Three things are measured *separately* so a well-formed JSON prediction
    is never silently counted as "unparseable":

    * ``json_parsed``  – the raw text yielded a JSON object/array at all.
    * ``normalized``   – that JSON coerced to {name, arguments}.
    * ``parse_ok``     – the call also satisfies the tool schema (types,
                         required keys, no unexpected keys).  This is the
                         *valid structured output* signal.

    Plus ``tool_correct``, ``args_correct``, ``exact_match`` and the
    abstention flags ``pred_is_notool`` / ``gold_is_notool``.
    """
    raw = extract_json(pred_text)
    json_parsed = raw is not None
    call = normalize_call(raw)
    normalized = call is not None
    problems = validate_call(call, schemas)
    # A no_tool abstention is valid *by definition* (it names a reserved
    # sentinel, not a schema tool) as long as it normalizes.
    is_notool = bool(call and call["name"] == NOTOOL)
    parse_ok = call is not None and not problems
    tool_correct = bool(call and call["name"] == gold["name"])
    args_correct = False
    if call and tool_correct:
        gp, pp = _canon(gold["arguments"]), _canon(call["arguments"])
        args_correct = gp == pp
    return {
        "json_parsed": json_parsed,
        "normalized": normalized,
        "parse_ok": parse_ok,
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "exact_match": bool(tool_correct and args_correct),
        "pred_is_notool": is_notool,
        "gold_is_notool": gold.get("name") == NOTOOL,
    }