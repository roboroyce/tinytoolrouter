#!/usr/bin/env python3
"""Read-only Hugging Face metadata collector for small-model landscape research.

Queries the public HF Hub API (no auth, no downloads of weights) and records
model name, license, params, downloads, likes, lastModified, and tags.

Usage:
    python3 src/research_hf.py            # prints a JSON snapshot to stdout
    python3 src/research_hf.py --out f.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models"
USER_AGENT = "hf-model-lab-research/0.1 (+read-only metadata)"

# Candidate small models in the 0.5B-3B band plus a few reference points.
CANDIDATES = [
    # Qwen family
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    # Llama family
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    # SmolLM / HuggingFaceTB
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    # Google
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    # Microsoft
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    # Others
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-2-1_6b-chat",
    "tiiuae/Falcon3-1B-Instruct",
    "facebook/opt-1.3b",
    # Tool-calling / structured-output specific candidates to check competition
    "Team-ACE/ToolACE-2-Llama-3.1-8B",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "ibm-granite/granite-3.3-2b-instruct",
    "ibm-granite/granite-3.1-2b-instruct",
    "Salesforce/xlam-function-calling-60k",
]


def fetch(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_info(model_id: str) -> dict:
    """Fetch expanded metadata for one model; tolerate 404/403 (gated)."""
    url = (f"{API}/{model_id}?blobs=false"
           "&expand[]=safetensors&expand[]=lastModified&expand[]=downloads"
           "&expand[]=likes&expand[]=tags&expand[]=cardData")
    try:
        raw = fetch(url)
    except urllib.error.HTTPError as e:
        return {"id": model_id, "error": f"HTTP {e.code}", "available": False}
    except Exception as e:  # noqa: BLE001
        return {"id": model_id, "error": repr(e), "available": False}

    st = raw.get("safetensors") or {}
    params = st.get("total")
    tags = raw.get("tags", []) or []
    lic = (raw.get("cardData") or {}).get("license")
    if not lic:
        lic = next((t.split(":", 1)[1] for t in tags
                    if t.startswith("license:")), None)
    rec = {
        "id": raw.get("id", model_id),
        "available": True,
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "lastModified": raw.get("lastModified"),
        "createdAt": raw.get("createdAt"),
        "pipeline_tag": raw.get("pipeline_tag"),
        "license": lic,
        "params_total": params,
        "params_billions": round(params / 1e9, 3) if isinstance(params, (int, float)) else None,
        "dtype": st.get("parameters", {}) if isinstance(st.get("parameters"), dict) else None,
        "tags": tags,
        "gated": raw.get("gated"),
    }
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write JSON to this path instead of stdout")
    args = ap.parse_args(argv)

    snapshot = {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": API,
        "models": [],
    }
    for mid in CANDIDATES:
        rec = model_info(mid)
        snapshot["models"].append(rec)
        status = "OK" if rec.get("available") else f"SKIP({rec.get('error')})"
        print(f"[{status}] {mid}", file=sys.stderr)

    text = json.dumps(snapshot, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())