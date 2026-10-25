#!/usr/bin/env python3
"""Publish TinyToolRouter-0.6B to the Hugging Face Hub — gated on real results.

Refuses to publish unless a finetune report exists AND the recorded metrics
clear the release target. This is deliberate: the mission forbids advertising
an improvement that was not measured.

Usage:
    python3 src/publish_hf.py --report reports/finetune.json \
        --model-dir artifacts/tinytoolrouter-0.6b/merged \
        --repo TinyToolRouter-0.6B --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RELEASE_TARGET = {
    "min_valid_structured_output_rate": 0.95,
    "min_exact_match_gain_vs_base": 0.10,  # absolute (percentage points /100)
    "min_heldout_cases": 200,
}


def check_gate(finetune: dict, base: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    f = finetune["overall"]
    b = base["overall"]
    if f["valid_structured_output_rate"] < RELEASE_TARGET["min_valid_structured_output_rate"]:
        reasons.append(
            f"valid_structured_output_rate {f['valid_structured_output_rate']} "
            f"< {RELEASE_TARGET['min_valid_structured_output_rate']}")
    gain = f["exact_match_accuracy"] - b["exact_match_accuracy"]
    if gain < RELEASE_TARGET["min_exact_match_gain_vs_base"]:
        reasons.append(f"exact-match gain {gain:+.4f} < "
                       f"{RELEASE_TARGET['min_exact_match_gain_vs_base']}")
    if finetune["overall"]["n"] < RELEASE_TARGET["min_heldout_cases"]:
        reasons.append(f"only {finetune['overall']['n']} held-out cases")
    return (not reasons), reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="finetune report JSON")
    ap.add_argument("--base-report", default="reports/base.json")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", default="TinyToolRouter-0.6B")
    ap.add_argument("--card", default="RELEASE_CARD.md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    finetune = json.loads(Path(args.report).read_text())
    base = json.loads(Path(args.base_report).read_text())
    ok, reasons = check_gate(finetune, base)
    if not ok:
        print("RELEASE GATE FAILED — not publishing:")
        for r in reasons:
            print("  -", r)
        return 1
    print("Release gate PASSED.")

    if args.dry_run:
        print("--dry-run set; would upload", args.model_dir, "->", args.repo)
        return 0

    try:
        from huggingface_hub import HfApi, login
    except Exception as e:  # noqa: BLE001
        print(f"huggingface_hub unavailable: {e!r}", file=sys.stderr)
        return 2

    api = HfApi()
    who = api.whoami()  # raises if not authenticated
    user = who.get("name")
    repo_id = f"{user}/{args.repo}"
    print("Authenticated as", user, "->", repo_id)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=args.model_dir, repo_id=repo_id)
    if Path(args.card).exists():
        api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                        repo_id=repo_id)
    print("Published:", f"https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())