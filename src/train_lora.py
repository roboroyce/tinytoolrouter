#!/usr/bin/env python3
"""LoRA supervised fine-tune of a tiny tool router (remote GPU only).

Runs on the AWS worker. Never run this on the local laptop: it downloads the
base weights.

Design notes
------------
* Base: ``Qwen/Qwen3-0.6B`` (Apache-2.0) by default.
* We format each example with the tokenizer's own chat template and **mask the
  prompt tokens** (labels = -100) so loss is computed on the assistant JSON
  only. No TRL dependency, which keeps the run reproducible across versions.
* LoRA (PEFT) so a 0.6B model trains comfortably on a 24GB A10G/L4.
* Every run writes ``run_meta.json`` with hyperparameters, library versions,
  dataset hashes and the git-less code hash so results are auditable.

Usage:
    python3 src/train_lora.py --data data --out artifacts/tinytoolrouter-0.6b \
        --base Qwen/Qwen3-0.6B --seed 7 --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_examples(rows: list[dict], tok, max_len: int):
    """Return (input_ids, attention_mask, labels) tensors with prompt masked."""
    import torch
    from model_utils import render_chat

    feats = []
    for row in rows:
        msgs = row["messages"]
        full = render_chat(tok, msgs, add_generation_prompt=False)
        prompt = render_chat(tok, msgs[:-1], add_generation_prompt=True)
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        plen = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(plen):
            labels[i] = -100
        feats.append({
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })
    return feats


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            out["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        return {k: torch.stack(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/tinytoolrouter-0.6b")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a smoke test")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (slower; not needed for 0.6B on 24GB)")
    ap.add_argument("--merge", action="store_true", help="also save merged fp16 weights")
    args = ap.parse_args(argv)

    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    data = Path(args.data)
    train_path = data / "train.jsonl"
    rows = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    if args.grad_ckpt:
        # Required for PEFT + gradient checkpointing, else the loss loses its graph.
        model.enable_input_require_grads()

    feats = build_examples(rows, tok, args.max_len)
    split = max(1, int(len(feats) * 0.05))
    eval_feats, train_feats = feats[:split], feats[split:]

    targs = TrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        report_to=[],
        max_steps=args.max_steps,
        gradient_checkpointing=args.grad_ckpt,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_feats,
                      eval_dataset=eval_feats, data_collator=Collator(tok.pad_token_id))
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"), safe_serialization=True)
        tok.save_pretrained(str(out / "merged"))

    meta = {
        "base_model": args.base,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "bsz": args.bsz,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "n_train_examples": len(train_feats),
        "n_eval_examples": len(eval_feats),
        "train_sha256": _sha256(train_path),
        "manifest_sha256": _sha256(data / "MANIFEST.json"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_metrics": {k: v for k, v in trainer.state.log_history[-1].items()},
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())