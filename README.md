# hf-model-lab — TinyToolRouter-0.6B

A 0.6B-parameter, CPU-first **tool router** built on
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0),
trained and evaluated on a temporary AWS GPU worker.

**Status:** trained, evaluated, gate passed — **not yet published to the Hugging
Face Hub** (upload blocked on HF authentication not being available on the
build host). Model artifacts are preserved under `artifacts/`.

**Measured results (231 held-out cases):**

| Metric | base Qwen3-0.6B | TinyToolRouter-0.6B |
|---|---|---|
| Exact match | 0.5195 | **0.9221** |
| Valid structured output | 0.9913 | **0.9913** |
| Tool selection | 0.9351 | **0.9870** |
| no_tool F1 | 0.8571 | **0.9877** |

Full model card with usage, training config, limitations and reproduction
commands: **[`RELEASE_CARD.md`](RELEASE_CARD.md)**.

## Layout

| Path | What |
|---|---|
| `RESEARCH.md` | competitive landscape + concept scoring |
| `MODEL_SPEC.md` | chosen design, base model, success criteria |
| `DATASET_CARD.md` | dataset provenance, splits, leakage controls |
| `EVALUATION_PLAN.md` | metrics, protocol, release gate |
| `RELEASE_CARD.md` | the Hugging Face model card (real numbers) |
| `PLAN.md` | phases, compute, risks, cost |
| `src/` | dataset generator, harness, trainer, publisher, HF research |
| `tests/` | 43 stdlib-only unit tests |
| `reports/` | base vs fine-tuned metric JSON + worker logs |
| `artifacts/` | merged model + GGUF builds (Q4_K_M / Q8_0 / f16) |
| `scripts/remote/` | AWS worker bootstrap + GGUF conversion |

## Quick start

```bash
python3 -m unittest discover -s tests
python3 src/gen_dataset.py --out data --seed 11 --n-train 600 --n-seen 140 \
    --n-unseen 60 --n-notool 40 --n-train-notool 90
python3 src/eval_harness.py --data data/test.jsonl --backend heuristic
python3 src/eval_harness.py --data data/test.jsonl --backend gold
```

No ML dependencies are needed for the harness's `heuristic`/`gold` backends;
the `transformers` backend additionally needs `transformers`+`torch` and local
weights.