# EVALUATION_PLAN.md — how we decide whether the fine-tune is better

**Principle:** no headline claim without a reproducible number, and no number
without the command that produced it.

## 1. Baselines & comparison set

| ID | What | Purpose |
|---|---|---|
| B0 | **Unmodified `Qwen/Qwen3-0.6B`** (`--backend transformers --model Qwen/Qwen3-0.6B`) | the honest comparison the mission requires |
| B1 | `heuristic` backend | weak floor; proves the harness; *not* a model |
| B2 | `gold` backend | sanity check: every metric must be 1.0 |
| FT | TinyToolRouter-0.6B (LoRA merged) | candidate |

All four run on the **same** `data/test.jsonl` with the same system/user
messages, `do_sample=False`, `max_new_tokens=128`.

## 2. Held-out data

`data/test.jsonl` — 231 cases in three families (see `DATASET_CARD.md`):

| family | n | what it isolates |
|---|---|---|
| `seen_paraphrase` | 131 | generalisation to unseen *phrasing* of seen tools |
| `unseen_schema` | 60 | generalisation to tools **never trained on** |
| `no_tool` | 40 | abstention on requests that match no tool |

Train templates **and** value pools are disjoint from all three families.

## 3. Metrics (computed by `src/eval_harness.py`)

| Metric | Definition |
|---|---|
| `json_parse_rate` | raw output yields JSON at all (shape only) |
| `valid_structured_output_rate` | JSON that also satisfies the selected tool's schema |
| `tool_selection_accuracy` | predicted `name` == gold `name` |
| `argument_accuracy` | right tool and all argument values canonical-equal |
| `exact_match_accuracy` | headline: right tool + right arguments |
| `no_tool_precision/recall/f1` | abstention quality |
| `unseen_schema_exact_match` | exact match on the held-out-schema family |

**Bug fixed and regression-tested:** JSON predictions that are *schema-invalid*
(like the heuristic's `{"name": X, "arguments": {}}`) were previously counted
as "unparseable". They are now reported as **parsed but schema-invalid**:
`json_parse_rate` and `valid_structured_output_rate` are separate numbers.
`tests/test_eval.py::TestJsonScoring` proves a well-formed JSON prediction is
parsed and scored correctly.

## 4. Protocol

```bash
python3 src/gen_dataset.py --out data --seed 7 --n-train 600 --n-seen 140 --n-unseen 60 --n-notool 40
python3 src/eval_harness.py --data data/test.jsonl --backend gold
python3 src/eval_harness.py --data data/test.jsonl --backend heuristic --out reports/baseline-heuristic.json
# on the GPU worker only:
python3 src/eval_harness.py --data data/test.jsonl --backend transformers --model Qwen/Qwen3-0.6B --out reports/base.json
python3 src/train_lora.py --data data --out artifacts/tinytoolrouter-0.6b --base Qwen/Qwen3-0.6B --seed 7 --epochs 3 --merge
python3 src/eval_harness.py --data data/test.jsonl --backend transformers --model artifacts/tinytoolrouter-0.6b/merged --out reports/finetune.json
```

## 5. Release target (pre-registered; enforced in `src/publish_hf.py`)

Publish **iff** all hold:

- `valid_structured_output_rate` ≥ **0.95**
- `exact_match_accuracy` gain vs base ≥ **+10 percentage points**
- ≥ **200** held-out cases (we have 231)
- `no_tool` and `unseen_schema` performance reported and non-trivial

Otherwise: publish the harness + honest negative result; never rename the base
or advertise an unmeasured improvement.

## 6. Threats to validity (and mitigations)

| Threat | Mitigation |
|---|---|
| Template/value leakage inflating scores | disjoint templates, pools, tools and ids; asserted by tests; `leakage_check: pass` |
| Format-only gain (base emits prose) | report `json_parse_rate` separately from schema-validity |
| Arguments copied verbatim from the request | still requires extraction + correct typing; accepted and stated |
| Small `no_tool`/`unseen` sets | reported per-family; treated as directional, not precise |
| Suspiciously perfect results | investigate leakage and metric bugs before claiming (mission rule); the `gold` backend is *expected* to be 1.0 and is never used as model evidence |
| Single seed | mission says start with one reproducible run; multi-seed only after the first works |

## 7. What justifies a public release

A `reports/finetune.json` produced by the frozen harness on the frozen split,
clearing §5, with baseline `reports/base.json` and the exact commands. Nothing
less; no marketing claim may exceed that file.