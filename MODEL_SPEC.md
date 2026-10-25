# MODEL_SPEC.md — TinyToolRouter-0.6B

**Status:** design only. No training has been run and no weights have been
downloaded or uploaded. This document specifies the intended release.

## 1. One-line pitch

A 0.6B-parameter, CPU-first **tool router** that turns a natural-language
request plus a caller-supplied tool schema into exactly one valid
`{"name", "arguments"}` call — or an explicit **abstain** decision — and whose
improvement over its base model is published as a reproducible benchmark.

## 2. Audience

- Builders running local agents (Ollama / llama.cpp / LM Studio) on laptops,
  Raspberry Pis, or CPU-only servers who need *reliable* routing without a GPU.
- Teams that need a permissive license for commercial embedding.
- Anyone who wants an honest base-vs-fine-tune number instead of a model card
  full of adjectives.

## 3. Base model

**Chosen: `Qwen/Qwen3-0.6B`** (observed 0.752B params; *safetensors* total
753,774,848; license tag `apache-2.0`; last modified 2025-07-26; 24,928,161
downloads / 1850 likes in the 2026-10-25 snapshot).

Why this base:

| Requirement | Evidence |
|---|---|
| Permissive license for redistribution | `apache-2.0` tag (verified in snapshot) |
| Tiny enough for CPU + cheap fine-tune | 0.752B params |
| Ships a **valid tool-calling chat template** | Qwen3 series provides `tools`/`tool_call` roles; avoids inventing a format |
| Large installed base for discovery | 24.9M downloads, highest of any ≤1B text-gen model observed |
| Fine-tuning permitted | Apache-2.0 grants derivative works |

**Verify before training (blocking checklist):**
- [ ] Read the current `Qwen/Qwen3-0.6B` card/`LICENSE` and confirm Apache-2.0
      still holds and no added restrictions appeared after 2026-10-25.
- [ ] Confirm the tokenizer's `apply_chat_template(..., tools=...)` renders a
      valid tool schema block and that `tools` is accepted in the template.
- [ ] Confirm TRL's supported chat template for this base (tool-call messages
      must round-trip, not be flattened to plain text).

Fallback base if Qwen3-0.6B licensing or template changes:
`HuggingFaceTB/SmolLM3-3B` (apache-2.0, but 4× larger) or
`HuggingFaceTB/SmolLM2-360M-Instruct` (apache-2.0, smallest viable).

**Rejected bases:** Llama-3.2 (Community License naming/derivative terms),
Gemma (Gemma Terms), Falcon3/StableLM/OPT (`other`/unclear), Qwen2.5-3B
(`other`, Qwen research license).

## 4. Capabilities (target behaviour)

Given `system` = router instructions and `user` = JSON containing `tools`
(list of valid JSON schemas) and `request` (natural language):

1. **Select** exactly one tool when the request maps to one (multi-tool lists,
   distractor tools present).
2. **Fill** required arguments with values taken only from the request; never
   invent a value that is absent (missing-value case → abstain or clarify).
3. **Abstain** with `{"name": "no_tool", "arguments": {}}` when no listed tool
   applies, rather than forcing a bad call.
4. **Stay schema-valid**: output must parse as JSON and satisfy the selected
   tool's `parameters` (types, required keys, no unexpected keys).

Non-goals: general chat, code generation, multi-step planning, executing tools.

## 5. Output contract

```json
{"name": "<tool_name or no_tool>", "arguments": { ... }}
```

- Rendered inside Qwen3's chat template; the model card will document the exact
  prompt template and a constrained-decoding option (llama.cpp GBNF) that makes
  the JSON grammar *structurally* impossible to violate.
- `no_tool` is a reserved sentinel name, also declared in the system prompt so
  constrained decoding can allow it.

## 6. Intended artifacts on release

| Artifact | Format | Note |
|---|---|---|
| `TinyToolRouter-0.6B` | transformers safetensors (merged LoRA) | primary |
| `TinyToolRouter-0.6B-GGUF` | Q4_K_M, Q8_0 | for CPU/Ollama users |
| Model card | README | includes harness + numbers |

## 7. Success criteria (see `EVALUATION_PLAN.md` for the protocol)

The release is justified **only if**, on the held-out test split:

- exact-match accuracy ≥ **0.80**, and
- valid-structured-output rate ≥ **0.98**, and
- abstention F1 ≥ **0.90**, and
- every headline metric beats the unmodified base by **≥ +15 points**.

If those fail, do **not** publish as an improvement; publish the harness and
negative result instead (that is still genuinely useful and honest).

## 8. Compute and cost (estimate — to be confirmed before any spend)

- LoRA (r=16, α=32) on 0.6B, ~2–5k examples, 2–3 epochs, max_len 512:
  fits a single 24GB GPU; ~30–90 min/epoch.
- AWS reference instance: **g5.xlarge** (1× A10G 24GB). On-demand list price is
  on the order of **≈ $1.00–1.10/hour** in us-east-1 *(verify current
  pricing — this was not fetched from AWS)*. A 3-hour session ⇒ **≈ $3–4**;
  budget **≤ $10** to be safe.
- Base weights download ≈ 1.5 GB (fp16) / ≈ 0.4 GB GGUF. Egress negligible.
- **Total estimated spend: < $10 against the $107.83 credit balance.**

All spend is **blocked** pending explicit approval (mission rules).