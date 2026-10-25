---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
base_model_relation: finetune
library_name: transformers
language:
  - en
pipeline_tag: text-generation
tags:
  - tool-calling
  - function-calling
  - tool-use
  - agent
  - router
  - structured-output
  - json
  - qwen3
  - lora
  - cpu
  - gguf
  - text-generation-inference
---

# TinyToolRouter-0.6B

A 0.6B-parameter, **CPU-first tool router**. Given a natural-language request
plus a list of available tool schemas, it returns exactly **one** tool call:

```json
{"name": "get_weather", "arguments": {"city": "Oslo"}}
```

or, when no listed tool applies, it **abstains**:

```json
{"name": "no_tool", "arguments": {}}
```

- **Base model:** [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0)
- **Method:** LoRA supervised fine-tuning (PEFT), merged into the base weights
- **Size:** 0.6B parameters (fp16/bf16 safetensors ≈ 1.2 GB; Q4_K_M GGUF ≈ 397 MB)
- **Language:** English
- **License:** Apache-2.0

Small models are usually *shaky* at tool routing: they hallucinate tools, emit
prose instead of JSON, or force a tool when none fits. This model is tuned
narrowly for reliable routing, valid structured output, and honest abstention —
and its improvement over the base model is published as a reproducible
benchmark rather than an assertion.

## Results

Evaluated on a **held-out set of 231 cases** across three families:

| Family | n | What it tests |
|---|---|---|
| `seen_paraphrase` | 131 | unseen *phrasing* of tools seen in training |
| `unseen_schema` | 60 | tools **never seen during training** |
| `no_tool` | 40 | requests matching no tool → must abstain |

Both models were run with identical prompts, `do_sample=False`,
`max_new_tokens=128`.

| Metric | Qwen3-0.6B (base) | **TinyToolRouter-0.6B** |
|---|---|---|
| **Exact match** (headline) | 0.5195 | **0.9221** |
| Valid structured output | 0.9913 | **0.9913** |
| Tool selection | 0.9351 | **0.9870** |
| Argument accuracy | 0.5195 | **0.9221** |
| `no_tool` precision | 0.7647 | **0.9756** |
| `no_tool` recall | 0.9750 | **1.0000** |
| `no_tool` F1 | 0.8571 | **0.9877** |

By family (exact match):

| Family | base | **fine-tuned** |
|---|---|---|
| seen_paraphrase | 0.229 | **0.9237** |
| unseen_schema | 0.850 | **0.8667** |
| no_tool | 0.975 | **1.0000** |

The largest gain is on paraphrased requests (+69.5 pts); generalisation to
genuinely unseen tool schemas improves only modestly (+1.7 pts) and is not
" solved".

## Quickstart

```bash
pip install transformers torch
```

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Robokid1134/TinyToolRouter-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

SYSTEM = (
    "You are a tool router. You are given a list of available tools and a user "
    "request. Reply with exactly one JSON object of the form "
    '{"name": <tool_name>, "arguments": {<argument>: <value>}}. '
    "Use only the listed tools and only their declared arguments. "
    'If no listed tool applies, reply {"name": "no_tool", "arguments": {}}. '
    "Output JSON only, no prose."
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": json.dumps(
        {"tools": tools, "request": "What's the weather in Oslo?"})},
]

prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# {"name": "get_weather", "arguments": {"city": "Oslo"}}
```

`enable_thinking=False` is recommended: the router should answer directly
rather than emit a reasoning block.

### GGUF (CPU / llama.cpp / Ollama)

The repository ships GGUF builds under `gguf/`:

| File | Size | Use |
|---|---|---|
| `gguf/model-Q4_K_M.gguf` | ≈ 397 MB | best size/quality for CPU |
| `gguf/model-Q8_0.gguf` | ≈ 639 MB | near-lossless |
| `gguf/model-f16.gguf` | ≈ 1.2 GB | full precision |

```bash
# llama.cpp
llama-cli -m gguf/model-Q4_K_M.gguf -p "<your prompt>"
```

> Note: the GGUF builds were converted from the merged model but have **not**
> been separately re-benchmarked; the metrics above come from the safetensors
> model.

## About the tool schemas

The model reads JSON-Schema-style tool definitions supplied in the user turn:

```json
{"tools": [{"name": "...", "description": "...",
            "parameters": {"type": "object",
                           "properties": {"...": {"type": "..."}},
                           "required": ["..."]}}],
 "request": "natural-language request"}
```

It was trained on `string`, `integer`, `number`, `boolean`, `array<string>`
and `array<number>` arguments, with required and optional parameters, and with
distractor tools present in the list.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` |
| Method | LoRA (PEFT), merged |
| LoRA | r=16, α=32, dropout=0.05, all attention+MLP projections |
| Epochs | 3 (57 steps) |
| LR | 1e-4, cosine, warmup 3% |
| Batch | 8 × grad-accum 4, max_len 512 |
| Precision | bf16 |
| Seed | 7 |
| Examples | 599 (train) / 231 (held-out) |
| Hardware | NVIDIA A10G 24 GB |
| Train time | ≈ 150 s |
| Libs | torch 2.5.1, transformers 4.57.6, peft 0.17.1, Python 3.9 |

Loss is computed on the assistant JSON only (prompt tokens masked).

## Dataset

**Original synthetic data** authored for this project (Apache-2.0). No scraped
text, no chat logs, no personal data. Slot values are invented
(`sam@example.com`, `2026-11-02T09:00`, city names, …).

- **Seen tools (9):** `get_weather`, `send_email`, `create_calendar_event`,
  `search_web`, `convert_currency`, `set_timer`, `translate_text`,
  `filter_items`, `tag_note`
- **Unseen tools (5, evaluation only):** `play_music`, `get_stock_price`,
  `book_restaurant`, `draw_chart`, `create_reminder`

**Split hygiene:** training and held-out sets use disjoint templates, disjoint
value pools and disjoint example ids; the `unseen_schema` family uses tools
absent from training entirely. A `no_tool` abstention set is present in
training with requests disjoint from the held-out `no_tool` set.

## Limitations

- **English only**; 14 tools; synthetic data. It teaches routing discipline and
  schema reading, not world knowledge.
- **Not an assistant.** It does not execute tools, hold conversations, or answer
  general questions — those requests should map to `no_tool`.
- **Unseen-schema gains are modest** (+1.7 pts). Robustness to radically novel
  schemas is improved but not solved.
- **Arguments usually appear verbatim in the request.** The hard capability
  measured here is selection + correct typing, not open-world extraction.
- **Single seed.** One reproducible run; multi-seed variance is unreported.
- **Small evaluation set** (231 cases). Treat differences of a few cases as
  noise.
- **GGUF builds unverified numerically.**
- Trained with `max_len` 512; very long tool lists may be truncated by the user.

## Intended use & out-of-scope

**Use:** local agents on CPU-only or low-VRAM machines; as a cheap routing
front-end before a larger model; as a reproducible baseline for tiny-model
tool-calling research.

**Do not use for:** safety-critical decisions, executing shell/system commands
without human review, or any task where a wrong tool call causes harm. The
model emits *proposed* calls; the caller must validate them.

## Citation

```bibtex
@misc{tinytoolrouter06b,
  title  = {TinyToolRouter-0.6B: a CPU-first, schema-strict tool router},
  author = {Robokid1134},
  year   = {2026},
  note   = {LoRA fine-tune of Qwen/Qwen3-0.6B},
  url    = {https://huggingface.co/Robokid1134/TinyToolRouter-0.6B}
}
```

## Attribution & license

Fine-tuned from [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B),
licensed **Apache-2.0**. This derivative is released under the same
**Apache-2.0** license; the upstream copyright notice and license are
preserved. Code and generated dataset for this project are Apache-2.0.