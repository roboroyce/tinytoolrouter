# RESEARCH.md — Competitive landscape and concept selection

**Project:** `hf-model-lab`
**Data collected:** 2026-10-25, 01:44–01:46 UTC (single snapshot)
**Method:** Read-only queries to the public Hugging Face Hub API
(`https://huggingface.co/api/models`). No weights downloaded, no auth used,
no remote repos modified.
**Raw snapshots (reproducible):**
- `research/snapshot-2026-10-24.json` — 27 detailed model records
- `research/search-2026-10-24.json` — discovery results for 9 niche queries
- collected by `src/research_hf.py` and `src/search_hf.py`

> All numbers below are **observed** values from that snapshot. Anything
> labelled *speculation* is inference, not data. Download counts on the Hub
> are all-time and include mirrored/derived repos, so they indicate *attention*,
> not active usage.

---

## 1. Small-model landscape (0.5B–3B), observed

Authoritative rows from `snapshot-2026-10-24.json`:

| Model | Params (B) | Downloads | Likes | License | Last modified |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.752 | 24,928,161 | 1850 | apache-2.0 | 2025-07-26 |
| Qwen/Qwen3-1.7B | 2.032 | 4,115,815 | 593 | apache-2.0 | 2025-07-26 |
| Qwen/Qwen2.5-0.5B-Instruct | 0.494 | 5,308,304 | 711 | apache-2.0 | 2024-09-25 |
| Qwen/Qwen2.5-1.5B-Instruct | 1.544 | 4,705,359 | 910 | apache-2.0 | 2024-09-25 |
| Qwen/Qwen2.5-3B-Instruct | 3.086 | 3,718,369 | 620 | other (Qwen research) | 2024-09-25 |
| meta-llama/Llama-3.2-1B-Instruct | 1.236 | 3,708,648 | 1887 | llama3.2 | 2024-10-24 |
| meta-llama/Llama-3.2-3B-Instruct | 3.213 | 1,120,299 | 2870 | llama3.2 | 2024-10-24 |
| HuggingFaceTB/SmolLM2-135M-Instruct | 0.135 | 1,066,330 | 462 | apache-2.0 | 2025-09-22 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.711 | 256,931 | 766 | apache-2.0 | 2025-04-21 |
| HuggingFaceTB/SmolLM3-3B | 3.075 | 723,577 | 1070 | apache-2.0 | 2025-09-10 |
| google/gemma-3-1b-it | 1.000 | 904,692 | 1268 | gemma | 2025-04-04 |
| google/gemma-2-2b-it | 2.614 | 250,867 | 1587 | gemma | 2024-08-27 |
| microsoft/Phi-3.5-mini-instruct | 3.821 | 375,768 | 1283 | mit | 2025-12-10 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.100 | 964,435 | 1884 | apache-2.0 | 2024-03-17 |
| ibm-granite/granite-3.3-2b-instruct | 2.534 | 339,424 | 87 | apache-2.0 | 2025-05-08 |
| tiiuae/Falcon3-1B-Instruct | 1.669 | 7,825 | 46 | other | 2025-01-10 |
| stabilityai/stablelm-2-1_6b-chat | 1.645 | 1,313 | 34 | other | 2024-06-03 |

### Observations

1. **Apache-2.0 is the only base license that removes legal friction** for a
   freely redistributable fine-tune. Qwen (2.5/3), SmolLM2/3, TinyLlama, and
   Granite are Apache-2.0. Llama-3.2 is under the *Llama 3.2 Community
   License* (needs derivative naming + acceptable-use terms); Gemma is under
   the *Gemma Terms*; Phi-3.5 is MIT; Falcon3/StableLM/OPT are `other`.
2. **Qwen3-0.6B is a clear outlier in attention**: 24.9M downloads and 1850
   likes at 0.752B parameters, Apache-2.0, updated 2025-07-26. Its sibling
   Qwen3-1.7B has ~4.1M. This is the strongest evidence of a large installed
   base that wants a *tiny* model.
3. **Base vs instruct gap**: `Qwen2.5-0.5B` (base) has 1.22M downloads while
   `Qwen2.5-0.5B-Instruct` has 5.31M — instruct-tuned tiny models are what
   people actually pull. A useful fine-tune targets the instruct-style
   distribution.
4. **Real quantization demand**: the niche searches return mostly GGUF
   re-uploads (`mradermacher/*`, `bartowski/*`), confirming that CPU/edge
   users consume small models as GGUF. A release should ship GGUF.
5. **Qwen3 ships an explicit tool-calling chat template**, so a tool-router
   fine-tune on Qwen3-0.6B inherits a valid conversation format instead of
   inventing one.

---

## 2. Niche competition (observed, `search-2026-10-24.json`)

### Function calling / tool use at ≤1B (the crowded part)
| Model | Downloads | Likes | License |
|---|---|---|---|
| Lordnyx/qwen3loop-0.6b-tool-calling-sft-v1 | 2,563 | 3 | apache-2.0 |
| ewin-reg/MiniCPM5-1B-Agentic-Tooluse-v3-GGUF | 2,774 | 23 | apache-2.0 |
| ewin-reg/MiniCPM5-1B-Agentic-Tooluse-GGUF | 1,404 | 91 | other |
| tiiuae/Falcon-H1-Tiny-Tool-Calling-90M[-GGUF] | 1,484 / 499 | 9 / 17 | other |
| MultiverseComputingCAI/LittleLamb-ToolCalling | 717 | 3 | apache-2.0 |
| Flexan/…qwen2.5-0.5B-tool-calling-GGUF | 427 | 0 | apache-2.0 |
| pmrccs/qwen3-1.7b-tool-calling-v4 | 518 | 2 | cc-by-nc-4.0 |
| nguyenthanhthuan/Llama_3.2_1B_Intruct_Tool_Calling_V2 | 481 | 8 | apache-2.0 |

### JSON extraction (thinner)
| Model | Downloads | Likes |
|---|---|---|
| Spierocho/Qwen2.5-0.5B_food_json_extraction_v0.2_d2_gguf | 92 | 1 |
| tahleels/qwen2.5-0.5b-json-extraction-qlora-dpo | 16 | 1 |
| satviksrivas7/Gemma-2-2B-it-JSON-data-extraction | 24 | 0 |
| Legeng/llama3.2-3b-json-extraction[-merged] | 0 | 1 each |

### Shell command (very thin, but almost no demand signal)
| Model | Downloads | Likes |
|---|---|---|
| louisguthmann/qwen3.5-2b-shellcommand-linux-gguf | 60 | 2 |
| vitali87/shell-commands-qwen2-1.5b-extended | 16 | 0 |
| Shojoy06/gemma4-shell-commands-finetuned-gguf | 21 | 0 |

### Observations
- The **≤1B tool-calling space is crowded but shallow**: no entry exceeds
  ~2.8k downloads, and the top ones are GGUF conversions, not the original
  fine-tune. Nothing in the list publishes a reproducible benchmark harness
  comparing against the *base* model — the exact thing a serious user wants.
- **CC-BY-NC** entries (`pmrccs`) cannot be used commercially; that is a
  differentiation opportunity for a permissive release.
- **JSON extraction** is underserved but demand evidence is weak (all ≤92 dl).
- **Shell command generation** has almost no demand signal and carries safety
  risk; deprioritized.

---

## 3. Underserved needs (evidence-based)

1. **Measured reliability, not vibes.** None of the competing ≤1B tool models
   publishes base-vs-fine-tune numbers on a held-out set. The Hub has plenty
   of *claims* and few *measurements*.
2. **Abstention.** A router that hallucinates a tool when the request needs
   none is worse than useless in an agent loop. Competitor cards do not
   advertise no-tool / clarify behaviour.
3. **Unseen schemas.** Most tiny fine-tunes are trained on a fixed tool list;
   robustness to a *new* schema at inference time is rarely evidenced.
4. **CPU-first packaging.** Downloads concentrate in GGUF, yet source repos
   rarely document llama.cpp/Ollama usage with a concrete example.
5. **Permissive license.** Several close neighbours are `other`/`cc-by-nc`.

---

## 4. Three candidate concepts, scored

Scoring 1 (weak) – 5 (strong). Weights reflect the priority order stated in
the mission (demand → differentiation → feasibility → measurability →
installability → discovery).

| Criterion (weight) | A. Tiny tool router w/ abstention | B. Schema-strict JSON extractor | C. Safe shell-command generator |
|---|---|---|---|
| 1. Demand evidence (0.25) | 4 — 24.9M base downloads; 8+ direct rivals prove demand | 3 — thin downloads, unclear pull | 2 — near-zero downloads |
| 2. Differentiation (0.20) | 4 — first ≤1B router shipping base-vs-FT harness + abstention | 4 — few rivals, easy to stand out | 3 — few rivals but thin |
| 3. Feasibility on limited compute (0.20) | 5 — 0.6B LoRA fits one 24GB GPU in <2h | 5 — same | 5 — same |
| 4. Measurable improvement (0.15) | 5 — exact-match, valid-rate, abstention F1 all objective | 4 — field-level F1 objective | 3 — subjective "did it work" |
| 5. Local install ease (0.10) | 5 — 0.6B Q4 GGUF ≈ 0.4GB, runs on CPU/Raspberry Pi | 5 — same | 5 — same |
| 6. Organic discovery (0.10) | 4 — rides Qwen3-0.6B + GGUF demand | 3 — niche audience | 2 — no audience |
| **Weighted total** | **4.35** | **3.85** | **3.20** |

### Selected: **Concept A — TinyToolRouter-0.6B**

A CPU-first, schema-strict tool router built on **Qwen/Qwen3-0.6B**
(Apache-2.0), differentiated by three things competitors do not demonstrate:

1. a **published reproducible harness** (`src/eval_harness.py`) comparing the
   fine-tune against the *unmodified base* on a leak-checked held-out set;
2. **explicit abstention** — emitting a `no_tool` decision when nothing fits;
3. **unseen-schema robustness**, measured as valid-structured-output rate on
   tool lists not seen during training.

Concept B is the fallback if Phase-4 numbers show A cannot clear the bar.
Concept C is rejected: no observed demand and avoidable safety surface.

Full design in `MODEL_SPEC.md`; dataset plan in `DATASET_CARD.md`; success
criteria in `EVALUATION_PLAN.md`.

---

## 5. Limitations of this research

- A single snapshot: Hub counters move. Re-run `src/research_hf.py` to refresh.
- The Hub API exposes `downloads`/`likes` but not active users; popularity is
  a proxy, not proof of need.
- `Salesforce/xlam-function-calling-60k` returned HTTP 401 (auth required), so
  that dataset's metadata is **unverified** here.
- We did not inspect each model card's dataset provenance or eval sections;
  niche competitor rows are download/like/license metadata only.
- No competitor was executed; capability claims in their cards are unverified.