# PLAN.md — implementation phases, prerequisites, compute

Project root: `/home/caligrex/projects/hf-model-lab`. All artifacts stay inside it.

Legend: ☑ done · ☐ to do · ⛔ blocked/awaiting

---

## Phase 0 — Environment & access verification ☑
- ☑ Local profile: 4 CPU, 7.7 GiB RAM, no GPU.
- ☑ HF metadata API reachable (read-only).
- ☑ AWS identity verified: account 606938392482, region `us-east-1`
  (assumed via the root login session; no credentials printed).
- ☑ No pre-existing EC2 instances. G/VT On-Demand quota = 4 vCPU
  (one `g5.xlarge`/`g6.xlarge` fits). Spot quota = 0 → Spot not used.
- ☑ Pricing verified via the AWS Pricing API: `g5.xlarge` = **$1.006/h**,
  `g6.xlarge` = **$0.8048/h** (us-east-1, On-Demand, Linux).

## Phase 1 — Research ☑ (`RESEARCH.md`, `research/*.json`)

## Phase 2 — Design ☑ (`MODEL_SPEC.md`; base `Qwen/Qwen3-0.6B`, Apache-2.0)

## Phase 3 — Build ☑ (v2)
- ☑ `src/schema_utils.py` — schemas, extraction, validation, scoring with the
  **json_parsed vs schema-valid** split (bug fix).
- ☑ `src/gen_dataset.py` v2 — 9 seen tools + 5 unseen tools, all JSON types,
  paraphrases, `no_tool`, disjoint templates/pools/tools; 600 train / 231
  held-out; `leakage_check: pass`.
- ☑ `src/eval_harness.py` — `heuristic`/`gold`/`transformers` backends, family
  slicing, `no_tool` precision/recall, `json_parse_rate`.
- ☑ `src/train_lora.py` — PEFT LoRA SFT with prompt masking + `run_meta.json`.
- ☑ `src/publish_hf.py` — refuses to publish unless the release gate passes.
- ☑ `scripts/remote/bootstrap.sh`, `scripts/remote/make_gguf.sh`.
- ☑ 43 unit tests, all passing.

## Phase 4 — Run remotely ☑/☐
- ☑ S3 bucket `tinytoolrouter-606938392482-us-east-1` (private, AES256, tagged).
- ☑ Worker launched: `g5.xlarge` (A10G 24GB), private SG (no inbound),
  encrypted 100 GB gp3 with delete-on-termination,
  `InstanceInitiatedShutdownBehavior=terminate`, in-instance 168-min deadline.
- ☐ Baseline eval (running) → `reports/base.json`
- ☐ LoRA training → `artifacts/tinytoolrouter-0.6b/`
- ☐ Fine-tuned eval → `reports/finetune.json`
- ☐ GGUF export (best effort)

## Phase 5 — Release ⛔
- ☐ Publish **iff** the gate passes (`valid_structured_output_rate ≥ 0.95`,
  exact-match gain ≥ +10 pts, ≥ 200 held-out) **and** HF auth is available.
  HF auth is **not currently available locally** (no token; `huggingface_hub`
  not installed) — publishing is blocked unless the owner provides a token.

## Cleanup ☐
- ☐ Copy artifacts back to the project directory.
- ☐ Terminate the worker (self-terminates via deadline; verify).
- ☐ Delete the temporary S3 bucket after artifacts are copied.
- ☐ Record runtime, estimated charges, cleanup actions.

## Compute & cost (verified pricing)
| Item | Rate | Planned |
|---|---|---|
| g5.xlarge On-Demand | $1.006/h | ≤ 3 h ⇒ **≤ $3.02** |
| 100 GB gp3 (encrypted) | $0.08/GB-mo | ~$0.03 |
| S3 (small objects) | ~$0.023/GB-mo | < $0.01 |
| **Total estimate** | | **≈ $3.1, cap $15** |

## Risks
| Risk | Mitigation |
|---|---|
| Generation is slow (~4 s/case) | budget allows; monitor stage-by-stage |
| Training doesn't clear +10 pts | publish harness + honest negative result |
| GGUF conversion fails | optional; guarded, never blocks the main result |
| HF auth missing | handled: report blocker, keep artifacts locally |
| Worker overrun | 168-min in-instance timer + `terminate` shutdown behavior |