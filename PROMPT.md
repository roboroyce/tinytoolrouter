# Mission: Build a Hugging Face Model People Actually Want

You are my lead ML researcher, model engineer, and open-source release strategist. I am using OpenClaw with DeepSeek V4.1 Flash.

## The goal

Help me create an original, genuinely useful small language model that has a realistic chance of attracting substantial organic downloads on Hugging Face.

I want an actual model release, not just a chatbot, website, or marketing plan. I want a project that provides a measurable advantage over existing models and is practical for ordinary people to run locally.

Do not promise that it will become popular. Use evidence to decide what is worth building.

## Phase 1: Research the opportunity

Research Hugging Face's current public model catalog using its API and relevant model cards.

Examine at least 15 comparable small language models, prioritizing models around 0.5B–3B parameters. Record, where available:

- Model name, parameter count, license, and intended use.
- Current downloads and likes.
- Recent updates and available quantized formats.
- Existing competing models and their main strengths.
- Specific weaknesses, unmet user needs, or underserved applications.

Never invent statistics. Record when you collected the information. Distinguish observed data from speculation.

Develop three possible model concepts and score each on:
1. Evidence of user demand.
2. Differentiation from existing releases.
3. Feasibility on limited compute.
4. Ability to demonstrate measurable improvements.
5. Ease of local installation and inference.
6. Potential for organic discovery on Hugging Face.

Choose the strongest idea based on the evidence, rather than automatically choosing a general chatbot or the first idea that comes to mind.

## Phase 2: Design the model

Consider small, permissively licensed base models such as Qwen3-0.6B, or another suitable model if research supports that choice. Verify the actual license, fine-tuning conditions, architecture, and distribution requirements before deciding.

A possible direction is a compact model that performs a narrow task exceptionally well, such as reliable tool selection, structured extraction, command interpretation, or another underserved application.

Do not assume these ideas are automatically good. Investigate competitors first.

The finished model must have an identifiable advantage that can be tested. Avoid merely renaming an existing model, making unsupported capability claims, or publishing a trivial fine-tune as a breakthrough.

## Phase 3: Build the local project

Work in the current project directory only.

Create these artifacts:

- `RESEARCH.md`: competitive landscape, evidence, and concept comparison.
- `MODEL_SPEC.md`: selected concept, intended audience, base model, and target capabilities.
- `DATASET_CARD.md`: dataset sources, licensing, provenance, limitations, and planned distribution.
- `EVALUATION_PLAN.md`: baselines, held-out test cases, metrics, and success criteria.
- `README.md`: a draft Hugging Face model card with installation and usage sections.
- `src/`: dataset preparation and evaluation scripts.
- `tests/`: lightweight tests for dataset handling, evaluation, and output validity.
- `PLAN.md`: implementation phases, prerequisites, and estimated compute requirements.

Build useful code, not just documents. Where possible, implement a reproducible dataset generator and a benchmark harness that compares the proposed model against its original base model.

Use original examples or datasets with verified compatible licenses. Keep evaluation examples separate from training data. Check for duplicates and data leakage. Document every dataset source.

Use Hugging Face's Transformers and TRL ecosystem where appropriate. Tool-calling fine-tuning should use valid tool schemas and conversation formats, not merely text that looks like JSON.

## Phase 4: Prove the model is better

Before recommending training, specify how we will measure whether the fine-tune improves performance.

Use clear metrics relevant to the selected task, such as exact-match accuracy, valid structured-output rate, correct tool selection, argument accuracy, latency, or memory use.

Compare with the unfine-tuned baseline using the same held-out tests. Include failure cases and honest limitations. Never fabricate results.

Choose a practical target that a small model can achieve, and explain what evidence would justify publishing the model as an improvement.

## Strict resource and approval rules

My AWS account has approximately $107.83 in estimated remaining credits. Protect this balance.

For this phase:

- Do not launch or modify EC2 instances, SageMaker endpoints, training jobs, or Hugging Face paid compute jobs.
- Do not request quota increases, start paid services, or purchase anything.
- Do not download large model weights or start training.
- Do not upload, publish, or modify any remote Hugging Face repository.
- Do not use API credits or other paid services.
- Do not read, print, copy, or expose API keys, access tokens, or other secrets.
- Do not delete files outside this project.

Read-only metadata requests are acceptable. Prefer lightweight local code and existing tools. Inspect hardware only with safe, read-only commands where appropriate.

Before any training or upload, give me an explicit cost estimate, required hardware, likely failure risks, and a proposed execution plan. Stop and wait for my approval before proceeding.

## Working style

Do not stop after writing a plan if you can safely implement local research, code, tests, or documentation.

Check whether relevant files already exist before overwriting them. Keep work organized and reproducible. Run lightweight tests and fix straightforward problems.

If network access, credentials, or tools are unavailable, explain the limitation and continue with what can be done locally. Never pretend a task succeeded when it did not.

## Final report

When the initial work is complete, tell me:

1. Which concept you selected and why.
2. What evidence supports it.
3. What files and working code you created.
4. Which tests passed or failed.
5. Which base model and datasets you recommend, with license evidence.
6. What remains before a real fine-tune and public release.
7. The next command I should run.
8. The exact proposed cost before any paid compute is used.

Start by inspecting the current project directory and verifying that network access to Hugging Face metadata works. Then begin the research and implementation.
