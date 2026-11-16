# Project Instructions

## Project and Authority

This repository studies Vietnamese scene-text visual question answering (ViTextVQA), with SG-VCD as the principal method. Treat [Research Plan v2](../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md) as the research-design reference and [PROJECT.md](../PROJECT.md) as a record of project architecture and prior work. They describe different things: a planned or reported item is not proof that it is implemented or independently verified. Preserve that distinction in code, reports, and handoffs.

SG-VCD is a two-stage method: (1) spatial grounding/localization and (2) contrastive decoding with its designated positive and negative visual inputs. CERV is a separate three-stage method, with code under `scripts/benchmark/cerv/`; do not merge it into SG-VCD or change either method's identity without Project Owner approval. The repository has SG-VCD modules under `scripts/benchmark/sg_vcd/`, evaluator/metrics at `scripts/benchmark/evaluate_vitextvqa.py` and `scripts/benchmark/metrics.py`, tests under `tests/`, and run artifacts under `results/`. Do not infer that every method proposed in the plan is implemented.

## Collaboration and Scope

- **Project Owner:** decides research direction, approves scope and experiment gates, and accepts/rejects deliverables.
- **ChatGPT:** reviews scientific reasoning, methodology, and experiment design when needed.
- **VS Code Coding Agent:** maintains these rules and skills, performs independent technical QA, and reports evidence and uncertainty. Do not trust an implementer's completion summary in place of inspecting the work.
- **Antigravity (Anti):** implements only from an approved task specification and returns the handoff package required by the quality-review skill.

Required handoff: `Approved Specification -> Anti Implementation -> VS Code Technical QA -> Anti Revision (if needed) -> ChatGPT Research Review -> Project Owner Acceptance`.

Stay within the approved task. Avoid unrelated refactors, unapproved research-method changes, unnecessary dependencies, and deletion of historical artifacts. Do not modify dataset/annotation contents without explicit authorization. Preserve interfaces and working behavior unless the task requires a change. Do not commit credentials, secrets, or avoidable machine-specific paths. Do not install dependencies, commit, push, or perform destructive actions unless the task explicitly authorizes them.

## Research Invariants

- Keep SG-VCD two-stage and CERV separate. Baselines must remain classified as Stage 1 localization, Stage 2 decoding, region-conditioned contrastive, end-to-end comparisons, proposed SG-VCD, or controls/diagnostics as specified in Research Plan v2. Vanilla Vintern is `CTRL-0`, not a paper-method baseline. Oracle ROI is diagnostic only. Do not reinstate the earlier five-condition exploratory pilot as the main paper-method comparison or relabel ablations as paper baselines.
- Preserve Token-F1, ANLS, and Substring Match as the three primary answer metrics. Do not silently change normalization, tokenization, reference aggregation, matching rules, or thresholds. Identify evaluator versions and do not mix their scores.
- The 400-image/1,222-QA collection has informed SG-VCD development. Treat the planned 100-image pilot as development/screening, not independent test evidence. Preserve IDs, references, annotation provenance, and a common image/question manifest. Do not drop samples silently or expose ground-truth answers/boxes to normal inference. Keep Oracle ROI separate.
- Treat historical results as reported unless raw code/configuration, predictions, logs, and evaluation can independently substantiate them. Flag discrepancies rather than choosing an unsupported value.
- Use consistent model/checkpoint, prompts, preprocessing, evaluator, reference protocol, and manifest in comparisons unless a paper requires a documented deviation. Record enough configuration, environment, seed, provenance, outputs, failures, and resource usage to identify each run.
- Distinguish verified facts, implementation observations, assumptions, proposals, and unresolved questions. Do not fabricate results, runtime, utilization, or compatibility.

## Implementation and Execution

Follow the observed module boundaries above and nearby code/test conventions; do not invent a framework or architecture. Keep modules focused, handle errors, make configuration reproducible, preserve traceable outputs, and add relevant tests. No secrets; avoid hardcoded local paths.

Local use is for environment checks, debugging, and small pipeline verification. `ORIGINAL_REQUEST.md` and `PROJECT.md` explicitly restrict local execution to synthetic/mock data and prohibit loading real 1B model weights or running the real benchmark locally. Follow that stricter rule: do not run real local inference. The plan's 20-sample integration gate does not override it; clarify with the Project Owner where that real-model gate is to run. Use Google Colab only for authorized, resource-intensive runs; the repository scan found no Colab notebook or dedicated Colab run guide. A separate Vast.ai guide exists and is not a Colab procedure. Take sample size and GPU configuration from the approved experiment specification. Never launch GPU-intensive or long-running work without explicit authorization.

Before a significant run, present objective, methods, sample size/manifest, data, model/configuration, metrics, outputs, expected resource needs, and limitations; wait for Project Owner authorization. Local and remote workflows are detailed in the `local-colab-experiment-workflow` skill.

## Required QA Gate

Whenever Anti reports completion of a coding or technical implementation task, use `anti-output-quality-review` before acceptance. Independently inspect actual diffs/files, configuration, dependencies, interfaces, removed files, tests, and execution evidence. Run safe, relevant targeted checks when feasible; never describe unrun tests as passed. Do not repair Anti's implementation during review unless assigned. Report `PASS`, `PASS WITH WARNINGS`, `FAIL`, or `BLOCKED`, evidence-backed severity findings, corrections, and technical acceptability. For changes affecting methods, baselines, metrics, data, experiments, or result interpretation, also use `research-plan-compliance-review`; escalate methodological uncertainty to ChatGPT and the Project Owner.

## Approval Ambiguity

Research Plan v2 is the required research reference for this governance task, but its header says “Revised research plan for researcher review; execution follows researcher approval,” and its final status says `READY_FOR_RESEARCHER_REVIEW`. Do not infer that experiment execution has been approved. Request the Project Owner/researcher's confirmation before a run or before describing the plan as formally approved for execution.