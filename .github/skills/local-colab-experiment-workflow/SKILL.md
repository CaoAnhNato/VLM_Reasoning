---
name: local-colab-experiment-workflow
description: Prepare, execute, or review authorized local smoke tests and larger Google Colab experiments; capture reproducibility evidence and enforce the experiment approval gate.
---

# Local and Colab Experiment Workflow

## Input

Collect the approved experiment specification, objective, method matrix, sample manifest, dataset and annotation versions, model/checkpoint, code/configuration, evaluator/reference protocol, target environment, resource constraints, and output location. Use [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md) as the experiment-design reference and [PROJECT.md](../../../PROJECT.md) / [ORIGINAL_REQUEST.md](../../../ORIGINAL_REQUEST.md) for observed execution constraints.

## Workflow

1. **Check the gate before execution.** For a significant experiment present objective, methods, sample size and exact manifest, dataset, model/config, metrics/evaluator, expected artifacts, resource requirements, and limitations. Wait for explicit Project Owner authorization. Never initiate GPU-intensive or long-running runs from a plan alone.
2. **Resolve the environment.** Confirm whether execution is local or Colab, active Python/environment and package versions, entry point, model/data paths and integrity, output path, available CPU/GPU/memory, and access permissions. Do not install packages or alter environments unless authorized.
3. **Local verification.** Use local execution for environment compatibility, debugging, and small pipeline checks. This repository explicitly restricts local runs to synthetic/mock data and prohibits local real-model inference; use the existing mock/adversarial tests and synthetic fixtures. The plan's 20-sample integration gate must not be interpreted as permission to load real weights locally. Ask the Project Owner to specify an authorized remote location if a real-model integration run is required. Do not call a mock smoke test a benchmark.
4. **Colab preparation.** Before an authorized larger run, verify the notebook/runtime, GPU availability and model compatibility, dataset access/checksums, exact manifest, methods/configurations, evaluator and reference version, storage/output location, and resource estimate based on measured pilot runs. The repository scan found no checked-in Colab notebook or dedicated Colab procedure; a Vast.ai guide is not a Colab guide. Obtain the actual approved notebook/procedure rather than inventing it. Never transfer API keys, tokens, private credentials, or unrelated sensitive local files to Colab.
5. **Fair execution.** Use the same selected images and associated questions for comparable methods. Keep model/checkpoint, prompt, preprocessing, decoding and evaluator consistent where applicable; document method-required deviations. Record failures and skipped samples, actual time and resource use, and never silently omit a failed run.
6. **Preserve evidence.** Save exact commands, configuration and prompt, source revision/dirty state, dependency versions, model identifier/revision, dataset/annotation/manifest hashes, seed, device/dtype, logs, warnings/errors, raw predictions, evaluation outputs, failures, and artifact paths. Use separate method/run output directories. Do not overwrite historical outputs without authorization.
7. **Classify conclusions.** Report smoke-test, development/screening, and independent evaluation separately. The 400-image collection has already informed method development; the plan's 100-image pilot is screening. A result is independently reproduced only when underlying artifacts and execution evidence support that statement.

## Output

Before execution: a concise authorization request containing all gate fields. After execution: environment and command, configuration/provenance, manifest and completion counts, actual test/run status, errors/skips, runtime/resources, output paths, and limitations. If no run occurred, state that plainly.

## Error and Escalation

Stop and report blockers for absent authorization, ambiguous method/sample/evaluator, missing manifest, unavailable notebook/runtime, insufficient hardware, or integrity mismatch. Escalate the 20-sample location and plan approval ambiguity to the Project Owner. Do not substitute a different dataset, method, evaluator, or provider without approval.

## References

- Project rules: [`.github/copilot-instructions.md`](../../copilot-instructions.md)
- Experiment design and prerequisites: [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md)
- Local-only mock restriction and recorded architecture: [PROJECT.md](../../../PROJECT.md), [ORIGINAL_REQUEST.md](../../../ORIGINAL_REQUEST.md)
- Existing remote execution guidance (Vast.ai, not Colab): [HUONG_DAN_CHAY_VASTAI.md](../../../docs/HUONG_DAN_CHAY_VASTAI.md)
- Existing test runner: [`tests/run_e2e_tests.py`](../../../tests/run_e2e_tests.py)