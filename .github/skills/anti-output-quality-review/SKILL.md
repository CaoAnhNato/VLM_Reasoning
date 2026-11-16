---
name: anti-output-quality-review
description: Mandatory independent technical QA whenever Antigravity (Anti) reports completion of a coding or technical implementation task; inspect evidence, verify relevant tests, check scope and research compliance, and report acceptance status.
---

# Anti Output Quality Review

## Input

Collect the original approved task/specification and acceptance criteria, applicable project instructions and skills, relevant requirements in [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md), Anti's completion handoff, and access to the actual worktree, tests, logs, and outputs. If the task or handoff is missing, record that limitation; do not infer acceptance criteria.

Anti's handoff should state: task summary; created/modified/deleted files; implementation decisions; dependency changes; commands and actual test results; limitations and unfinished requirements; output locations.

## Workflow

1. **Understand the task.** Extract acceptance criteria and relevant project constraints. Determine whether `research-plan-compliance-review` is also required.
2. **Inspect changes.** Review the actual Git diff when available, changed/new/deleted files, configuration, dependencies, interfaces, and unrelated changes. Inspect the worktree even if Anti's summary is convincing. Never discard user changes.
3. **Verify implementation.** Trace the requested behavior through its actual call path. Check input/output contracts, data handling, error paths, configuration, integration into the pipeline, and consistency with observed repository architecture. Identify missing or disconnected functionality.
4. **Verify tests and evidence.** Inspect relevant tests and run safe, targeted tests when feasible. Do not launch training, large inference, or destructive commands for QA. Classify each check as executed/passed, executed/failed, not executed, blocked (state why), or static inspection only. A plausible output or static review does not replace a failed test.
5. **Check research compliance where relevant.** Use the dedicated compliance skill for architecture, baselines, metrics, dataset, experiment, or result changes. Check for GT leakage, manifest/reference drift, unsupported claims, and mixing evaluator versions. Escalate unresolved scientific choices; do not change methodology.
6. **Write an evidence-based report.** Set one status: `PASS`, `PASS WITH WARNINGS`, `FAIL`, or `BLOCKED`. Each finding must include ID, severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), file plus line/function, evidence, impact, and recommended correction. Separate mandatory follow-up from optional improvements.
7. **Decide acceptance.** Explicitly state whether the deliverable is technically acceptable. On `FAIL`, give Anti a structured correction list. On `BLOCKED`, name the missing permission, information, dependency, data, or environment. On warnings, say which are blocking acceptance, if any.
8. **Review revisions afresh.** Revisit each prior finding, inspect the new diff, run relevant regression checks, and look for regressions. Do not accept a revision based on Anti's assertion alone.

Do not modify Anti's implementation during a review unless the Project Owner explicitly assigns a repair. Never hide failed checks, imply unrun checks passed, or approve solely because output looks plausible.

## Output

Provide task/specification reference; changed-file and scope summary; status; evidence-backed findings; checks and actual outcomes; research-compliance result when applicable; blockers/assumptions; required revisions; and explicit technical-acceptability decision. Keep claims tied to file/function, test result, log, configuration, or artifact evidence.

## Error and Escalation

Use `BLOCKED` when review cannot be completed due to unavailable task criteria, permissions, environment, dependencies, or evidence. Use `FAIL` when an acceptance criterion is violated. Escalate research-method or interpretation uncertainty to ChatGPT and the Project Owner. Do not silently resolve it or expand review scope into implementation work.

## References

- Project rules: [`.github/copilot-instructions.md`](../../copilot-instructions.md)
- Research requirements: [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md)
- Observed architecture and historical project record: [PROJECT.md](../../../PROJECT.md)
- Test suite: [`tests/`](../../../tests/)