---
name: research-plan-compliance-review
description: Review implementation proposals, baseline configurations, experiments, datasets, metrics, or result interpretation for compliance with Research Plan v2 without inventing or changing research methodology.
---

# Research Plan Compliance Review

## Input

Gather the proposed/implemented change, task acceptance criteria, relevant code/configuration/data/results, and [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md). Note the plan's researcher-approval status and any required decision record. Treat `PROJECT.md` and older research documents as repository evidence, not replacements for Research Plan v2.

## Workflow

1. Establish which plan requirements apply and inspect the actual implementation/artifacts. Distinguish implementation observation from proposal and reported history.
2. Check method identity: SG-VCD remains two-stage (spatial localization, then contrastive decoding); CERV remains a separate three-stage method unless the Project Owner explicitly changes scope.
3. Check condition classification: Stage 1 localization, Stage 2 decoding, region-conditioned contrastive, end-to-end, proposed SG-VCD, controls, and diagnostics remain distinct. Vanilla Vintern is `CTRL-0`, not a paper-method baseline. Oracle ROI is diagnostic only. Ablations are not automatically baselines; the former five-condition exploratory pilot is not the main comparison.
4. Check the three primary metrics remain Token-F1, ANLS, and Substring Match. Verify evaluator version and consistency across methods, including normalization, tokenization, multi-reference handling, match rules, and thresholds. Identify any different evaluator explicitly; never combine its scores silently.
5. Check dataset integrity: the 400-image/1,222-QA collection has informed development; the planned 100-image pilot is screening, not an independent test. Verify image/question IDs, common manifest, all associated QA, answer references, annotation provenance, failed/skipped records, and absence of silent filtering or GT answer/box leakage in ordinary inference. Keep Oracle ROI results separate.
6. Check run comparability and provenance: model/checkpoint, prompt, preprocessing, decoding, evaluator, reference protocol, seed, manifest, and outputs are aligned or deviations documented. Keep reported historical results distinct from independently reproduced results; flag inconsistencies and missing raw evidence.
7. Check approval gates. Feasibility/evaluator/annotation/manifest and integration prerequisites must be met before a pilot; explicit Project Owner authorization is required for significant runs. Respect the repository's mock-only local restriction. Do not resolve the plan header's researcher-approval ambiguity by assumption.
8. Record each requirement as satisfied, violated, or unverified with concrete evidence. Refer methodological choices and interpretations to ChatGPT and the Project Owner; this skill verifies compliance, it does not propose a new method or revise the experiment plan.

## Output

Report compliance status; requirements satisfied; violations and required corrections; unverified assumptions; evidence with file/function/config/test/artifact references; and items requiring ChatGPT or Project Owner review. State clearly if evidence is insufficient to establish compliance.

## Error and Escalation

If the research plan is missing, stop methodology-specific conclusions and report the required filename. If code, manifests, provenance, or protocol evidence is inaccessible, mark affected requirements unverified or the review blocked. Escalate unresolved design choices and conflicting approval/status language; never invent a decision.

## References

- Project rules: [`.github/copilot-instructions.md`](../../copilot-instructions.md)
- Primary design reference: [Research Plan v2](../../../docs/research/SG_VCD_Research_Review_Consolidated_Draft_v2.md)
- Reported/implemented project context: [PROJECT.md](../../../PROJECT.md), [SG-VCD method report](../../../docs/Proposed_Method_SG_VCD.md), [CERV proposal](../../../docs/Proposed_Method_CERV.md)
- Existing evaluator and tests: [`scripts/benchmark/metrics.py`](../../../scripts/benchmark/metrics.py), [`tests/`](../../../tests/)