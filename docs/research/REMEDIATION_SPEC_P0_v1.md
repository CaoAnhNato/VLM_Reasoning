# P0 Remediation Specification and Acceptance-Test Design

**Version:** 1.1-draft  
**Date:** 2026-11-05  
**Status:** Design only; pending Project Owner and ChatGPT review where specified  
**Scope:** R-01 evaluation protocol, R-02 annotation/reference integrity, R-03 SG-VCD contrastive formula, R-04 result/failure and spatial semantics.

This document specifies proposed implementation contracts and acceptance tests. It does not authorize or implement code, change data, select unresolved denominators/ROI aggregation, or authorize an experiment. Research Plan v2 is the Owner-approved working design for this task. No separate external leader approval is asserted.

## Status Labels

- `OWNER-APPROVED`: Binding decision explicitly supplied by the Project Owner in Task 04.
- `SOURCE-VERIFIED`: Directly established from current repository files.
- `RECOMMENDED`: Proposed technical approach, not yet approved.
- `PENDING-OWNER-APPROVAL`: Requires a Project Owner decision.
- `PENDING-CHATGPT-REVIEW`: Requires methodological review.
- `UNVERIFIED`: Repository evidence is missing or insufficient.
## Approved Decisions

| ID | Status | Binding decision |
|---|---|---|
| D-01 | `OWNER-APPROVED` | Strict is the provisional primary evaluator; Token-F1, ANLS, and Substring Match remain primary. Keep Round 7 separately named and never mix unlabeled scores. |
| D-02 | `OWNER-APPROVED` | Preserve original, added, reviewed, adjudicated, rejected, and disputed references distinctly. Bbox availability and non-empty `verified_all_answers` do not establish answer adjudication. Keep count units distinct. |
| D-03 | `OWNER-APPROVED` | Canonical SG-VCD uses the signed formula. ReLU-clipped scoring is a separate historical variant; preserve its results and post-calibration behavior. |
| D-04 | `OWNER-APPROVED` | Local is mock/synthetic only. Real-model integration and larger comparisons are Colab-only with required approvals. No execution is authorized by this task. |

The Project Owner has approved Research Plan v2 as the working plan. The approval-ambiguity wording in existing rules is stale relative to this decision and is not changed here. The Local mock-only rule remains in force.

## R-01: Strict Evaluation Protocol

### Current Strict Definitions

Source of truth: [metrics.py](../../scripts/benchmark/metrics.py#L33), not metric names or historical tables.

| Metric | Exact current behavior (`SOURCE-VERIFIED`) |
|---|---|
| `normalize_answer(text)` | Input is annotated `Any`; `None` becomes `""`; otherwise cast to `str`, Unicode NFC, lowercase, replace characters in Python `string.punctuation` (ASCII punctuation) with spaces, then collapse/split/join whitespace. Diacritics are preserved; Unicode punctuation outside `string.punctuation` is not replaced. |
| Token-F1 | Prediction `str`, references sequence; a reference `str` is wrapped into a list. Normalize both, whitespace-split, use multiset `Counter` overlap, conventional precision/recall harmonic mean; take maximum across non-empty references. No substring/containment/length-relaxation override. Empty normalized prediction, empty reference list, or no non-empty matching reference scores `0.0`. Output is `float` in `[0,1]`. |
| ANLS | Prediction `str`, references sequence, `tau=0.5`. Normalize as above; character-level Levenshtein similarity `1 - distance/max(lengths)`; retain a reference score only when similarity `>= tau`; take maximum. Empty normalized prediction, empty reference list, or empty references scores `0.0`. Output is `float` in `[0,1]`. |
| Substring Match | Normalize both sides; return `1.0` iff any non-empty normalized reference is a contiguous substring of normalized prediction (`reference in prediction`), otherwise `0.0`. Single reference `str` is wrapped; empty prediction/references score `0.0`. Matching is one-way, not symmetric. |
| Multi-reference handling | Token-F1, ANLS, and Substring Match independently choose the maximum reference score. Order does not affect a max; source does not deduplicate first. |
| Current aggregate | `MetricsAggregator` increments a shared count on every update and takes question-level arithmetic means. Missing keys currently act numerically as zero because the shared count remains the denominator. This is an observed implementation defect/limitation, not the approved output contract below. |

Supporting code: [`compute_token_f1`](../../scripts/benchmark/metrics.py#L109), [`compute_anls`](../../scripts/benchmark/metrics.py#L144), [`compute_substring_match`](../../scripts/benchmark/metrics.py#L170), [`compute_stage2_metrics`](../../scripts/benchmark/metrics.py#L191), and [`MetricsAggregator`](../../scripts/benchmark/metrics.py#L393).

### Protocol Identity and Metric Applicability

`RECOMMENDED` Strict ID: `vitextvqa.strict.v1`. A score-changing definition change creates a new immutable protocol ID. Every per-question record, checkpoint, summary, and export carries `metric_protocol_id`, implementation revision/hash, and `reference_set_id`/hash. Mixed IDs must be separately grouped or rejected.

Each metric value uses one envelope: `computed` with a numeric value (including true zero), `not_applicable`, `unavailable`, or `pending_policy`; non-computed states carry a reason and no numeric value.

- Strict answer metrics apply to answer-producing conditions. A valid answer plus selected references permits computation. Missing references or an invalid returned answer is `unavailable`; failed/missing/skipped sample treatment is `pending_policy` pending the R-04 denominator decision. The current Vintern control is answer-producing.
- Spatial metrics apply to localization-producing conditions, including failed localizations (which are `unavailable`). A valid localization under approved policy is `computed`; true zero overlap remains numeric `0.0`. A non-localizing method is `not_applicable`. An unapproved ROI aggregation is `pending_policy`.
- Per metric report `expected_manifest_count`, `applicable_count`, `computed_count`, `not_applicable_count`, `unavailable_count`, `pending_policy_count`, and `numeric_denominator`. Numeric denominator equals computed count; only computed values enter the conditional mean. Required invariants are `expected_manifest_count = applicable_count + not_applicable_count` and `applicable_count = computed_count + unavailable_count + pending_policy_count`; therefore expected also equals computed + unavailable + pending + N/A. Failed/N/A/pending values never become zero silently.

### Round 7: Documented Behavior and Gaps

The report ([Proposed_Method_SG_VCD.md](../Proposed_Method_SG_VCD.md#L1506)) documents a concise prediction bound; symmetric ANLS/F1 containment and sub-span relaxations; an ANLS sub-span threshold of `0.85`; numeric/unit normalization examples; and defenses against paragraph dumps and very short substring matches.

`UNVERIFIED`/incomplete Round 7 details that must not be invented: span enumeration and length bounds; conflict between the displayed `OR` concise-bound formula and prose about a hard 25-token cap; complete numeric/unit mapping; Substring Match direction/definition; full tokenization, Unicode and punctuation rules; all empty-value/reference aggregation rules; and whether this documented evaluator generated the reported R7 scores. [test_adversarial_metrics_r7.py](../../tests/test_adversarial_metrics_r7.py#L19) imports helpers absent from current `metrics.py` and has expansion/concise-answer assertions conflicting with [test_metrics_relaxation.py](../../tests/test_metrics_relaxation.py#L47). Neither file proves Round 7 executes or passes.

### Historical Comparability

| Result class | Strict comparison status |
|---|---|
| Current Strict result with fixed protocol and reference IDs | Comparable to another result using the same IDs and implementation. |
| Earlier rounds before the reported R7 evaluator change | Not directly comparable to R7; exact run code/config/protocol provenance is needed. |
| Reported R7 Token-F1/ANLS | Not comparable to Strict absent exact R7 behavior and provenance. |
| Reported R7 Substring Match | Not certified comparable; R7 matching behavior is incompletely documented. |
| Historical predictions rescored with Strict | Potentially comparable only with identical prediction records, selected reference snapshot, and Strict implementation; not yet rescored. |
| Historical results lacking run provenance | Reported/partially traceable, not independently reproduced. |

### R-01 Acceptance-Test Matrix

Tests are designs only; none were run. Follow existing `unittest` style in [test_metrics_relaxation.py](../../tests/test_metrics_relaxation.py#L26) and [test_tier1_features.py](../../tests/test_tier1_features.py#L229).

| Test ID / Ticket | Objective | Preconditions / Input fixture | Expected behavior | Failure condition | Level | Dependencies |
|---|---|---|---|---|---|---|
| R01-T01 / R-01 | NFC/case/punctuation/whitespace | `normalize_answer("  Nhà Thuốc Minh-An!  ")` | Exactly `"nhà thuốc minh an"`. | Different result. | Unit | Strict v1 |
| R01-T02 / R-01 | Diacritics | `"Thái"` vs `"Thai"` | Normalizations differ; Token-F1/Substr `0`; ANLS is computed by Strict character edit rule (`0.75` for these four-character forms). | Marks stripped or value differs. | Unit | Strict v1 |
| R01-T03 / R-01 | Token multiset | Pred `"red red"`; ref `"red blue"`. | Precision/recall/F1 each `0.5`. | Containment override or different score. | Unit | Strict Token-F1 |
| R01-T04 / R-01 | Token boundary | Pred `"red/blue"`; ref `"red blue"`. | ASCII slash becomes whitespace; Token-F1 `1.0`. | Slash is joined or retained as token. | Unit | Strict normalization |
| R01-T05 / R-01 | ANLS inclusive threshold | `"ab"` vs `"ac"`, `tau=.5`. | Similarity `0.5` is retained. | Threshold is exclusive or score differs. | Unit | Strict ANLS |
| R01-T06 / R-01 | ANLS below threshold | `"abc"` vs `"xyz"`, `tau=.5`. | Score `0.0`. | Nonzero result. | Unit | Strict ANLS |
| R01-T07 / R-01 | Best-reference order | Pred `"red blue"`; refs `"red green"`, `"red blue"`, then reversed. | Each metric returns its max score invariant to order. | Order changes score or references are averaged. | Unit | Strict multi-ref |
| R01-T08 / R-01 | Substring direction | Pred `"Thái"`, ref `"nồi lẩu thái siêu to"`, then reverse containment. | First score `0`; reverse score `1`. | Symmetric matching. | Unit | Strict Substring |
| R01-T09 / R-01 | Empty values | Empty/whitespace/None predictions and empty/empty-string refs. | Metric outputs `0.0`; no exception. | Nonzero, NaN, or exception. | Unit | Strict empty behavior |
| R01-T10 / R-01 | NFC/NFD | Equivalent NFC/NFD Vietnamese phrase. | Same normalized string. | Different normalized values. | Unit | Unicode NFC |
| R01-T11 / R-01 | Question-level mean | Two computed F1 values `0`, `1`. | Mean `0.5`, denominator `2`. | Other weighting. | Unit/contract | Aggregator |
| R01-T12 / R-01 | Protocol persistence | Strict fixture through exports/checkpoint. | Same protocol implementation and reference IDs in every output. | Missing/mismatched IDs. | Contract | R-02/R-04 |
| R01-T13 / R-01 | No protocol mixing | Strict and R7 records. | Separate partitions or explicit reject. | Unlabeled pooled score. | Contract | Protocol registry |
| R01-T14 / R-01 | Incomplete R7 refusal | Request R7 when required rule unspecified. | Refuse/unavailable; no guessed rule. | Guessed span or substring behavior. | Contract | Research approval |
| R01-T15 / R-01 | Genuine zero | Computed F1 `0` and `1`. | Both are computed; denominator `2`, mean `.5`. | Zero dropped or treated as N/A. | Unit | Metric envelope |
| R01-T16 / R-01 | N/A denominator | Computed `1`; one N/A row. | Expected `2`, computed `1`, N/A `1`, denominator `1`, mean `1`. | N/A becomes zero. | Contract | Applicability |
| R01-T17 / R-01 | Unavailable denominator | Computed `1`; one unavailable applicable row. | Counts computed `1`, unavailable `1`, denominator `1`, mean `1`; reason retained. | Unavailable becomes zero/disappears. | Contract | R-02 refs |
| R01-T18 / R-01 | Pending denominator | One failure-policy-pending row. | Pending count `1`; excluded from numeric mean; no score. | Converted to zero/N/A or omitted. | Contract | R-04 decision |
| R01-T19 / R-01 | Metric-specific applicability | Answer-only control and localizing method on same manifest. | Answer metrics applicable to both; spatial N/A for control, independently evaluated for localizer. | Shared applicability incorrectly imposed. | Contract | Method registry |
| R01-T20 / R-01 | Count reconciliation | Four rows: computed, N/A, unavailable, pending. | Expected `4`; each state count `1`; applicable `3`; denominator `1`. | Counts do not reconcile. | Contract | Metric envelope |

## R-02: Annotation and Reference Integrity

### Verified Data Flow

1. [run_swin_auto_annotate.py](../../scripts/benchmark/run_swin_auto_annotate.py#L104) prefers `data/vitextvqa/ViTextVQA_dev.json`; if absent, it can read QA fields from tracked prediction JSONL ([same file](../../scripts/benchmark/run_swin_auto_annotate.py#L165)). The original QA source file is absent in this checkout, so the provenance of the current original references is not independently established.
2. That script stores total and matched detector/localization QA counts as `total_quests`, `matched_quests`, and `quest_match_rate_pct` ([same file](../../scripts/benchmark/run_swin_auto_annotate.py#L661)). Current master: 1,222 question rows, 1,153 initial matches, 69 unmatched, 94.35% initial match rate.
3. `setup_xanylabeling_workspace.py` groups records by question-index/workspace structure and counts unique `quest_idx` values, yielding 1,213 grouped workspace quest indices. Its manifest also records 69 unmatched localization records. These are not the same count unit as the 1,222 canonical question rows ([setup_xanylabeling_workspace.py](../../scripts/benchmark/setup_xanylabeling_workspace.py#L471)). The nine-record difference remains unexplained at the individual-record level and must not be normalized away.
4. During merge, the script finds boxes by question ID/group or label. `verified_quests` increments when `matched_boxes` is nonempty; `still_missing_quests` increments otherwise. It writes `verified_quests` and recomputes `quest_match_rate_pct` from that box count but leaves original `matched_quests` untouched ([same file](../../scripts/benchmark/setup_xanylabeling_workspace.py#L721), [same file](../../scripts/benchmark/setup_xanylabeling_workspace.py#L800)). Current metadata can therefore say `matched_quests=1153`, `verified_quests=1222`, `still_missing_quests=0`, and `quest_match_rate_pct=100.0`. These fields measure different phases/units; 100% is not evidence of answer adjudication.
5. `extract_answers_from_shape` obtains text from X-AnyLabeling descriptions and unconditionally splits the extracted text on `/` ([same file](../../scripts/benchmark/setup_xanylabeling_workspace.py#L646)). The source notation also uses slash-separated answers. The implementation cannot distinguish an actual separator from a literal slash in a date, amount/unit, URL/path, or answer without additional structure.
6. Merge retains original `gt_answers` and populates `added_gt_answers` and `verified_all_answers` only when new answer strings are found ([same file](../../scripts/benchmark/setup_xanylabeling_workspace.py#L785)). Current canonical file has 79 rows with non-empty `verified_all_answers` and 1,143 without. This count is a count of rows with explicit added sets, not all human-reviewed rows; no per-reference reviewer/adjudication log was found.
7. The evaluator maps `verified_all_answers` to a sample field, falling back to `gt_answers` where it is absent/empty ([evaluate_vitextvqa.py](../../scripts/benchmark/evaluate_vitextvqa.py#L343)); `compute_stage2_metrics` repeats a non-empty verified-or-original fallback ([metrics.py](../../scripts/benchmark/metrics.py#L223)). This fallback conceals the difference between “no adjudicated reference” and “adjudicated set equal to original.”

Current aggregate counts (`SOURCE-VERIFIED`):

| Count unit | Count/status | Meaning and limit |
|---|---:|---|
| Canonical master question records | 1,222 | Rows in the 400-image canonical master; unique question IDs/sample keys were observed in prior read-only audit. |
| Images and workspace image/JSON pairs | 400 / 400 pairs | Four workspaces each contain 100 JPGs with a same-stem JSON. Raw upstream image/source dataset is not present at `data/vitextvqa/`. |
| Initial localization matches | 1,153 | Auto-annotator match count; not answer review. |
| Unmatched localization records | 69 | 1,222 minus 1,153; also workspace manifest total. Not missing QA records. |
| Grouped workspace quest indices | 1,213 | Per-image grouped indices; not canonical question records. Nine-count difference has not been reconciled record by record. |
| Merged box availability | Metadata says 1,222; `still_missing_quests=0` | Merge checks box presence; does not prove answer-text review. Post-merge distinct localization-box count is not established by this metadata. |
| Original `gt_answers` availability | 1,222 nonempty rows in current canonical snapshot | Source provenance is not independently verifiable because raw source QA file is absent. |
| Explicit `verified_all_answers` | 79 nonempty rows | Merge-added answer sets; not known to be all, or the only, reviewed rows. |
| Reviewed references | Unknown | No complete per-reference review record found. |
| Adjudicated references | Unknown/not established | No traceable adjudication record found; do not equate this with a proven count of zero. |
| Rejected/disputed references | Unknown/not represented consistently | No complete status inventory found. |

### Proposed Minimal Contract

Keep records compatible with existing `images[].quests[]`, but separate answer-reference records from localization annotations. Empty reference lists and `not_reviewed` are valid; the schema must not fabricate adjudication.

```json
{
  "dataset_version": "vitextvqa-dev-400.<version>",
  "annotation_version": "<version>",
  "source_file_hashes": [{"path_id": "<logical-source-id>", "sha256": "<hash>"}],
  "images": [{
    "image_id": "<stable-source-id>",
    "image_name": "<basename>",
      "quests": [{
      "question_id": "<stable-source-id>",
      "question_text": "<text>",
      "original_references": [{"reference_id": "<id>", "text": "<text>", "source_record_id": "<id>"}],
      "additional_references": [{"reference_id": "<id>", "text": "<text>", "source_type": "annotator-added", "provenance_id": "<id>"}],
      "reviewed_references": [],
      "adjudicated_references": [],
      "rejected_references": [],
      "disputed_references": [],
      "review_status": "not_reviewed",
      "review_provenance": [],
      "localization_annotations": []
    }]
  }]
}
```

The contract preserves the current `images[].quests[]` key used by the canonical annotation JSON, workspace merger, and evaluator (`evaluate_vitextvqa.py` iterates `img.get("quests", [])`). The nested proposed field names add reference provenance but do not rename the question collection. If an internal adapter uses `questions`, it must explicitly map `images[].quests[]` to the adapter field at a named boundary, preserve stable IDs and round-trip data, and never write back a silently renamed source schema. No migration to `images[].questions[]` is proposed.

Reference entries should include stable IDs, exact original text, source/source-record, provenance, status (`proposed`, `accepted`, `rejected`, `disputed` where relevant), and reviewer/action/time for review events. Localization entries should record boxes, coordinate convention, annotator/detector source, and localization-review provenance separately. Legacy `verified_all_answers` may be emitted only as a documented derived view; it is not the authority for adjudication.

### Reference Selection Contract

**Recommended default, pending Owner confirmation:** Strict primary comparison uses a named snapshot of `original_references` as present in the frozen master, labeled `original-as-present-in-master`; do not call them official or independently verified until source QA provenance is confirmed. Do not silently add workspace text to this set. Added/reviewed/adjudicated references are evaluated in separately named reference tracks. For an adjudicated track, include only explicitly accepted references with provenance; disputed/rejected entries are excluded and surfaced. If the selected set is absent/empty, mark that sample `reference_status=unavailable` and the score unavailable; do not fall back silently to another set. The current implementation’s fallback should be represented explicitly if retained for backward-compatibility rescoring.

Every run records `reference_set_id`, `reference_set_version`, `reference_set_hash`, per-record selected reference IDs, and missing/disputed counts. Since the raw QA source is absent, source evidence or manual review is required before claiming that current `gt_answers` are original-source references.

### Slash Parsing Contract

**Recommended:** Never infer multiple-answer structure by splitting unescaped `/` in free text. Preserve `raw_description` exactly. Prefer a structured list of answer entries in annotation data. If legacy descriptions must be interpreted, split only when a separately documented, unambiguous separator field/escape convention says the slash is a delimiter; otherwise preserve the complete value and flag it for review. Do not define a new escape grammar without validating real annotator files.

Required fixtures: `20/11/2024`, `15.000đ/kg`, `https://host/path`, `A/B` literal, Vietnamese answer with punctuation, and structured `[`"bếp gas"`, `"bếp từ"`]` as the unambiguous multiple-answer representation.

### R-02 Acceptance-Test Matrix

| Test ID / Ticket | Objective | Preconditions / Input fixture | Expected behavior | Failure condition | Level | Dependencies |
|---|---|---|---|---|---|---|
| R02-T01 / R-02 | Preserve original references | Master row with `original_references=[r1]`, then add `additional_references=[r2]`. | `r1` remains byte/text-identical; `r2` is separate and carries provenance; original list is not overwritten. | Original is mutated, concatenated without lineage, or removed. | Unit | Annotation contract |
| R02-T02 / R-02 | Do not infer adjudication from bbox | Row has a matched localization box and no answer review event. | Localization status can be reviewed/available while answer `review_status=not_reviewed`; adjudicated list stays empty. | Box availability marks answer adjudicated/verified. | Unit | Annotation/reference separation |
| R02-T03 / R-02 | Explicit reviewed/adjudicated provenance | Accepted reference without reviewer/action/source provenance. | Validation rejects adjudicated status or flags it invalid; accepted record requires provenance fields. | Provenance-free reference becomes adjudicated. | Contract | Provenance schema |
| R02-T04 / R-02 | Absent source QA evidence | Current master exists, raw source dataset absent. | Dataset record/manifest reports `source_status=missing_or_unverified`; no “official original” claim is generated. | Missing source is inferred or fabricated. | Integration/contract | Source manifest |
| R02-T05 / R-02 | Preserve slash literal | Raw answer values `20/11/2024`, `15.000đ/kg`, `https://host/path`. | Each remains exactly one reference when no structured separator marker exists. | Any literal is split or rewritten. | Unit | Parsing contract/source format evidence |
| R02-T06 / R-02 | Parse structured multiple answers | Structured annotation contains two answer objects for “bếp gas” and “bếp từ”. | Two separate references retain IDs/provenance; no text heuristic is needed. | They are merged or provenance is lost. | Unit | Structured reference schema |
| R02-T07 / R-02 | Count-unit report | Fixture has 5 question rows, 4 distinct grouped workspace indices, 3 initial localizer matches, and 2 merged boxes. | Report emits each as distinct named fields; does not assert 100% match from box availability. | Any count overwrites or is mislabeled as another unit. | Contract | Count definitions |
| R02-T08 / R-02 | Do not erase unexplained nine-row difference | Current counts: master `1222`, grouped workspace `1213`. | Reconciliation report preserves both counts and a `difference=9; status=unresolved` entry with no silent dropping/duplication. | Count is forced to 1,222 by duplication or truncation. | Integration | Source IDs/group map |
| R02-T09 / R-02 | Deterministic reference selection | Row has original, added, accepted adjudicated, disputed and rejected references. | Given `reference_set_id`, evaluator selects exactly configured reference IDs and records the set ID/hash; disputed/rejected do not enter accepted adjudicated set. | Selection changes with record order or hidden fallback. | Contract/integration | D-01 protocol ID; Owner approves primary set |
| R02-T10 / R-02 | No silent fallback | Selected adjudicated set is empty while original set exists. | Adjudicated score is `unavailable`; separate original score may be emitted under its own reference-set ID. | Adjudicated score is silently computed from original references. | Integration | Reference selection contract |
| R02-T11 / R-02 | Preserve development classification | Manifest source is the existing 400-image development collection. | Run metadata and report say development/screening; not independent test. | It is labeled independent/confirmatory. | Contract | Research Plan v2 |

## R-03: Canonical Contrastive Decoding

### Formula and Source Flow

`OWNER-APPROVED` canonical formula:

$$z^*=(1+\alpha_t)z_{pos}-\alpha_tz_{neg}$$

ReLU-clipped historical variant:

$$z^{relu}=z_{pos}+\alpha_t\operatorname{ReLU}(z_{pos}-z_{neg})$$

The current code computes `delta_z = relu(pos_logits - neg_logits)` and then `calibrated_logits = (pos_logits + alpha_t * delta_z) / t_entropy` in [dynamic_cd.py](../../scripts/benchmark/sg_vcd/dynamic_cd.py#L212) and [same file](../../scripts/benchmark/sg_vcd/dynamic_cd.py#L275). This is the ReLU variant, not the approved canonical formula.

In `_decode_spatial_pmi_contrastive`, the positive view uses the global thumbnail plus edge-enhanced sharp ROI tiles; the negative view uses the same global thumbnail plus low-pass/blurred ROI tiles ([pipeline.py](../../scripts/benchmark/sg_vcd/pipeline.py#L389)). It builds same-question suffixes for both branches and extracts logits as float tensors ([same file](../../scripts/benchmark/sg_vcd/pipeline.py#L495)). The selected token is embedded once and passed to both positive and negative KV-cache branches on subsequent steps ([same file](../../scripts/benchmark/sg_vcd/pipeline.py#L586)). Thus both branches are intended to share vocabulary and generated prefix at each step.

### Required Calibration Order

1. Obtain `z_pos` and `z_neg` for the same batch, decode step, vocabulary, and generated prefix. Require identical tensor shapes; do not rely on accidental broadcasting.
2. Calculate canonical signed logits elementwise: `(1 + alpha_t) * z_pos - alpha_t * z_neg`.
3. Apply the existing positive entropy temperature division `T(H)` after formula calculation. Preserve its existing floor and calculation unless separately approved.
4. Preserve current EOS adjustment after temperature; it is currently conditional on EOS already passing the candidate-validity mask and the configured EOS boost being nonzero.
5. Preserve candidate masking and final argmax after those operations. The existing candidate-support signal uses `relu(z_pos-z_neg)` to define positive-contrast plausibility. Keep that mask signal distinct from the **canonical score formula**; changing it is outside this formula correction absent separate approval.
6. Preserve alpha scheduling: current calculation uses spatial confidence, time decay, KL/JS gating, entropy gating, QIACF/AHRA factors, phase cooldown, and clips alpha to `[0,2]` ([dynamic_cd.py](../../scripts/benchmark/sg_vcd/dynamic_cd.py#L225)). Do not change those parameters in R-03.

All statements about the behavior of post-operations are source observations. No decoding execution was performed.

### Formula Fixtures

For `alpha=0.5`, `z_pos=[4,2]`, `z_neg=[1,3]`:

- Canonical signed result: `[5.5, 1.5]`.
- ReLU variant result: `[5.5, 2.0]`.

For `alpha=0.5`, scalar `z_pos=2`, `z_neg=3`:

- Canonical signed result: `1.5`.
- ReLU variant result: `2.0` because the negative difference is clipped to zero.

For equal vectors `z_pos=[2,-1]`, `z_neg=[2,-1]`, both return `[2,-1]`. For `alpha=0`, both reduce to `z_pos`.

### Variant Identity Proposal

`RECOMMENDED` immutable identifiers:

- Canonical future method: `sg_vcd.signed.v1`.
- Historical ReLU-clipped implementation: `sg_vcd.relu_clipped.v1`.
- Existing artifacts without commit/config evidence identifying their formula: `sg_vcd.variant_unknown` until source/run provenance resolves them. Do not retroactively relabel them based on current code alone.

Store `method_variant_id` in config, logs, checkpoints, per-sample predictions, and reports. The method ID is separate from metric protocol and reference-set IDs. Keep historical ReLU results and never pool them with canonical signed results in an unlabeled table.

### R-03 Acceptance-Test Matrix

| Test ID / Ticket | Objective | Preconditions / Input fixture | Expected behavior | Failure condition | Level | Dependencies |
|---|---|---|---|---|---|---|
| R03-T01 / R-03 | Positive difference exact vector | `alpha=.5`, `pos=[4,2]`, `neg=[1,3]`. | Canonical intermediate logits exactly `[5.5,1.5]`; ReLU variant `[5.5,2.0]`. | Canonical outputs ReLU result or variant labels are absent. | Unit | Formula IDs |
| R03-T02 / R-03 | Negative difference distinguishes methods | `alpha=.5`, `pos=[2]`, `neg=[3]`. | Canonical `1.5`; ReLU variant `2.0`. | Canonical clamps negative difference or values differ. | Unit | Formula IDs |
| R03-T03 / R-03 | Equal logits | `alpha=.5`, `pos=[2,-1]`, `neg=[2,-1]`. | Both formula outputs `[2,-1]`. | Nonzero contrast appears. | Unit | Formula IDs |
| R03-T04 / R-03 | Zero alpha | Any finite same-shaped vectors; `alpha=0`. | Both outputs equal `pos` exactly. | Negative logits influence output. | Unit | Formula stage |
| R03-T05 / R-03 | Shape/vocabulary alignment | Equal vectors then mismatched shapes, e.g. `[1,2]` vs `[1,2,3]`. | Equal shape is processed elementwise; mismatched shape is rejected with a clear input-contract error; no silent broadcasting. | Shape mismatch silently broadcasts or produces misaligned logits. | Unit | Branch contract |
| R03-T06 / R-03 | Finiteness and dtype | Finite float32 logits and bounded finite alpha. | Formula intermediate has same shape and finite values; implementation preserves expected float precision. | NaN/Inf or unintended dtype/device changes. | Unit | Numerical contract |
| R03-T07 / R-03 | Temperature order | Controlled alpha, logits and fixed positive `T`; no EOS boost/mask interference. | Pre-mask score equals `formula_logits / T` elementwise. | Temperature is applied to only one branch or before formula contrary to spec. | Unit | Approved existing `T(H)` |
| R03-T08 / R-03 | EOS and mask order preserved | Controlled EOS logit, candidate mask, `kappa_eos=0` and then enabled fixture where EOS is valid. | With boost disabled, no EOS addition; with boost enabled, addition follows temperature and only applies if valid mask allows EOS; final invalid candidates are `-inf`. | EOS bypasses mask or ordering changes. | Unit | Existing EOS/mask policy |
| R03-T09 / R-03 | Positive-contrast mask remains distinct | Token has `z_pos<z_neg` but is otherwise tested under APC support; inspect score formula and validity mask separately. | Canonical score is signed; existing positive-contrast support remains based on the separately specified ReLU delta unless approved otherwise. | Score formula change silently changes candidate mask semantics. | Unit | R-03 order spec |
| R03-T10 / R-03 | Canonical signed calibration before mask | Call the canonical calibration stage with `alpha=.5`, `pos=[10,0]`, `neg=[0,1]`, `T=1`, and no EOS adjustment. | Pre-mask canonical score is exactly `[15,-0.5]`; the second token remains a negative signed score at this stage. | Formula uses ReLU or the negative component is lost before mask evaluation. | Unit | Canonical formula; expose/test pre-mask stage |
| R03-T11 / R-03 | Candidate support independently | For the R03-T10 fixture, use current APC settings `beta_0=.035`, `theta_c=.35`; assert `apc_mask`, `contrast_mask`, and `valid_mask` separately. | `delta_z=[10,0]`; the negative-difference token has no positive-contrast support and is below APC probability threshold; expected masks are `apc=[true,false]`, `contrast=[true,false]`, `valid=[true,false]`. | Mask is inferred from signed score only, or mask results are not observable separately. | Unit | Existing candidate-support policy |
| R03-T12 / R-03 | Negative-difference mask impact observable | Apply R03-T11 mask to the R03-T10 pre-mask values. | The second token's pre-mask score is `-0.5`; its final masked score is `-inf`; the record/test reports both the pre-mask score and mask reason. | Test reports only final `-inf`, making score formula indistinguishable from clamping, or the mask is silently changed. | Unit | R03-T10/T11; mask diagnostics |
| R03-T13 / R-03 | Formula and support remain independent | Compare canonical signed score and existing ReLU candidate-support signal for the same `z_pos<z_neg` token. | Canonical score remains signed; candidate-support remains the existing positive-difference ReLU rule; changing either contract requires its own approved change. | Candidate-support ReLU is removed/changed as an unreviewed side effect of formula correction. | Unit/contract | D-03; candidate-support policy |
| R03-T14 / R-03 | Same per-step prefix | Mock branch models record initial suffix and each generated token supplied to both KV branches. | Same question suffix and same previously selected token IDs are observed at each step; positive/negative logits have same vocab size. | Branch prefixes diverge or tokens are independently sampled. | Mock integration | Pipeline branch contract |
| R03-T15 / R-03 | Variant provenance | Run config and result artifacts for each variant. | Every artifact contains `method_variant_id`; signed and ReLU records aggregate separately. | Missing ID or cross-variant pooled metric. | Contract/integration | R-04 result schema |

## R-04: Failure and Spatial-Metric Semantics

### Proposed Result Record

The record is a design contract, not executable code. Every expected question from the frozen manifest must have a result row, including missing images, skipped work, and failures.

```json
{
  "run_id": "<id>",
  "sample_key": "<stable-question-image-key>",
  "image_id": "<id>",
  "image_name": "<basename>",
  "question_id": "<id>",
  "method_id": "<method>",
  "method_variant_id": "sg_vcd.signed.v1",
  "metric_protocol_id": "vitextvqa.strict.v1",
  "reference_set_id": "<frozen-reference-set>",
  "run_status": "completed",
  "answer": {"status": "valid", "text": "<answer>"},
  "localization": {
    "status": "valid",
    "applicability": "applicable",
    "roi_source": "predicted",
    "coordinate_space": "original_image_xyxy",
    "boxes": [{"box_id": "roi-1", "bbox": [0, 0, 1, 1], "confidence": null}],
    "box_metadata_status": "available",
    "aggregation": {"status": "computed", "policy_id": "<approved-policy>", "reason": null}
  },
  "failure": null,
  "metrics": {
    "answer": {"token_f1": {"status": "computed", "value": 0.0}},
    "spatial": {"iou": {"status": "computed", "value": 0.0}}
  },
  "attempts": []
}
```

`run_status` describes completion of the configured stages; answer and localization remain independent dimensions. Required deterministic states:

| Dimension | Allowed values | Meaning |
|---|---|---|
| `run_status` | `completed`, `partial`, `failed`, `missing_image`, `skipped` | `completed`: all configured stages returned without execution errors; answer may still be invalid. `partial`: at least one configured stage returned a result and another failed. `failed`: execution failed before any configured stage returned a result usable by the contract. `missing_image`/`skipped`: image unavailable or row intentionally not attempted. |
| `answer.status` | `valid`, `invalid`, `not_attempted` | `not_attempted`: no answer output was returned. `invalid`: an output was returned but failed the approved validator. `valid`: returned output passed that validator. Validator rules (including empty text) must be approved before implementation. |
| `localization.status` | `valid`, `failed`, `not_produced`, `not_applicable` | Valid ROI output; configured localizer error; expected localizer returned no valid boxes without exception; or localization is not part of this method/condition, respectively. |
| `localization.applicability` | `applicable`, `not_applicable` | Whether the method/condition is expected to output localization. |
| `localization.roi_source` | `predicted`, `oracle_gt`, `none` | Oracle must be a separate diagnostic condition; never mix into predicted-ROI headline results. |
| Per-metric `status` | `computed`, `not_applicable`, `unavailable`, `pending_policy` | `computed` includes genuine zero values; N/A has no value; unavailable means required input/runtime prevented computation; pending means the governing inclusion or aggregation policy is not approved. |
| `localization.box_metadata_status` | `available`, `unavailable`, `not_applicable` | Availability of individual ROI data/coordinate metadata, independent of aggregate scoring. |
| `localization.aggregation.status` | `computed`, `pending_policy`, `unavailable`, `not_applicable` | Approved aggregate; aggregation policy not approved; aggregation failed/could not compute; or localization is not applicable. |

`failure` should record stage, stable error category, sanitized message, and retryability; it must not store secrets or credentials. Keep a per-attempt history so resume can distinguish a prior failure from later success. Retries use stable `sample_key` plus attempt ID; summaries count the manifest row once and retain attempt history. For each metric, retain R-01's expected, applicable, computed, N/A, unavailable, pending-policy counts and computed-value denominator.

**Deterministic state transitions (`RECOMMENDED`):**

1. Enumerate expected rows from the manifest before processing.
2. If the image cannot be found/opened, set `run_status=missing_image`, `answer.status=not_attempted`; localization is `not_applicable` for a non-localizing method and otherwise `not_produced`; spatial metric is respectively `not_applicable` or `unavailable`. Persist a row.
3. If a row is explicitly skipped, set `run_status=skipped`, `answer.status=not_attempted`, persist the reason and row; answer metric is `pending_policy` until denominator treatment is approved.
4. If the entire execution fails before any configured stage returns an output, set `run_status=failed`, `answer.status=not_attempted`, and localization `failed` if the localizer was invoked or `not_produced` if it was not. Persist failure; never assign a synthetic/full-image box.
5. If Stage 1 fails but Stage 2 returns a validator-approved answer, set `run_status=partial`, localization `failed`, answer `valid`; answer metrics can be `computed` if references exist, while spatial metrics are `unavailable`.
6. If Stage 1 succeeds but Stage 2 fails before returning any answer output, set `run_status=partial`, localization `valid`, answer `not_attempted`. Spatial metrics can be computed if the approved ROI policy allows; answer metric status is `pending_policy` pending denominator approval.
7. If answer generation returns text, apply the approved validator: pass => `valid`; fail => `invalid`. Do not label returned-but-invalid text as `not_attempted`. The validator's exact rules remain an implementation prerequisite, not a new scientific metric definition.
8. A completed answer-only control has `run_status=completed`, localization `not_applicable`, and spatial metrics `not_applicable`.
9. A multi-ROI localizer can have `localization=valid`, all boxes and per-box metadata available, while aggregate spatial metrics are `pending_policy` until matching policy approval.
10. Only valid localization outputs under an approved policy receive numerical spatial scores. A full-image box is valid only if actually produced by the declared method, never as a failure/default sentinel.

### Answer Denominator Options

| Option | Definition | Benefit | Risk |
|---|---|---|---|
| A: Conditional quality | Average answer metrics over successful, valid predictions only; report failure/completion rates separately. | Measures answer quality conditional on producing an answer. | Can look better if difficult samples fail; must show denominator and failure rate. |
| B: Fixed-manifest, failure-zero | Use all expected manifest rows as denominator; failed/missing/skipped predictions contribute zero under a predeclared rule. | Reflects end-to-end completion and prevents silent omission. | Conflates system reliability with conditional answer quality; missing input may be outside method control. Requires a precise policy for invalid/missing/skipped cases. |
| C: Dual reporting | Report conditional quality and completion/failure counts; optionally report a separately named fixed-manifest score if its treatment of each failure class is approved. | Separates answer correctness from operational completion and avoids hiding failures. | More than one result view; users may overemphasize one. No composite/product is defined here. |

**Recommendation:** Option C. Keep a fixed expected-row denominator and publish conditional answer quality plus success/failure/missing/skipped counts. Whether to additionally publish a failure-zero score, and the treatment of `skipped` versus `missing_image`, is `PENDING-OWNER-APPROVAL` and `PENDING-CHATGPT-REVIEW`. Do not silently drop failures.

### Spatial Applicability and Multi-ROI Options

`compute_stage1_metrics` currently takes one predicted box and one primary GT box, with optional sub-boxes ([metrics.py](../../scripts/benchmark/metrics.py#L308)). It calculates sub-box area coverage but also an “any sub-box recall >= 0.5” hit; `multi_box_text_coverage_pct` aggregates this hit, not the area coverage ([same file](../../scripts/benchmark/metrics.py#L334), [same file](../../scripts/benchmark/metrics.py#L429)). The pipeline exposes `stage1_bboxes`, while the evaluator extracts `stage1_bbox` only ([pipeline.py](../../scripts/benchmark/sg_vcd/pipeline.py#L926), [evaluate_vitextvqa.py](../../scripts/benchmark/evaluate_vitextvqa.py#L1112)). No multi-ROI aggregation formula is selected here.

| Possible policy | Mathematical interpretation | Multi-GT / duplicate behavior | Compatibility and risk |
|---|---|---|---|
| A: Any predicted ROI per target | For each GT region, evaluate the best per-ROI overlap/recall, then threshold under a preregistered rule. | Can show whether any ROI finds each target; redundant boxes are not penalized unless separately reported. | Close to hit-rate semantics; can reward submitting many boxes. Threshold/aggregation still needs approval. |
| B: Geometric union | Form the geometric union of all predicted boxes and compare with GT box/union using existing IoU, inclusion recall, and precision definitions. | Overlapping duplicates do not add union area; extra boxes can lower precision/IoU. GT union must be explicitly defined. | Uses existing primitives but can conceal which ROI matched which text and may reward broad unions on recall. |
| C: One-to-one assignment | Assign predicted boxes to GT boxes under a declared matching objective/threshold; report matched and unmatched boxes. | Duplicates cannot match the same GT twice; unmatched predictions/GTs remain visible. | Better exposes multiplicity but requires a matching algorithm, objective, threshold, and aggregation choice not in the current evaluator. |

**Pending:** ChatGPT and Project Owner must choose the ROI-to-GT matching policy, thresholds, and aggregation weights. Until then, store every predicted box with stable `box_id`, coordinates, source, ordering, and available per-box metadata; set `localization.status=valid`, `box_metadata_status=available`, `localization.aggregation.status=pending_policy`, and each affected spatial aggregate metric to `pending_policy`. Do not calculate a multi-ROI aggregate or silently use only the first box. If aggregation code is attempted under an approved policy but raises or lacks required data, retain the boxes and set aggregation/metric state to `unavailable`, with a reason. `not_applicable` is reserved for a method/condition with no localization task; it is not an alias for pending or unavailable.

Condition rules:

- Predicted-ROI method with valid boxes: localization is `valid`; retain all boxes and metadata. Spatial metrics are `computed` only under an approved aggregation policy; otherwise `pending_policy`.
- Method without localization: localization, box metadata, aggregation, and spatial metrics are `not_applicable`; do not assign a full-image box as a substitute.
- Expected localizer returns an empty set without exception: localization `not_produced`; empty-list metadata may be `available`; required spatial metrics are `unavailable` because no prediction exists.
- Localizer execution fails: localization `failed`; spatial metrics are `unavailable`.
- Multi-ROI aggregation itself fails at runtime: localization remains `valid`, individual box metadata remains `available`, aggregation and affected metrics are `unavailable`.
- Image missing or localization stage not attempted: localization is `not_produced` for a localization-eligible method; spatial metrics are `unavailable`. For a non-localizing method they remain `not_applicable`.
- Oracle ROI: separate `condition_id`/`roi_source=oracle_gt` diagnostic result; exclude from predicted-ROI/end-to-end headline rows. Its localization boxes may be valid oracle inputs, but they are not predicted localization outputs.

### R-04 Acceptance-Test Matrix

| Test ID / Ticket | Objective | Preconditions / Input fixture | Expected behavior | Failure condition | Level | Dependencies |
|---|---|---|---|---|---|---|
| R04-T01 / R-04 | Successful answer and localization | Manifest row; all configured stages complete; valid answer; one valid predicted ROI under an approved single-ROI policy. | `run_status=completed`; answer `valid`; localization `valid`; box metadata `available`; aggregation `computed`; answer and spatial metrics `computed`. | Any state differs or a computed zero is encoded as a non-computed state. | Integration | R-01–R-03 IDs; single-ROI policy |
| R04-T02 / R-04 | Non-localizing control | Baseline completes and returns a valid answer, no ROI by design. | `run_status=completed`; answer `valid`; localization `not_applicable`; box metadata and spatial aggregation `not_applicable`; answer metrics computed. | Spatial numeric score is emitted. | Integration | Method condition registry |
| R04-T03 / R-04 | Entire model execution fails before any output | Stub raises before any configured model stage is invoked and returns no output. | `run_status=failed`; answer `not_attempted`; localization `not_produced`; answer metric `pending_policy`; spatial metric `unavailable`; sanitized failure present; no box. | Any other status or numeric full-image score. | Integration | Result schema; denominator decision |
| R04-T04 / R-04 | Missing image | Expected manifest row points to absent/unreadable image; method normally localizes and answers. | `run_status=missing_image`; answer `not_attempted`; localization `not_produced`; box metadata `unavailable`; answer metric `pending_policy`; spatial metric `unavailable`; row persists. | Row is omitted or any output is reported as successful. | Integration | Manifest/result writer |
| R04-T05 / R-04 | Stage 1 fails, Stage 2 returns valid answer | Localizer throws; Stage 2 nevertheless returns validator-approved text; references exist. | `run_status=partial`; answer `valid`; localization `failed`; answer metrics `computed`; spatial metrics `unavailable`; no full-image fallback. | Run marked completed/failed, or answer discarded solely because localization failed. | Integration | Status semantics; answer validator |
| R04-T06 / R-04 | Localizer returns no box without exception | Stage 1 returns empty box set; Stage 2 returns valid answer. | `run_status=completed`; answer `valid`; localization `not_produced`; box metadata `available` with an empty list; spatial metrics `unavailable`; answer metrics `computed`. | Empty set converted to full image, or confused with `not_applicable`. | Unit/integration | Localizer output contract |
| R04-T07 / R-04 | Preserve all ROIs pending policy | Pipeline returns two valid distinct boxes; no multi-ROI policy approved. | `run_status=completed`; localization `valid`; both boxes and metadata retained; `box_metadata_status=available`; aggregation and spatial metric states `pending_policy`; no aggregate number. | First box only is retained or a formula is assumed. | Contract/integration | R-03 output contract; Owner/ChatGPT ROI policy |
| R04-T08 / R-04 | Oracle isolation | Separate oracle diagnostic and predicted-ROI conditions; fixture has one valid oracle box and an approved diagnostic scoring policy. | Oracle row has `run_status=completed`, localization `valid`, box metadata `available`, `roi_source=oracle_gt`, and its own condition ID; its spatial metric is `computed` only in the oracle diagnostic summary. | Oracle ROI enters predicted-ROI aggregate or receives an ambiguous status. | Integration | No-GT-leakage contract; diagnostic policy |
| R04-T09 / R-04 | Failed localization cannot become full-image success | Stage 1 throws after image dimensions are known; no Stage 2 answer is returned. | `run_status=failed`; answer `not_attempted`; localization `failed`; spatial metric `unavailable`; no `[0,0,W,H]` box or spatial score. | Any failure gets full-image recall/coverage. | Regression | R04-T03 |
| R04-T10 / R-04 | Resume after partial failure | Two expected rows: first completed; second failed before output. Retry the second and return valid answer and localization. | Final manifest rows: first `run_status=completed`; second final `run_status=completed`, answer `valid`, localization `valid`; second attempt history retains initial failed attempt; exactly two final rows. | Duplicate final rows, lost attempt history, or failed attempt treated as completion. | Integration | Checkpoint/provenance schema |
| R04-T11 / R-04 | Run counters | Manifest has one each of completed, partial, failed, missing-image, and skipped rows. | `expected=5`; each run-status count is exactly `1`; five rows persist and counts sum to `5`. | Any category is omitted, double counted, or silently discarded. | Contract | Run-status definitions |
| R04-T12 / R-04 | N/A versus genuine zero | One control has no localization; one applicable valid predicted ROI has zero overlap. | Control spatial result is `not_applicable` with no numeric value; predicted ROI spatial metric is `computed` with `value=0.0`. | N/A is encoded as zero or zero as N/A. | Unit/contract | Metric applicability |
| R04-T13 / R-04 | Denominator status is explicit | Fixture includes one valid answer, one failed/no-answer row, one missing reference, one N/A condition, and one pending failure-policy row. | Per-metric states are respectively `computed`, `pending_policy`, `unavailable`, `not_applicable`, `pending_policy`; summary reports all counts and denominator=`computed_count`. | Any non-computed state becomes zero or disappears. | Contract | R-01 metric envelope; pending denominator decision |
| R04-T14 / R-04 | Full-image baseline regression | Baseline completes with valid answer and no localization; GT is a small region. | `run_status=completed`; answer `valid`; localization and spatial aggregation `not_applicable`; no perfect spatial recall is emitted. | Full image is scored as predicted ROI. | Integration | Method registry |
| R04-T15 / R-04 | Stage 2 fails after successful localization | Stage 1 returns one valid box under approved single-ROI policy; Stage 2 throws before returning any answer. | `run_status=partial`; answer `not_attempted`; localization `valid`; box metadata `available`; spatial metric `computed`; answer metric `pending_policy`; failure identifies Stage 2. | Run is marked completed, localization erased, or answer failure silently scored as zero. | Integration | Failure/denominator policy |
| R04-T16 / R-04 | Returned answer fails validation | Stage 1 and Stage 2 execute without error; Stage 1 returns one valid ROI; Stage 2 returns a string rejected by a fixed test validator. | `run_status=completed`; answer `invalid`; localization `valid`; box metadata `available`; single-ROI spatial metric `computed`; answer metrics `unavailable`. | Returned invalid text is marked `not_attempted` or any status differs. | Contract/integration | Approved answer-output validator; single-ROI policy |
| R04-T17 / R-04 | Multi-ROI metadata versus aggregate runtime failure | Localizer returns two boxes; individual metadata serializes; aggregation procedure throws. | `run_status=completed`; localization `valid`; both boxes retained; box metadata `available`; aggregation and spatial metrics `unavailable` with aggregation error reason. | Error changes localization to `not_applicable`, drops boxes, or emits a score. | Integration | ROI policy implementation |
| R04-T18 / R-04 | Single ROI but aggregate policy pending | Localizer returns one valid box, but no policy ID has been configured for spatial evaluation. | `run_status=completed`; localization `valid`; box metadata `available`; aggregation and spatial metric `pending_policy`. | An implicit metric policy is used. | Contract | Policy registry |
| R04-T19 / R-04 | No-localization condition state | Non-localizing method completes successfully. | `run_status=completed`; localization `not_applicable`; box metadata and aggregation `not_applicable`; spatial metrics have no numeric values. | State reported as `unavailable`, `failed`, or numeric zero. | Unit/contract | Method condition registry |

## Cross-Ticket Integration

| Contract edge | Requirement |
|---|---|
| R-01 applicability and denominator → R-04 aggregation | R-01 defines metric states and counts; R-04 applies them per expected manifest row and reports the computed-value denominator plus failure/run counters. Failed rows remain visible; fixed-manifest answer scoring stays pending until approved. |
| R-02 reference selection → R-01 metric input | R-02 supplies one `reference_set_id`/version/hash and selected reference IDs; R-01 Strict best-reference behavior operates only over that set. No implicit cross-set fallback. |
| R-03 method variant → R-04 result | R-03 supplies `method_variant_id` and configuration; R-04 stores it with the result. Signed and ReLU records never pool under one variant. |
| R-01 protocol + R-02 reference set + R-03 method variant → R-04 result | Each result stores the existing `metric_protocol_id`, `reference_set_id`, and `method_variant_id` once, with `run_id` and stable `sample_key`; reports group by those identities rather than duplicating their definitions. |
| R-04 statuses → report aggregation | Each expected row has deterministic `run_status`, answer status, localization status, and per-metric state. Metric counts reconcile to expected rows; only computed values form numeric means. Spatial aggregation status distinguishes computed, pending-policy, unavailable, and not-applicable cases. |

**Potential circular dependency:** R-01 score artifacts require a reference selection contract, while R-02 wants to know the scoring protocol that consumes references. Resolve by defining independent immutable IDs first: `metric_protocol_id` (R-01), `reference_set_id` (R-02), and `method_variant_id` (R-03). R-02 supplies selected reference IDs; R-01 scores only those IDs; R-04 records the three IDs once per result/run scope. None encodes another. `run_id` and `sample_key` identify execution and row, not another protocol.

## Open Decision Register

| ID | Question | Current evidence / options | Recommendation | Required reviewer |
|---|---|---|---|---|
| OD-01 | What Strict reference set is primary: original-as-present, reviewed, or adjudicated? | Source QA is absent; current loader falls back between fields; only 79 rows have explicit additions. | Use a named original-as-present snapshot only as provisional Strict input; separately report adjudicated scores when records exist. Confirm before baseline runs. | Project Owner; source evidence/annotation review |
| OD-02 | Which incomplete Round 7 rules were used for historic results? | Report has ANLS/F1 descriptions; span generation, exact normalization, reference aggregation, and Substring Match are incomplete; supporting test imports missing functions. | Do not claim exact R7 reproduction. Recover source/config/evaluator artifacts or keep results `Round7-reported-unversioned`. | Project Owner, ChatGPT, additional run source |
| OD-03 | Is Round 7 concise bound truly `OR` or a hard 25-token cap? | Formula and prose conflict. | No implementation choice until corrected/approved. | ChatGPT, Project Owner |
| OD-04 | What denominator is primary for answer quality when failures occur? | Conditional, fixed-manifest zero, or dual reporting have different interpretations. | Recommend dual reporting; Owner/ChatGPT decide whether a failure-zero score is headline. | Project Owner, ChatGPT |
| OD-05 | How should multiple ROIs match multiple GT sub-boxes? | Any-hit, geometric union, and one-to-one assignment are viable but differ materially. | Preserve all boxes; do not publish a single multi-ROI aggregate until policy is preregistered. | Project Owner, ChatGPT |
| OD-06 | Are original references traceable to official source records? | Raw `ViTextVQA_dev.json` absent; generator can fall back to tracked predictions. | Obtain the exact source file/version/hash or label source provenance incomplete. | Project Owner, data custodian |
| OD-07 | What explains 1,222 master rows vs 1,213 grouped workspace indices? | Grouping by per-image `quest_idx`; nine-row difference not itemized. | Produce an ID-level reconciliation without dropping/duplicating rows. | Project Owner, annotation owners |
| OD-08 | Do existing results identify the actual logit formula? | Checkpoint lacks source commit/config; current code is ReLU, canonical docs are signed. | Preserve `variant_unknown` unless run-specific source evidence resolves it; do not retro-label. | Project Owner, experiment custodian |
| OD-09 | How are `valid` and `invalid` answer outputs determined? | Current code treats exceptions as empty strings; output-validation policy is not explicit. | Define parser/output contract and test empty/unparseable responses separately. | Project Owner, technical reviewer |
| OD-10 | Are existing skills/rules updated for Owner approval? | Rules still mention plan awaiting researcher approval, superseded by explicit Owner decision. | Record as stale; modify only under a separately authorized governance task. | Project Owner |

## Implementation Sequence Recommendation

This sequence is dependency-aware; it does not assume pending decisions are approved.

1. **Approve stable identities and contracts:** Strict metric ID, reference-set ID, method-variant IDs, result status dimensions. No source edits until their fields and compatibility requirements are reviewed.
2. **Reconcile annotation evidence (R-02):** recover/source the original QA dataset or explicitly record its absence; reconcile question/group counts; document answer-review provenance and slash-notation evidence. Do not rewrite legacy source files as part of this design.
3. **Finalize evaluator specification (R-01):** preserve Strict behavior as `vitextvqa.strict.v1`; keep Round 7 as separate, incompletely specified historical protocol unless its missing rules are approved. Confirm primary reference set and version all score outputs.
4. **Implement canonical formula variant (R-03):** signed formula under canonical method identity; preserve the ReLU variant with its own ID if required for historical/configured use. Keep alpha schedule, candidate support, temperature, EOS, and masking behavior unchanged except for the approved score formula.
5. **Implement failure/result contract (R-04):** expected row per manifest question, orthogonal statuses, no fallback bbox, preserved ROI lists, protocol/reference/method IDs, attempt-aware resume, and explicit counters. Do not select unresolved answer denominators or ROI aggregation in code without approval.
6. **Run mock-only acceptance checks locally** only after approved implementation and with environment dependencies authorized/available. Real-model integration and larger comparisons remain Colab-only and require explicit experiment authorization after data/evaluator/config gates pass.

There is a dependency loop between references and metric artifacts; separate `reference_set_id` from `metric_protocol_id` as described above. Spatial aggregation and failure denominator are independent scientific/reporting decisions and must be approved before headline tables are implemented.

## Approval Checklist

- [ ] Project Owner confirms Strict primary reference-set selection (`original-as-present` versus an adjudicated set) and handling of missing/disputed references.
- [ ] Project Owner and ChatGPT approve the status/count definitions and the 1,222 vs 1,213 reconciliation procedure.
- [ ] ChatGPT and Project Owner confirm Round 7 gaps will remain historical/unversioned unless exact source behavior is recovered; approve any future full specification separately.
- [ ] ChatGPT reviews the signed canonical formula against the approved SG-VCD method; Project Owner approves the implementation/variant IDs.
- [ ] Project Owner and ChatGPT choose answer denominator/reporting and multi-ROI spatial policy before their respective aggregation code is implemented.
- [ ] Project Owner approves the result schema and failure/resume semantics.
- [ ] Project Owner separately authorizes any subsequent Colab real-model integration or baseline experiment; this document itself authorizes none.
- [ ] ChatGPT research review and Project Owner acceptance occur before Anti receives implementation work.

## Scope Confirmation

Only this specification document is created. No implementation code, annotation, metric, test, experiment output, Rule, Skill, environment, or notebook is changed. No test, benchmark, model inference, dependency installation, data/model download, Colab run, commit, or push is performed. This specification is pending review and is not an Anti implementation authorization.