# SG-VCD Research Review and Baseline Comparison Plan — Revised Draft

**Version:** 2.0  
**Status:** Revised research plan for researcher review; execution follows researcher approval  
**Target method:** SG-VCD (two-stage method)  
**Target backbone:** `5CD-AI/Vintern-1B-v3_5`  
**Primary task:** Vietnamese scene-text visual question answering (ViTextVQA)  
**Primary project documents:** `Proposed_Method_SG_VCD.md`, `Diacritic_Aware_Contrastive_Decoding_Overview.md`  
**New related proposal:** `Proposed_Method_CERV.md` (separate three-stage proposal; not merged into SG-VCD in this plan)  
**Dataset scope reported by the leader:** 400 images / 1,222 QA pairs  
**Purpose of this revision:** Convert the leader's feedback into a paper-method baseline comparison plan that respects the work already reported in SG-VCD Rounds 1–7.

> **Important distinction:** The leader's SG-VCD document reports completed experiments through Round 7 on 400 images / 1,222 QA pairs. This plan does not treat those reported runs as if they had not happened. The next experiment is a reproducible *comparative baseline run* against relevant paper methods, not another attempt to rebuild SG-VCD from zero. The numerical results below are recorded as results reported in the leader's document; they have not been independently reproduced from raw predictions and logs in this planning document.

---

## 1. Executive Decisions After Leader Feedback

The leader's four feedback points are adopted as binding design constraints:

1. SG-VCD has **two stages**. The research review must include methods comparable to **Stage 1 (spatial grounding / localization)** as well as methods comparable to **Stage 2 (contrastive or calibrated decoding)**. A list dominated by Stage 2 methods is incomplete.
2. Vanilla Vintern is **not a paper-method baseline**. It is a backbone control/reference condition used to measure the value of intervention.
3. Pilot conditions must be built from identifiable paper methods and must answer baseline-comparison questions. The previous five-condition proposal (Vanilla, Oracle ROI, low-pass ROI, full-image VCD, ROI contrastive core) mixed controls, diagnostics and component ablations; it did not constitute a complete paper-method baseline comparison. Those component diagnostics may be retained for a later ablation study, but they are not the main baseline pilot.
4. The three required answer metrics remain **Token-F1, ANLS and Substring Match**. No replacement metric may be silently substituted. Additional metrics are supplementary only and must not replace the three leader-requested metrics.

### 1.1 What has already been reported as completed

The latest `Proposed_Method_SG_VCD.md` describes the following development history on `5CD-AI/Vintern-1B-v3_5` and the 400-image / 1,222-QA ViTextVQA subset:

| Reported phase | Main reported work | Reported state |
|---|---|---|
| Initial baseline | Vanilla Vintern / pre-upgrade reference | Reported as the starting control |
| Round 1 | SG-VCD execution on Google Colab NVIDIA L4, single worker | Reported complete: 1,222 QA |
| Round 2 | Four-worker data sharding and atomic aggregation | Reported complete: 1,222 QA |
| Round 3 | Adaptive Top-K grounding, Laplacian edge re-emphasis, two-phase read-then-reason calibration | Reported complete: 1,222 QA |
| Round 4 | Wider spatial coverage, horizontal candidate fusion, Vietnamese diacritic-peak protection, soft cooldown | Reported complete: 1,222 QA |
| Round 5 | Dynamic multi-patch ROI tiling; CAPC, AHRA, QIACF, TESG and ESSC decoding components | Reported complete: 1,222 QA |
| Round 6 | EOS boosting and syntax/post-processing changes; regression observed | Reported as a degraded round, not a successful final configuration |
| Round 7 | Restoration of generation fidelity and Symmetric Bounded Sub-Span Evaluation | Reported as the latest/final SG-VCD round: 1,222 QA |

The report describes the move from one worker to four workers on one L4 GPU and records a Round 7 wall time of 1,072.5 seconds (17.88 minutes), with no reported CUDA OOM errors. It also reports the following Round 7 values:

| Metric reported by leader | Baseline column in Round 7 table | Round 7 | Reported change |
|---|---:|---:|---:|
| Substring Match / Inclusion, Verified GT | 62.27% | 80.14% | +17.87 percentage points |
| Token-F1, Verified GT | 72.70% (the table also shows a 51.60% / 72.70% pair in the baseline cell) | 80.93% | +8.23 points if 72.70% is the intended Verified value |
| Relaxed Accuracy / ANLS, Verified GT | 67.99% | 80.78% | +12.79 percentage points |
| Exact Match, Verified GT | 42.10% | 54.09% | +11.99 percentage points |
| Mean GT Inclusion Recall | 24.15% in the Round 7 progression table | 83.34% | +59.19 points, subject to baseline-definition reconciliation |
| Multi-Box Text Coverage | 23.80% | 85.84% | +62.04 points |

**Interpretation:** these results establish the leader-reported SG-VCD R7 result as the current method reference for planning the next baseline comparison. They do not replace a paper-method comparison, and they do not prove that SG-VCD outperforms methods such as ViCrop, VCD, M3ID, AGLA, AvisC or CRG unless those methods are evaluated under the same dataset version, reference-answer protocol, evaluator and execution setup.

### 1.2 Items that must be reconciled before using the reported values as a comparison table

The leader's report contains details that should be checked against code and raw artifacts before the next comparative run:

- The Round 7 table's baseline Token-F1 cell contains `51.60% / 72.70%`; the table should explicitly identify which value corresponds to Original References and which to Verified/Reviewed References.
- Section 8 reports a pre-upgrade Stage 1 Mean GT Inclusion Recall of 80.32%, while the Round 7 progression table uses 24.15% for the baseline. These appear to refer to different baseline definitions or evaluation configurations and must not be combined without reconciliation.
- Round 7 reports 1,072.5 seconds for 1,222 QA. This implies approximately 68.34 QA/minute by wall-time arithmetic, whereas the table reports approximately 80.35 QA/minute combined. The throughput definition (aggregate wall-clock throughput versus sum of worker throughput) must be stated explicitly.
- Round 1 and Round 5/7 also contain wall-time/throughput values whose calculation basis is not fully clear from the table alone.
- Round 7 introduces `Symmetric Bounded Sub-Span Evaluation` and changes the Token-F1/ANLS behavior. Scores produced with that evaluator must not be mixed with earlier-round scores or another evaluator without an explicit protocol label.

These are **report reconciliation items**, not a reason to discard the reported experiments. The next run should freeze one evaluator and record the exact run configuration so every method is directly comparable.

---

## 2. Research Objective and Questions

### 2.1 Objective

Evaluate whether the two-stage SG-VCD method provides measurable value over relevant published methods at each stage, under a common Vintern-based ViTextVQA evaluation setup where technically feasible.

The comparison must answer two separate questions:

- Does SG-VCD's **Stage 1** localize question-relevant scene-text evidence as effectively as relevant visual-grounding/cropping methods?
- Does SG-VCD's **Stage 2** decoding improve answer quality over published contrastive/calibration methods when the input evidence and model conditions are held constant?

A final end-to-end comparison should then assess whether the complete two-stage SG-VCD pipeline provides value beyond the strongest feasible stage-specific baselines.

### 2.2 Research questions

- **RQ1 — Stage 1:** How does SG-VCD localization compare with published training-free attention/grounding/cropping methods on question-conditioned scene-text localization?
- **RQ2 — Stage 1 to VQA:** When different localization methods feed the same Vintern answer-generation interface, which localization method produces the best downstream Token-F1, ANLS and Substring Match?
- **RQ3 — Stage 2:** Do published contrastive/calibration decoding methods improve ViTextVQA over the Vanilla Vintern control when evaluated with the same checkpoint, prompt, input image and answer evaluator?
- **RQ4 — ROI-conditioned decoding:** When a shared ROI is supplied, does SG-VCD's sharp-ROI versus low-pass-ROI contrastive mechanism improve answers beyond other region-conditioned guidance methods?
- **RQ5 — End-to-end:** Does the complete SG-VCD R7 pipeline outperform the strongest feasible Stage 1 and Stage 2 paper-method baselines under the same evaluation protocol?
- **RQ6 — Trade-off:** Are answer-quality changes worth the added latency, memory use and execution complexity?
- **RQ7 — Error profile:** Are gains consistent for small text, Vietnamese diacritics, numbers/phone numbers, multi-region questions and long signboard text?

These are hypotheses to test, not findings.

---

## 3. Method Taxonomy and Baseline Shortlist

A method is assigned to a stage by its *main intervention*, not simply because its implementation happens to use attention or image crops. Some methods span more than one stage; these must be labeled as hybrid or composite rather than forced into a single category.

### 3.1 Stage 1 — Spatial grounding, answer localization and visual cropping

| Priority | Paper / method | Core mechanism relevant to Stage 1 | Training requirement in the paper | Public source status checked for this plan | Fit and implementation decision |
|---|---|---|---|---|---|
| **Core S1-1** | **ViCrop** — *MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs*, ICLR 2025 | Uses model-internal attention and/or gradient importance maps to select question-relevant image crops; includes Relative Attention-based Visual Cropping and gradient-based variants. | Training-free inference-time intervention | Official repository: `https://github.com/saccharomycetes/mllms_know` | **Highest-priority Stage 1 comparator.** The repository includes implementations for selected LVLM families (including Qwen2.5-VL support); Vintern still requires a model-specific adapter. Reproduce the paper's selected variant exactly and label the Vintern port as an adaptation if it differs from the official path. |
| **Core S1-2** | **Localization Heads** — *Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding*, CVPR 2025 | Identifies a small set of text-to-image attention heads with high image-attention mass and spatially concentrated maps; combines them into a region mask/bounding box. | Training-free grounding using a frozen LVLM; head discovery/selection procedure must be preserved | Public repository: `https://github.com/seilk/LocalizationHeads` | **Second high-priority Stage 1 comparator.** The public repository provides attention collection, head analysis, mask and box generation. Its demonstrated model configuration is not Vintern; build a Vintern adapter only after confirming attention layout and token mapping. Do not transfer head indices from another architecture without validation. |
| **Supplementary S1-3** | **ST-VQA_Loc** — *Multimodal grid features and cell pointers for scene text visual question answering*, Pattern Recognition Letters 2021 | Question-conditioned attention over multimodal spatial grid features; cell pointers localize answer text. | Task-specific trained architecture using visual and OCR/text features | Public repository: `https://github.com/lluisgomez/ST-VQA_Loc` | Highly relevant to scene-text QA localization, but it is not a plug-in localizer for Vintern. Treat as a domain-specific architectural baseline/reference. Include in executable comparison only if its original training/data requirements can be reproduced fairly and within the agreed scope. |
| **Supplementary S1-4** | **Locate Then Generate (LTG)** — *Locate Then Generate: Bridging Vision and Language with Bounding Box for Scene-Text VQA*, AAAI 2023 | Answer Location Module (region proposal + language refinement) predicts answer-word regions; Answer Generation Module generates the answer. | Task-specific trained architecture | An official code repository was **not confirmed** in the source review for this plan. | Directly relevant to ST-VQA and bounding-box localization, but not a same-backbone Vintern adapter. Keep in related work unless official code and training requirements are verified. Do not reimplement it from the paper during the first pilot. |
| **Watchlist S1-5** | **FOCUS** — *Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering*, NeurIPS 2025 | Uses prompt-derived object/target concepts and KV-cache relevance maps to rank candidate crops for fine-grained VQA. | Training-free method described by the paper | Project page currently lists `Paper` and `Code (coming soon)`; no official code link was confirmed: `https://focus-mllm-vqa.github.io/` | Strong conceptual fit for small-detail cropping. Revisit if official source code becomes available and its licensing/model support are suitable. Do not implement from scratch in the first pilot because the leader prioritizes code-backed reproduction. |

**Stage 1 selection decision:** implement/port ViCrop and Localization Heads as the core Stage 1 paper-method comparisons. ST-VQA_Loc and LTG provide scene-text-specific methodological context; they are not assumed to be directly compatible with Vintern. FOCUS remains a watchlist item until official code is available.

### 3.2 Stage 2 — Contrastive decoding and visual/attention calibration

| Priority | Paper / method | Core mechanism relevant to Stage 2 | Public source status checked for this plan | Fit and implementation decision |
|---|---|---|---|---|
| **Core S2-1** | **VCD** — *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024 | Contrasts logits from the original image and a diffusion-style distorted/noised image at each autoregressive step. | Official repository: `https://github.com/DAMO-NLP-SG/VCD` | Required Stage 2 paper baseline. Use the original diffusion-noise transformation and aligned per-step logits. Do not replace it with Gaussian blur or ROI low-pass filtering and still call it original VCD. Adapt the decoding interface to Vintern and document deviations. |
| **Core S2-2** | **M3ID** — *Multi-Modal Hallucination Control by Visual Information Grounding*, CVPR 2024 | Compares image-conditioned logits with a text-conditioned reference that retains the question and generated prefix but omits visual conditioning; uses time-dependent correction and a confidence gate. | Paper and formula verified; an official source repository was **not confirmed** in the source review for this plan. | High-priority conceptual baseline. If implemented, call it a paper-faithful reimplementation/adaptation, not an official-code reproduction. Verify that the unconditioned branch truly removes image conditioning and scores the same text prefix. |
| **Core S2-3** | **AGLA** — *Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention*, CVPR 2025 | Uses image-prompt matching to build a prompt-relevant augmented view and combines global generative information with local discriminative information. | Official repository: `https://github.com/Lackel/AGLA` | Relevant hybrid global/local Stage 2 comparator. Preserve image-prompt matching and the paper's logit assembly; do not substitute a generic GT crop or a low-pass ROI and label it AGLA. Vintern compatibility requires an adapter. |
| **Core S2-4** | **AvisC** — *Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models*, Findings of ACL 2025 | Detects over-attended, query-irrelevant “blind tokens” and applies contrastive calibration against a branch with selected blind-token features suppressed. | Official repository: `https://github.com/sangminwoo/AvisC` | Relevant test-time attention-calibration baseline. Preserve the blind-token definition and contrastive branch. Do not describe it as direct attention-weight rescaling unless the source implementation does so. Adapt to Vintern only after inspecting attention tensors and token indexing. |
| **Supplementary S2-5** | **CRG** — *Contrastive Region Guidance: Improving Grounding in Vision-Language Models without Training*, ECCV 2024 | Uses a region prompt and a counterfactual image with the target region blacked out to down-weight predictions that do not depend on the selected region. | Public repository: `https://github.com/meetdavidwan/crg` | Relevant region-conditioned contrastive comparator. It is not original VCD and its negative view is not SG-VCD's low-pass ROI. Compare it only with the same ROI source (predicted or oracle) as other region-conditioned methods. |

### 3.3 Backbone control versus method baselines

| Label | Condition | Correct role | Must not be described as |
|---|---|---|---|
| **CTRL-0** | Vanilla `5CD-AI/Vintern-1B-v3_5` on the full image | Backbone control/reference; measures performance without the intervention under study | A paper-method baseline or a separate proposed method |
| **S1 baseline** | ViCrop or Localization Heads produces ROI; standard Vintern decoding reads the ROI under a common, documented input policy | Stage 1 paper-method comparator | SG-VCD or full original paper pipeline if the adaptation changes its native inference protocol |
| **S2 baseline** | VCD, M3ID, AGLA or AvisC on the full image | Stage 2 paper-method comparator | Stage 1 localization method |
| **ROI-guidance baseline** | CRG applied using a specified shared ROI | Region-conditioned Stage 2 comparator | Original VCD or SG-VCD |
| **Proposed method** | Latest SG-VCD R7 pipeline | Two-stage proposed method under evaluation | A paper baseline |
| **Composite comparator (optional)** | ViCrop localization followed by VCD decoding | A transparently named combination of two paper mechanisms | Original ViCrop, original VCD, or SG-VCD as a whole |

Vanilla Vintern remains in the experiment because a control is necessary to measure the value of intervention. It is not counted among the paper-method baselines in the baseline taxonomy or research contribution table.

### 3.4 Keep CERV separate from this SG-VCD comparison

The newly added `Proposed_Method_CERV.md` describes **CERV (Counterfactual Evidence Routing & Verification)** as a three-stage proposal centered on a Bayesian Semantic-Fork Gate, Prefix-Conditioned VAQ Localization and Fork-Anchored Counterfactual Verification.

CERV is not a renamed SG-VCD: it changes the number of stages and introduces a prefix-conditioned decision point. The current documentation package contains a method proposal but does not provide a CERV experiment report, run manifest or comparative result table. Therefore:

- This plan continues to evaluate the **two-stage SG-VCD** and its paper-method baselines.
- CERV is recorded as a separate future method proposal.
- Do not merge CERV mechanisms into SG-VCD baselines or label SG-VCD results as CERV results.
- A CERV pilot requires a separate research decision and protocol after the current baseline comparison is complete or explicitly re-scoped by the leader.

---

## 4. SG-VCD Method Being Compared

Use the latest leader-documented two-stage SG-VCD formulation as the proposed method reference.

### Stage 1 — Spatial grounding

The report describes question-conditioned, multi-layer attention/feature analysis to localize relevant text; remove attention-sink bias; estimate character scale; propagate spatial seeds; protect Vietnamese diacritic regions; and create one or more ROI tiles when required.

The Round 3–5/7 report describes successive enhancements including adaptive Top-K candidate grounding, pre-ROI Laplacian edge re-emphasis, wider/horizontally aligned regions and dynamic multi-patch ROI tiling.

### Stage 2 — Prior-calibrated contrastive decoding

The proposal defines:

- Positive visual input: global image plus sharp ROI.
- Negative visual input: the same global image plus low-pass-filtered ROI.
- Synchronized per-step logits over the same text prefix.
- Contrastive logit combination with a time-/evidence-dependent factor and candidate filtering.
- Round-specific additions such as CAPC, AHRA, QIACF, TESG and ESSC, as described in the latest report.

The exact Round 7 source code, config and evaluator version must be pinned for the comparative run. Do not substitute an earlier-round configuration while retaining the Round 7 label.

### Core methodological distinction

SG-VCD's low-pass ROI negative branch is **not** the same as:

- VCD's diffusion-forward-noised image;
- CRG's blacked-out target region;
- M3ID's text-only reference branch;
- AvisC's blind-token-suppressed branch; or
- AGLA's prompt-matched augmented image view.

These methods may all be contrastive or calibrated at decoding time, but they construct different counterfactual/reference inputs and test different hypotheses.

---

## 5. Revised Baseline Comparison Experiment

The main pilot is now a **paper-method baseline comparison** organized by SG-VCD's two stages. It is not the previous five-condition component ablation.

### 5.1 Experimental tracks

#### Track A — Stage 1 localization comparison

Compare the core Stage 1 methods:

1. ViCrop Relative-Attention-based Visual Cropping.
2. Localization Heads.
3. SG-VCD R7 Stage 1 (proposed method).
4. Oracle ROI only as a diagnostic upper-reference condition, not as a paper baseline.

For downstream answer comparison, feed every method's predicted region through the same Vintern answer reader and the same declared global/ROI input format. If a method natively uses multiple crops or a native multi-image interface, preserve its method logic and separately mark any harmonized Vintern adaptation. Report both localization quality and downstream answer quality.

Do not force every localizer into an identical one-box output if the paper method natively produces multiple regions. Instead, pre-register a common evaluation policy (for example top-1 ROI as the primary comparison and native top-K as a supplementary comparison) before seeing pilot scores.

#### Track B — Stage 2 full-image decoding comparison

Use the same original full image, question, Vintern checkpoint, prompt and standard preprocessing for:

1. Vanilla Vintern — control only.
2. Original VCD — paper-method baseline.
3. M3ID — paper-method baseline/reimplementation.
4. AGLA — paper-method baseline/adaptation.
5. AvisC — paper-method baseline/adaptation.

This track compares Stage 2 methods without introducing different Stage 1 crops. It therefore isolates decoding/calibration behavior on the full image.

#### Track C — Region-conditioned contrastive comparison

Compare methods that require or use a selected region:

1. CRG using a fixed, declared ROI source.
2. SG-VCD's global + sharp ROI versus global + low-pass ROI Stage 2.
3. Optional ViCrop → VCD composite comparator, if retained by the researcher.

For a fair Stage 2 comparison, every region-conditioned method in this track must receive the **same ROI coordinates** within a given sample. Run an oracle-ROI diagnostic separately from the predicted-ROI end-to-end condition. Never mix GT boxes into the inference path of the main predicted-ROI evaluation.

#### Track D — End-to-end proposed-method comparison

Compare the complete SG-VCD R7 pipeline against the strongest feasible stage-specific baseline pipelines from Tracks A–C. The final table must identify the exact localization method and decoding method used for each row. Do not call a combination of two papers an original paper baseline.

### 5.2 Main pilot condition matrix

| ID | Stage / role | Method condition | ROI source | Decoder | Comparison purpose |
|---|---|---|---|---|---|
| CTRL-0 | Control | Vanilla Vintern | None; full image | Standard decoding | Backbone reference only |
| S1-1 | Stage 1 paper baseline | ViCrop | Predicted by ViCrop | Standard Vintern reader | Training-free relative-attention cropping |
| S1-2 | Stage 1 paper baseline | Localization Heads | Predicted by Localization Heads | Standard Vintern reader | Training-free attention-head grounding |
| S1-3 | Proposed Stage 1 | SG-VCD R7 localizer | Predicted by SG-VCD | Standard Vintern reader for isolation; full SG-VCD also in Track D | Stage 1 comparison |
| ORACLE-ROI | Diagnostic | Human-reviewed GT ROI | Ground-truth annotation supplied only in a separately labeled oracle diagnostic | Standard Vintern reader | Diagnostic only; excluded from the main predicted-ROI/no-GT-leakage results |
| S2-1 | Stage 2 paper baseline | Original VCD | Full image | Official diffusion-noise VCD | Full-image contrastive decoding |
| S2-2 | Stage 2 paper baseline | M3ID | Full image + text-only reference branch | Paper-faithful M3ID | Visual-information / language-prior calibration |
| S2-3 | Stage 2 paper baseline | AGLA | Original + prompt-matched augmented view | Paper-faithful AGLA | Global/local feature assembly |
| S2-4 | Stage 2 paper baseline | AvisC | Original image + blind-token-suppressed branch | Paper-faithful AvisC | Attention-token calibration |
| S2-5 | Region-conditioned baseline | CRG | Same fixed ROI as the paired comparison | CRG black-out counterfactual | Region evidence guidance |
| PROP-0 | Proposed method | SG-VCD R7 end-to-end | SG-VCD predicted ROI(s) | SG-VCD R7 Stage 2 | Main proposed-method result |
| COMP-0 (optional) | Composite comparator | ViCrop → VCD | ViCrop predicted ROI | VCD | Explicitly labeled two-paper composite; not an original method |

The matrix may be reduced only after source-code compatibility and compute feasibility are documented. Do not remove Stage 1 comparison altogether. If one method cannot be ported, report the blocker and retain it in the method taxonomy rather than silently dropping it.

### 5.3 What is not part of the main pilot

The following are component diagnostics/ablations, not paper-method baselines:

- Sharp ROI versus low-pass ROI alone.
- Removing or adding the global image view.
- Fixed alpha versus dynamic alpha.
- Fixed beta versus entropy-adaptive beta.
- Disabling one SG-VCD gate at a time.
- Oracle ROI versus predicted ROI as a stand-alone component study.

These can be run later as ablations once the paper-method comparison has been completed. Do not mix them into the baseline table as if they were published methods.

---

## 6. Dataset, Annotation and Reference-Answer Protocol

### 6.1 Dataset scope

The leader's current report describes a 400-image / 1,222-QA subset. The research review also records that the member annotation folders (`Cao_Anh`, `Cam_Tu`, `Thanh_Tam`, `Trieu`) were updated and that the merged master must be rebuilt or verified against those latest files.

Before the comparative pilot:

1. Identify the exact current member annotation folders and current master answer file in the cloned repository.
2. Record SHA-256 hashes for every image, annotation JSON, answer master and dataset manifest.
3. Rebuild or verify a **new versioned** merged annotation master; never destructively overwrite the existing file.
4. Compare question IDs, image IDs, question text, answer references, sub-box coordinates and missing/duplicate records.
5. Preserve both original references and newly reviewed/adjudicated references as separate fields.
6. Record the source and review provenance for every additional reference answer. Do not automatically merge answers merely because two question strings are identical.
7. Do not describe a field as `human-verified` unless reviewer/provenance information supports that claim.

A previous independent audit of an earlier experiment/worktree identified risks in automatic answer expansion and slash-based splitting. Those findings must be rechecked against the latest cloned annotation files and their hashes; do not assume they automatically describe the newest leader-updated snapshot, and do not ignore them without comparing versions.

### 6.2 Reference-answer protocol

Use one of the following versioned protocols for the primary comparison, with the leader's approval:

- **Original references:** answer strings in the original answer master, preserved exactly before normalization.
- **Reviewed/adjudicated references:** answers explicitly confirmed against the latest annotation/image evidence with a traceable decision record.

If a second reference protocol is retained for sensitivity analysis, report it in a separate table. Do not combine Original and Reviewed references into a single unlabeled score.

The word `Verified` must be used only if its meaning and review provenance are defined. A pipeline-generated field name does not prove human verification by itself.

### 6.3 Development versus final evaluation

The 400-image / 1,222-QA subset has already been used for error analysis, method development and SG-VCD iteration. Treat it as **development/screening data**, not an untouched independent final test set.

For the proposed comparison pilot:

- Use a deterministic, image-level sample of 100 images from the 400-image development subset.
- Retain **all QA associated with each selected image**; do not sample individual questions independently while splitting the same image across partitions.
- Record the exact image IDs and resulting QA count in a versioned sample manifest before running any method.
- Use the same image/question manifest for all methods.
- A result on this 100-image pilot is a method-screening result, not confirmatory test evidence.
- If the pilot is stable, the researcher may authorize a full 400-image / 1,222-QA comparative run; that remains a development benchmark because the data have already informed method design.
- A genuinely confirmatory result requires an untouched evaluation source and a check that its images/predictions were not used for prior model or parameter selection.

---

## 7. Evaluation Metrics and Reporting Protocol

### 7.1 Required primary metrics

The three primary answer metrics remain exactly those required by the leader:

1. **Token-F1**
2. **ANLS**
3. **Substring Match**

Every method must be evaluated with the same frozen evaluator version and the same reference-answer protocol. Do not silently replace these metrics with Exact Match, CER, relaxed accuracy or another metric.

### 7.2 Evaluator freeze required before the pilot

The leader's Round 7 report introduces `Symmetric Bounded Sub-Span Evaluation`, with symmetric containment and bounded sub-span matching for ANLS/Token-F1. This is a project-specific evaluation variant and must be versioned and described explicitly.

Before running baselines:

- Identify the exact source-code version used to generate the Round 7 Token-F1, ANLS and Substring Match columns.
- Freeze the normalization rules (case, punctuation, whitespace, Unicode normalization, Vietnamese diacritics, numbers and units).
- Freeze tokenization and multi-reference aggregation rules.
- Freeze the symmetric containment/sub-span thresholds and length bounds, if the Round 7 evaluator is the selected primary evaluator.
- Add hand-checked test cases for Vietnamese diacritics, truncated phone numbers, dates containing `/`, prices/units, short and long references, multi-reference scoring and hallucinated extra text.
- Run all baseline methods through the same evaluator and save the evaluator source hash with every result.

If the researcher also wants a stricter conventional implementation for sensitivity analysis, report it as a separate evaluator track. Never compare one method scored with the Round 7 evaluator to another method scored with an older or stricter evaluator in the same headline column.

### 7.3 Supplementary diagnostics

The following may be included as supplementary metrics but do not replace the three required metrics:

- Exact Match / normalized EM;
- Character Error Rate (CER);
- Accent-bearing word or character/grapheme error analysis;
- Per-category scores for phone numbers, prices, addresses, names and Vietnamese diacritic-heavy words;
- End-to-end latency, throughput, peak VRAM/RAM, failure count and completion count.

If a diacritic-specific score is reported, define the Unicode normalization/alignment and denominators. Do not strip Vietnamese marks and then claim to measure diacritic recognition.

### 7.4 Stage 1 spatial metrics

For methods that output boxes or masks, report a common set of spatial measurements:

- Mean IoU against the appropriate per-question target box or explicitly defined target union;
- GT inclusion recall;
- Precision / predicted-area inclusion;
- Per-sub-box recall for multi-region questions;
- Area-weighted sub-box coverage;
- Fraction of required sub-boxes meeting a stated recall threshold;
- Fraction of all required boxes fully covered;
- Hit rate at pre-registered IoU/recall thresholds.

Do not use `MaxSubRecall = 1.0` or an “any box hit” metric as proof that all text regions were localized. Report single-region and multi-region QA separately. If a method does not output spatial localization, report the relevant spatial metrics as `N/A`, not as zero or invented values.

### 7.5 Statistical comparison

Because several questions may belong to one image, use paired image-level bootstrap for differences between methods. Report:

- Mean metric for each method;
- Paired difference versus the relevant control/proposed method;
- Confidence interval from image-level resampling;
- Number of images and QA pairs;
- Per-category breakdown and notable failures.

The 100-image pilot is primarily an engineering/effect-size screen. Avoid strong confirmatory claims from it.

---

## 8. Fairness, Reproducibility and Execution Controls

All comparable conditions must share:

- The same pinned Vintern checkpoint/revision;
- The same image list and question text;
- The same prompt format wherever the method does not require a paper-specific prompt;
- The same image preprocessing where possible, with paper-required differences explicitly documented;
- The same output extraction and stopping policy unless the original method requires another policy;
- The same answer evaluator and reference-answer protocol;
- A recorded random seed and decoding configuration;
- Separate output directories for each method and run.

For each paper method, record:

1. Paper citation and official repository URL;
2. Repository commit/tag and license;
3. Original supported model families;
4. Exact algorithm components reproduced;
5. Components adapted for Vintern;
6. Any deviations from the paper and why;
7. Model revision, prompt, image preprocessing and generation parameters;
8. Runtime and peak memory;
9. Output completeness and failure cases.

### 8.1 No ground-truth leakage

The inference interface may receive only the image, question and any method-required *predicted* region/metadata. Ground-truth answers and boxes must be loaded only by the evaluator after prediction is complete.

Oracle ROI is permitted only in a clearly labeled diagnostic condition. It must never be mixed into predicted-ROI headline results.

### 8.2 Immutable provenance

Every run must preserve:

- Git commit and dirty state;
- hashes of all executed source files, including untracked files;
- config and prompt hashes;
- model repository and pinned revision;
- tokenizer/config hash where available;
- dataset, annotation and sample-manifest hashes;
- environment/library versions;
- device and dtype;
- seed and generation settings;
- per-sample prediction, latency and method trace;
- checkpoint identity and resume provenance.

Use fresh method-specific run directories. A resumed run must verify configuration and dataset identity and skip completed samples by stable sample key, not cursor position alone. Do not mix inherited records from a different code/evaluator version into the new comparative results.

---

## 9. Revised Execution Roadmap

### Phase 0 — Reconcile the leader's reported SG-VCD state

1. Read the latest `Proposed_Method_SG_VCD.md` and preserve the leader-reported Round 1–7 results as historical experiment records.
2. Locate the actual Round 7 source code, config, raw predictions, evaluator and execution logs in the cloned repository or experiment storage.
3. Reconcile the baseline Token-F1 pair, Stage 1 baseline recall discrepancy and throughput arithmetic noted in Section 1.2.
4. Freeze the exact SG-VCD R7 checkpoint/config/evaluator identity to be used as the proposed-method reference.
5. Do not rerun SG-VCD R1–R7 unless an integrity discrepancy makes a targeted reproduction necessary.

### Phase 1 — Stage 1 paper-method feasibility and code audit

1. Inspect official ViCrop source and map its attention/crop logic to Vintern's actual visual token layout.
2. Inspect official Localization Heads source and identify how attention layers/heads map to Vintern; do not reuse LLaVA head indices blindly.
3. Run a read-only compatibility smoke test and produce a method adapter design for each.
4. Record ST-VQA_Loc and LTG requirements and decide whether their original training/data needs make them feasible as supplementary architecture baselines.
5. Recheck FOCUS code availability; do not independently recode the method solely because source code is unavailable.

Deliverable: `STAGE1_BASELINE_FEASIBILITY.md` with algorithm summary, code entry points, model compatibility, dependencies, license, adaptation points and estimated compute.

### Phase 2 — Stage 2 paper-method feasibility and code audit

1. Inspect official VCD implementation and reproduce its diffusion-noise preprocessing and synchronized decoding logic.
2. Inspect M3ID paper equations and determine whether a faithful Vintern implementation is possible; clearly mark it as a reimplementation if no official code is available.
3. Inspect official AGLA implementation and preserve the image-prompt matching/global-local assembly mechanism.
4. Inspect official AvisC implementation and preserve its blind-token selection and contrastive decoding mechanism.
5. Inspect official CRG implementation and define the same ROI source for paired region-conditioned comparisons.

Deliverable: `STAGE2_BASELINE_FEASIBILITY.md` with the same fields as Phase 1.

### Phase 3 — Evaluator and annotation freeze

1. Rebuild/verify a new versioned annotation master from the current leader-updated member folders.
2. Freeze Original versus Reviewed/Adjudicated reference protocols.
3. Freeze the required three metrics and exact evaluator version.
4. Run hand-checked metric test cases and save expected outputs.
5. Generate the deterministic 100-image manifest with all QA per selected image.

Deliverable: dataset manifest, annotation provenance report, evaluator specification and test report.

### Phase 4 — Twenty-sample integration test

Run a small deterministic integration set (approximately 20 QA, covering several images and relevant question types) for each method that has passed compatibility review.

This is a technical smoke test only. Check:

- Model loading and correct pinned revision;
- Correct attention/visual token indexing for Stage 1;
- Correct per-step logits/prefix alignment for Stage 2;
- Same sample coverage across methods;
- No ground-truth leakage;
- No failed/duplicate IDs;
- Output parsing and evaluator reproducibility;
- Runtime and peak resource measurement.

Do not publish 20-sample scores as baseline performance.

### Phase 5 — 100-image paper-method comparative pilot

Run the approved conditions in Section 5.2 on the frozen deterministic 100-image subset, retaining all associated QA. The primary objective is method comparison and integration feasibility, not hyperparameter tuning.

Report separate tables for:

1. Stage 1 localization results;
2. Stage 1 downstream VQA results using the shared reader;
3. Stage 2 full-image decoder results;
4. Region-conditioned decoder results;
5. End-to-end SG-VCD versus feasible paper-method pipelines;
6. Latency and resource use;
7. Error categories and examples.

Do not add arbitrary component-ablation conditions to this pilot.

### Phase 6 — Researcher review and decision

After the pilot, the researcher decides:

- Which Stage 1 method is the strongest feasible comparator;
- Which Stage 2 methods were reproduced faithfully enough to retain;
- Whether the 400-image comparative run is justified;
- Whether CERV should become a separate research track;
- Whether any later ablation is needed to explain a method-level result.

No full 400-image comparative benchmark or CERV experiment is launched automatically by this plan.

---

## 10. Decisions Ready to Adopt vs. Decisions Still Open

### 10.1 Ready to adopt in this plan

- SG-VCD remains a two-stage method under evaluation.
- Leader-reported Rounds 1–7 are treated as prior completed development experiments, subject to raw-artifact verification before quoting them as independently reproduced.
- Vanilla Vintern is a backbone control, not a paper-method baseline.
- Stage 1 baselines must be explicitly researched and compared; ViCrop and Localization Heads are the core code-backed candidates for a Vintern adaptation.
- ST-VQA_Loc and LTG remain domain-specific scene-text localization references; their training/architecture differences must be disclosed.
- Stage 2 baseline shortlist includes VCD, M3ID, AGLA and AvisC; CRG is a region-conditioned contrastive comparator.
- The main pilot compares paper methods by stage; component ablations are deferred.
- Token-F1, ANLS and Substring Match remain the three required answer metrics.
- The 400-image / 1,222-QA subset is development/screening data, not a pristine independent test set.
- CERV is kept separate from SG-VCD until the research scope is explicitly changed.

### 10.2 Decisions requiring researcher/leader confirmation

1. **Stage 1 core set:** approve ViCrop and Localization Heads as the two executable Stage 1 baselines on Vintern, subject to compatibility audit.
2. **Domain-specific methods:** decide whether ST-VQA_Loc or LTG should be run as separate trained architecture baselines or retained as related-work references only.
3. **Stage 2 scope:** approve VCD, M3ID, AGLA and AvisC as the core full-image decoding baseline set; approve CRG as a separate region-conditioned comparator.
4. **Metric version:** approve the exact Round 7 evaluator implementation as the primary scoring protocol, with any stricter/sensitivity evaluator reported separately.
5. **Reference protocol:** approve the versioned Original and Reviewed/Adjudicated answer-reference handling and clarify which is the primary reference set.
6. **Pilot size:** approve the deterministic 100-image development pilot and confirm the available Colab/Vast.ai compute budget.
7. **SG-VCD R7 provenance:** identify the exact source/config/evaluator/prediction artifact set corresponding to the leader-reported Round 7 table.
8. **CERV scope:** confirm that CERV is a separate future method proposal rather than the replacement target for this SG-VCD baseline comparison.

### 10.3 Implementation gate

Do not begin baseline coding until the Stage 1/Stage 2 feasibility reports, evaluator version, annotation master, sample manifest and approved method matrix are available. Do not start the 100-image pilot until the 20-sample integration test passes and the researcher explicitly authorizes the run.

---

## 11. Primary References

### Stage 1 — Localization / cropping

1. Zhang et al. (2025), *MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal MLLMs*, ICLR 2025. Paper: `https://proceedings.iclr.cc/paper_files/paper/2025/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html` · Official code: `https://github.com/saccharomycetes/mllms_know`
2. Kang et al. (2025), *Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding*, CVPR 2025. Paper: `https://openaccess.thecvf.com/content/CVPR2025/html/Kang_Your_Large_Vision-Language_Model_Only_Needs_A_Few_Attention_Heads_CVPR_2025_paper.html` · Public code: `https://github.com/seilk/LocalizationHeads`
3. Gómez et al. (2021), *Multimodal grid features and cell pointers for scene text visual question answering*, Pattern Recognition Letters. DOI: `https://doi.org/10.1016/j.patrec.2021.06.026` · Code: `https://github.com/lluisgomez/ST-VQA_Loc`
4. Zhu et al. (2023), *Locate Then Generate: Bridging Vision and Language with Bounding Box for Scene-Text VQA*, AAAI 2023. Paper: `https://ojs.aaai.org/index.php/AAAI/article/view/26357`
5. Zhong et al. (2025), *FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering*, NeurIPS 2025. Paper: `https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b02cdb04bc6351d36565b5c874bb804-Abstract-Conference.html` · Project page (code availability to be rechecked): `https://focus-mllm-vqa.github.io/`

### Stage 2 — Contrastive decoding / calibration

6. Leng et al. (2024), *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024. Paper: `https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html` · Official code: `https://github.com/DAMO-NLP-SG/VCD`
7. Favero et al. (2024), *Multi-Modal Hallucination Control by Visual Information Grounding*, CVPR 2024 (M3ID). Paper: `https://openaccess.thecvf.com/content/CVPR2024/html/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.html` · arXiv: `https://arxiv.org/abs/2403.14003`
8. An et al. (2025), *Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention*, CVPR 2025 (AGLA). Paper: `https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html` · Official code: `https://github.com/Lackel/AGLA`
9. Woo et al. (2025), *Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models*, Findings of ACL 2025 (AvisC). Paper: `https://aclanthology.org/2025.findings-acl.99/` · Official code: `https://github.com/sangminwoo/AvisC`
10. Wan et al. (2024), *Contrastive Region Guidance: Improving Grounding in Vision-Language Models without Training*, ECCV 2024 (CRG). Paper/project page: `https://contrastive-region-guidance.github.io/` · Official code: `https://github.com/meetdavidwan/crg`

---

## 12. Final Research Position

The next run should not attempt to prove SG-VCD again by comparing it only with Vanilla Vintern or by varying its own components. The leader has already reported a sequence of SG-VCD experiments through Round 7. The remaining research gap is a fair comparison against methods that address the same functions as the two SG-VCD stages.

Therefore, the next approved experiment should:

1. Compare at least two code-backed Stage 1 localizers (ViCrop and Localization Heads) on the same image/question set;
2. Compare code-backed Stage 2 decoding/calibration methods (VCD, AGLA, AvisC, and CRG where region input is required), plus a clearly labeled M3ID reimplementation if feasible;
3. Retain Vanilla Vintern strictly as the backbone control;
4. Evaluate all methods with the same Token-F1, ANLS and Substring Match implementation and reference-answer protocol;
5. Include the latest SG-VCD R7 as the proposed-method reference only after its provenance and evaluator are pinned;
6. Run a deterministic development pilot with paired image-level analysis and complete per-method provenance.

**Current status:** `READY_FOR_RESEARCHER_REVIEW — BASELINE PILOT DESIGN; EXECUTION REQUIRES RESEARCHER APPROVAL`.
