# SG-VCD Research Review and Experiment Decisions (Draft)

**Status:** Research design draft — not yet an implementation authorization  
**Target backbone:** `5CD-AI/Vintern-1B-v3_5`  
**Task:** Select and evaluate related decoding baselines for Vietnamese Text VQA, with emphasis on scene text and Vietnamese diacritics.  
**Primary project documents:** `docs/Proposed_Method_SG_VCD.md`, `docs/Diacritic_Aware_Contrastive_Decoding_Overview.md`  
**Data note:** The team reports that member annotation folders (`Cao_Anh`, `Cam_Tu`, `Thanh_Tam`, and `Trieu`) have been human-reviewed and updated. The merged annotation file must be rebuilt or verified against that latest version before a benchmark run.

> This document consolidates the research discussion and the two Antigravity App reviews. It distinguishes (1) decisions that are ready to adopt as a research plan, (2) hypotheses requiring experiments, and (3) decisions requiring project-leader approval. No code should be written from unresolved assumptions.

---

## 1. Research Objective

Determine whether localized ROI contrastive decoding provides measurable value for ViTextVQA beyond:

1. Vanilla Vintern inference;
2. Original Visual Contrastive Decoding (VCD) on the full image;
3. ROI augmentation without contrastive decoding; and
4. Other relevant methods such as M3ID and, if feasible, AGLA.

The research emphasis is Vietnamese scene text, small text regions, and diacritic-sensitive answers. General hallucination results in other VLM benchmarks do not establish that a method will improve ViTextVQA; this must be measured directly.

### Research questions

- **RQ1:** Do decoding-time contrastive methods improve ViTextVQA answers over vanilla Vintern?
- **RQ2:** Does adding a ground-truth ROI crop improve answers without contrastive decoding?
- **RQ3:** Does contrasting a sharp ROI against a low-pass ROI add benefit beyond using the sharp ROI alone and beyond whole-image VCD?
- **RQ4:** Are observed gains present on accent-preserving and character-level metrics, and are they worth the extra inference cost?
- **RQ5:** After the core mechanism is validated, does automated spatial grounding add benefit beyond oracle ROI inputs?

These are research questions, not established findings.

---

## 2. Methods and Source References

### 2.1 Vanilla Vintern

Use `5CD-AI/Vintern-1B-v3_5` as the control condition. Lock the model checkpoint, prompt template, image preprocessing, stopping criteria, and evaluation script. Re-run baseline predictions into a fresh output directory if the existing result files have uncertain provenance.

### 2.2 VCD — Visual Contrastive Decoding

**Primary paper:** Sicong Leng et al., *Mitigating Object Hallucination in Large Vision-Language Models via Visual Contrastive Decoding*, CVPR 2024.  
**Paper:** https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html  
**Official repository identified by the project review:** https://github.com/dungda/VCD

VCD contrasts the current-token logits from the original image against those from a corrupted visual input. The official implementation uses a diffusion-style forward noising helper (`add_diffusion_noise`) with a noise schedule and a `noise_step` parameter; it should not be substituted with Gaussian blur or described as arbitrary fixed-sigma pixel noise. Reproduce the official transformation at the correct point in the preprocessing pipeline and record the exact implementation and parameters used.

General contrastive form:

\[
\tilde z_t=(1+\alpha)z_t^{(+)}-\alpha z_t^{(-)}
\]

Apply the combination to aligned logits at every generation step, using the same text prefix for both branches. Two completed answers generated independently and compared after generation are not VCD.

### 2.3 M3ID — Multi-Modal Mutual-Information Decoding

**Primary paper:** Alessandro Favero et al., *Multi-Modal Hallucination Control by Visual Information Grounding*, CVPR 2024.  
**Paper:** https://openaccess.thecvf.com/content/CVPR2024/html/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.html  
**arXiv:** https://arxiv.org/abs/2403.14003

M3ID compares image-conditioned logits with a text-conditioned reference that preserves the question and preceding generated tokens but removes visual conditioning. Its correction also includes a step-dependent coefficient and is gated by the model's confidence according to the original formulation. It is not equivalent to VCD with a blurred image, and its reference branch should not be described as an entirely empty text prompt.

Adaptation to Vintern needs careful handling of its model-specific chat/image-input interface: the reference branch must truly omit visual conditioning, while both branches score the same next token under the same generated text prefix.

### 2.4 AGLA — Assembly of Global and Local Attention

**Primary paper:** Wenbin An et al., *Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention*, CVPR 2025.  
**Paper:** https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html  
**Official repository:** https://github.com/Lackel/AGLA

AGLA combines global image information useful for generation with prompt-relevant local information used for visual discrimination, including an image-prompt matching process. It is not synonymous with cropping a ground-truth text box and blurring it. Any Vintern adaptation must preserve the defining AGLA mechanism; a simplified global-plus-ROI experiment must be named as an adaptation/ablation, not claimed to reproduce the original AGLA unless validated against its algorithm.

### 2.5 AvisC — Attentional Vision Calibration

**Primary paper:** Sangmin Woo et al., *Don’t Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models*, Findings of ACL 2025.  
**Paper:** https://aclanthology.org/2025.findings-acl.99/  
**Official repository:** https://github.com/sangminwoo/AvisC

AvisC identifies “blind” image tokens through layer-wise attention analysis and uses a contrastive decoding strategy comparing logits from the original inputs with logits after selectively zeroing blind-token features. The paper describes this as calibration without directly modifying the underlying attention mechanism. Do not characterize it as directly scaling attention weights or key/value tensors unless the exact official implementation path being reused demonstrates that behavior.

**Status:** Keep AvisC in related work. Reconsider implementation only after inspecting the official algorithm, its required attention outputs, and adaptation cost on Vintern.

---

## 3. SG-VCD Proposal: Current Conceptual Definition

The current proposal contains two broad stages:

1. **Spatial grounding:** identify a relevant text region and form a high-resolution ROI.
2. **ROI contrastive decoding:** use a positive branch containing the global image plus a sharp ROI, and a negative branch containing the same global image plus a low-pass-filtered version of that ROI; combine logits at each autoregressive step.

Conceptual branch inputs:

\[
X^{(+)}=[I_{global},I_{ROI,sharp}],\qquad
X^{(-)}=[I_{global},I_{ROI,lowpass}]
\]

The global view and question should be held constant across the branches. The ROI variants should have identical crop coordinates and dimensions; only the documented filtering operation should differ. Both branches must use the same current generated text prefix and their logits must be aligned for per-step combination.

This is the proposal's core hypothesis, not a proven result. Because multimodal transformer computations are nonlinear, retaining the global image in both branches does not guarantee that features for global distractors cancel perfectly.

---

## 4. Mathematical Risk Register

### 4.1 Temporal multiplier

The proposal specifies:

\[
M_{decay}(t)=1+\gamma\frac{1-e^{-\lambda t}}{e^{-\lambda t}+\epsilon}
\]

For fixed `epsilon > 0`, this function has a finite limiting value:

\[
\lim_{t\to\infty}M_{decay}(t)=1+\frac{\gamma}{\epsilon}
\]

It is therefore inaccurate to call it unbounded to infinity with a fixed positive epsilon. However, if epsilon is chosen extremely small as if it were only a machine-precision guard, the finite ceiling can still be dangerously large. The proposal does not yet lock concrete values for all parameters.

**Required experiment before adopting the dynamic formula:** evaluate the exact proposal configuration over the intended answer length, logging `M_decay(t)`, the effective contrastive coefficient, logit ranges, and decoding anomalies. Compare the original expression with a bounded alternative as a separately labeled variant. Do not silently replace the proposal formula.

### 4.2 Entropy-adaptive candidate threshold

\[
\beta_t=\beta_0e^{-H(P_t^{(+)})}
\]

This is a design heuristic. Higher entropy reduces the relative candidate threshold, but entropy alone does not determine the exact number of retained tokens; the full probability distribution matters. A wider candidate set may preserve plausible low-probability characters, but it may also admit noisy tail tokens that contrastive subtraction elevates.

**Pilot check:** log entropy, `beta_t`, candidate-set size, rank of selected tokens under the positive distribution, and frequency of outputs whose positive-branch rank is outside the top 50. Compare against fixed beta on development data.

### 4.3 KL/JS divergence gate

With natural logarithms, `D_JS` lies in `[0, ln(2)]`; with base-2 logarithms, it lies in `[0, 1]`. KL is asymmetric and unbounded, so it can saturate a `tanh` factor and can be numerically fragile when calculations are performed in low precision.

**Pilot check:** compute log probabilities and divergence terms with stable operations in at least FP32, avoid unnecessary probability truncation, document the log base, and log non-finite values and the distributions that cause them. Ensure the JS threshold uses the same log base as its chosen threshold.

### 4.4 Character-height estimator

The proposed autocorrelation/second-derivative expression is not operationally complete. It lacks a defined local image window, projection axis, smoothing rule, and deterministic root-selection rule. Arbitrarily clipping its output is not sufficient to define a stable estimator.

**Decision:** exclude this estimator from the initial pilot. If the core ROI contrastive hypothesis is supported, define and test the spatial-grounding estimator separately. Use an established text spotter only as an explicitly authorized fallback, not silently as the original SG-VCD Stage 1.

### 4.5 Distractor cancellation

The claim that logits for a global distractor cancel because the global image is identical in both branches is unsupported. The two ROI inputs can alter transformer hidden states and cross-modal interactions. Treat cancellation as a hypothesis to inspect through error cases, not as a mathematical guarantee.

---

## 5. Five-Condition Exploratory Pilot

The first pilot tests the core mechanism using human-reviewed ground-truth ROIs, avoiding the additional uncertainty introduced by automatic attention-based localization. It is an exploratory engineering/effect-size screen, not final confirmatory evidence.

All five conditions must use the same selected image-question set, checkpoint, prompt logic, text normalization for evaluation, maximum generation length, and stopping criteria. Keep all questions associated with an image in the same sample/split. Use a deterministic image-level manifest.

| Condition | Inputs | Decoding | What it tests | What it cannot prove |
|---|---|---|---|---|
| **C0: Vanilla** | Global image + question | Standard greedy decoding | Base model performance | Cannot isolate OCR resolution vs language-prior errors |
| **C1: Oracle ROI** | Global image + sharp human-reviewed ROI + question | Standard greedy decoding | Whether extra high-resolution ROI context helps | Not a guaranteed upper bound; added view may confuse or distract the model |
| **C1-lowpass** | Global image + low-pass ROI + question | Standard greedy decoding | The model response to the filtered local view | A lower score alone does not prove Vietnamese diacritics specifically were removed |
| **C2: Original VCD** | Positive branch: global image; reference branch: diffusion-corrupted global image | Original VCD at each generation step | Whole-image contrastive decoding under the official transformation | Cannot by itself establish performance on diacritic errors without targeted analysis |
| **C3: ROI contrastive core** | Positive: global + sharp ROI; negative: same global + low-pass ROI | Fixed-parameter synchronized contrastive decoding per token | Whether localized ROI contrastive decoding adds benefit beyond C1 and C2 | Does not establish that full dynamic SG-VCD or automatic grounding is superior |

For C2, reproduce the official VCD corruption operation and its parameters faithfully. Do not substitute Gaussian blur or an arbitrary noise transform and still call it the original VCD.

For C3, the sharp and low-pass ROI must have the same crop geometry and resolution. The negative branch must preserve the exact same global image, question, and generated prefix. The contrastive logit operation must run at each autoregressive token step.

### 5.1 Extra diagnostic needed

C1-lowpass is retained as a diagnostic. Its score should be interpreted alongside qualitative crop inspection and character-level outcomes. If the low-pass ROI does not visibly or measurably attenuate relevant fine strokes, revise the filter as a separate experiment rather than assuming it forms a valid negative visual view.

A future condition comparing the sharp-ROI positive branch against a text-only reference could help compare localized low-pass contrast with a language-prior reference (conceptually related to M3ID). This belongs in a later comparison, not automatically in the first pilot unless the leader expands the scope.

### 5.2 Pilot scale and interpretation

A deterministic sample of 100 images from the reviewed 400-image set (retaining every question attached to each selected image) may be used for engineering screening. The 400-image subset has already contributed to error analysis and method design, so it must be treated as **development/screening data**, not an independent final test.

Do not conclude that all SG-VCD variants are invalid if one pilot configuration fails. A negative pilot result weakens the tested configuration and triggers diagnosis of crop quality, filtering, prompt formatting, and logit calibration.

---

## 6. Evaluation Metrics

### Required general metrics

1. **Answer EM / normalized EM:** match the project's official evaluator and clearly document each normalization operation. If using the word “strict,” specify whether punctuation and case are ignored; do not describe a punctuation-stripped comparison as raw literal exact match.
2. **Accent-preserving ANLS or normalized edit similarity:** normalize to Unicode NFC, apply the documented case/punctuation policy, and preserve every Vietnamese vowel modifier and tone mark.
3. **Character Error Rate (CER):** calculated after accent-preserving normalization, so diacritic changes remain errors.
4. **Runtime and resource metrics:** end-to-end latency per question, throughput, peak VRAM/RAM, failure count, and number of completed predictions.

### Diacritic-specific analysis

The previous word-level metric called “Diacritic F1” conflates vocabulary errors with diacritic errors. Rename exact word matching for words containing marked Vietnamese letters to **Accent-Bearing Word Exact Match (AB-WEM)** if retained.

If reporting a separate diacritic metric, use a reproducible character/grapheme alignment procedure and distinguish:

- Base-letter substitution/insertion/deletion;
- Vowel-shape differences (`a`/`ă`/`â`, `e`/`ê`, `o`/`ô`/`ơ`, `u`/`ư`);
- Tone-mark differences (level/unmarked vs sắc, huyền, hỏi, ngã, nặng);
- `d` versus `đ` as a separate letter category;
- Character insertions/deletions where alignment cannot attribute an error solely to a mark.

Use Unicode NFC for comparison and, if decomposing characters to inspect marks, use Unicode NFD internally while preserving the mapping to the original grapheme. Do not strip Vietnamese marks. Define precision/recall denominators and alignment rules before producing a number called `Diacritic F1`.

### Spatial metrics

IoU and ground-truth inclusion metrics apply only to methods that produce comparable boxes. For other methods, do not invent spatial results. Oracle-ROI VQA performance can be reported as an oracle condition, not as predicted localization accuracy.

---

## 7. Data, Split, and Statistical Policy

1. The four member annotation folders have been reviewed and updated according to the team. Before running, rebuild the aggregate file into a **new versioned output** (or back up the existing master) and verify it against the latest member files. Record a manifest, file counts, stable image IDs, and checksums. Avoid destructive overwrite until equivalence is checked.
2. Treat the existing 400-image / 1,222-QA subset as development and screening data because it was used for error analysis and method design. Do not create a pristine test claim by splitting it post hoc.
3. Preserve image-level grouping: all questions tied to a single image remain in one partition.
4. A remaining portion of the official dev set may be an independent evaluation candidate only after verifying that its images and predictions were not used for prior model/parameter selection. Verify counts from manifests rather than relying on estimates in reports.
5. The official test set can be evaluated on answer metrics if its ground-truth answers are available and it has not been used for model selection. Spatial metrics requiring human boxes need additional reviewed annotations or must be omitted.
6. Use paired image-level bootstrap for score differences because multiple questions share each image. Report the paired difference and a confidence interval; clearly distinguish pilot screening from confirmatory evaluation.
7. Fix one evaluator implementation and run every condition through it. Use fresh, configuration-specific output directories. Every result file must store checkpoint ID, method/configuration, dataset manifest hash, prompt, generation settings, and timestamp/seed where applicable. Never trust a “100% completed” resume flag without verifying configuration identity and output integrity.

---

## 8. Recommended Baseline Roadmap

### Phase 0 — Integrity checks

- Verify the latest reviewed annotations and regenerate a versioned master file.
- Resolve prior cases where supposedly different model prediction files were byte-identical.
- Verify that existing baseline outputs correspond to the intended checkpoints and configuration.
- Confirm the evaluator's normalization and metric definitions.

### Phase 1 — Minimal core pilot

Run C0, C1, C1-lowpass, C2, and C3 on the same deterministic 100-image exploratory sample. Do not implement attention-based Stage 1 at this point.

### Phase 2 — Baseline comparison

If the pilot and integration checks are valid, implement **M3ID** as a key reference baseline alongside vanilla, original VCD, and the ROI contrastive core. Inspect official AGLA code and estimate porting cost before deciding whether it is a core baseline or a supplementary experiment. Keep AvisC in related work unless its implementation path on Vintern is shown to be feasible.

### Phase 3 — Full SG-VCD and ablations

Only after the ROI contrastive core shows a credible signal, implement automatic spatial grounding and the dynamic gates. Ablate localization, ROI filtering/negative view, and dynamic calibration separately. Label any formula changes as method variants.

---

## 9. Decisions Ready to Adopt vs. Decisions Still Open

### Ready as a research plan

- Compare against vanilla Vintern and preserve the original VCD definition.
- Evaluate M3ID as the next high-priority baseline after the core pilot.
- Use the five-condition exploratory pilot defined in Section 5.
- Treat 400 reviewed images as development/screening data.
- Preserve Vietnamese diacritics in evaluation and define diacritic-specific metrics transparently.
- Keep attention-based localization and dynamic gates out of the first pilot.
- Record all outputs/configurations separately and ensure paired logits are combined at each generation step.

### Decisions requiring project-leader confirmation

1. **Pilot scope:** approve the five-condition screening experiment on 100 images.
2. **Annotation master:** approve freezing a newly regenerated, versioned aggregate annotation file after it is checked against the latest reviewed member files.
3. **Proposal Table 5.1 claims:** confirm whether the reported 72.70% Token F1 and 50.08% EM are results from an uncommitted preliminary run, estimates, or target values. Do not silently re-label them; document the answer from the leader.
4. **Stage 1 fallback:** decide whether an established text spotter such as SwinTextSpotter may be used as an ROI provider if the proposed attention-based localizer is unstable. If allowed, report this as a fallback/localization variant.
5. **Evaluation source and compute:** confirm which untouched data can serve as final evaluation and whether Colab/Vast.ai GPU resources are available for confirmatory runs.

### Gate before implementation

Do not hand off a final coding prompt until the pilot scope, annotation-master policy, proposal-result provenance, fallback policy, and evaluation source are recorded. The IDE can inspect the repository and propose an implementation plan only after reading this document; it must not start coding while any explicitly blocking decision remains unresolved.

---

## 10. Primary References

- Leng et al., CVPR 2024, VCD: https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html
- Official VCD repository: https://github.com/dungda/VCD
- Favero et al., CVPR 2024, M3ID: https://openaccess.thecvf.com/content/CVPR2024/html/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.html
- M3ID arXiv: https://arxiv.org/abs/2403.14003
- An et al., CVPR 2025, AGLA: https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html
- Official AGLA repository: https://github.com/Lackel/AGLA
- Woo et al., Findings of ACL 2025, AvisC: https://aclanthology.org/2025.findings-acl.99/
- Official AvisC repository: https://github.com/sangminwoo/AvisC
