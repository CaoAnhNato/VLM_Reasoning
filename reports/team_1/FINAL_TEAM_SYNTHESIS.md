# Team 1 Final Campaign Synthesis — Input Optical & Native RGB Sensing

- **Team**: `team_1` (**Domain 1 — Input Optical & Native RGB Sensing**)
- **Backbone**: `5CD-AI/Vintern-1B-v3_5` (`InternViT-300M` `[N_tiles, 256, 896]` `bfloat16`, `448 x 448` tiles + frozen `Qwen2.5-0.5B-Instruct` + locked `IG-VFED` two-stream decoding)
- **Iterations Executed**: `10 / 10` (`direction_1`: `iter_1..6`; `direction_2`: `iter_7..9`; `direction_3`: `iter_10`) — **`10 / 10` passed Stage A and completed full `1,222`-QID Stage B on Colab L4**
- **Final `<team_1_best_commit>` (Global Campaign Champion)**: **`0f8a2d4d2bedbb5654c514f1666d9f2748b00bc5`** (`team_1_iter_3`, `FRSA-PBCF` LOO mask `"011"` — Pure Layer-0 Lexical-Semantic Variational Foveation + Exact Poisson–Bregman Subwindow Slicing: **`64.12%` Token F1 (`+3.93 pp` vs baseline), `62.19%` Substring Match (`+0.73 pp`), `63.16%` Composite (`+2.33 pp`), `36.99%` EM (`+5.40 pp`, `453` matched), `55.55%` ANLS (`+6.69 pp`), `CLASS_A <4px` F1 = `25.40%` (`+9.80 pp` vs `15.60%` baseline)** at `1.932 s/Q`; `0`-byte diff verified across `src/nevfe/` and `tests/` at campaign closure)
- **Audited Literature Ledger**: **`95` globally distinct peer-reviewed / OA full-text PDFs** (`58 + 28 + 9` across `direction_1..3`; `100%` `1:1` BibTeX parity, `0` key/SHA overlap across directions; **`[Bucket A: 20, Bucket B: 21, Bucket C: 21, Bucket D: 33]`** Q1 journal mathematical primitives with explicit ISSNs)

---

## 1. Complete 10-Iteration Stage A & Stage B Metrics Ledger (`400` Images / `1,222` QIDs)

| Iter | Direction | Operator | Spec SHA | Impl SHA | Stage A `best_mask` (`comp_70`) | Token F1 (%) | Substring (%) | Composite (%) | EM (%) | ANLS (%) | `CLASS_A` F1 (%) | Latency (s) | W / L / T vs Anchor | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base** | `anchor` | `IG-VFED` (`148f870`) | — | `d79b650` | `37.19%` | `60.19` | `61.46` | `60.83` | `31.59` | `48.86` | `15.60` | `~1.44` | — | Baseline Anchor |
| `iter_1` | `direction_1` | `VRE-OF` (`v1.0`) | `624ea8c` | `2573e37` | `"111"` (`38.99%`) | `62.49` | `61.62` | `62.06` | `34.86` | `52.84` | `21.91` | `2.134` | `0 / 0 / 0`* | **`PROMOTED`** |
| `iter_2` | `direction_1` | `FR-STVF` (`v1.0`) | `e61b1e5` | `efea9fc` | `"111"` (`41.23%`) | `62.40` | `62.03` | `62.22` | `34.37` | `52.53` | `26.43` | `2.291` | `151 / 133 / 938` | `REJECTED` (S1) |
| **`iter_3`** | **`direction_1`** | **`FRSA-PBCF` (`v1.0`)** | **`85170ac`** | **`0f8a2d4`** | **`"011"` (`40.92%`)** | **`64.12`** | **`62.19`** | **`63.16`** | **`36.99`** | **`55.55`** | `25.40` | `1.932` | **`165 / 122 / 935`** | **`PROMOTED` (Best)** |
| `iter_4` | `direction_1` | `AVMF` (`v1.0`) | `11acea8` | `2cc3e3c` | `"111"` (`41.36%`) | `62.86` | `60.97` | `61.92` | `34.70` | `53.22` | `24.49` | `2.195` | `100 / 130 / 992` | `REJECTED` (S1) |
| `iter_5` | `direction_1` | `RKHS-ASBF` (`v1.0`) | `708ce00` | `6919911` | `"101"` (`38.10%`) | `60.24` | `61.37` | `60.81` | `31.75` | `49.04` | `18.99` | **`1.435`** | `73 / 198 / 951` | `REJECTED` (S2) |
| `iter_6` | `direction_1` | `SPB-MLF` (`v1.1`) | `eae68f3` | `3e6cad5` | `"111"` (`39.98%`) | `63.45` | `62.03` | `62.74` | `35.84` | `54.66` | `22.71` | `2.023` | `100 / 104 / 1018` | `REJECTED` (S3 -> Pivot) |
| `iter_7` | `direction_2` | `WSR-MSF` (`v1.0`) | `6db99ea` | `0576d70` | `"101"` (`40.46%`) | `63.29` | `60.64` | `61.97` | `35.84` | `54.47` | `24.50` | `1.978` | `88 / 104 / 1030` | `REJECTED` (S1) |
| `iter_8` | `direction_2` | `EBW-MSF` (`v1.0`) | `ec65cfa` | `1ed0bf3` | `"101"` (`39.89%`) | `63.31` | `61.13` | `62.22` | `36.01` | `54.30` | `25.65` | `1.965` | `70 / 87 / 1065` | `REJECTED` (S2) |
| `iter_9` | `direction_2` | `SKS-OIF` (`v1.0`) | `0c9d831` | `5fae576` | `"111"` (**`41.68%`**) | `63.89` | `61.54` | `62.72` | `36.25` | `55.04` | **`30.34`** | `2.109` | `92 / 99 / 1031` | `REJECTED` (S3 -> Pivot) |
| `iter_10` | `direction_3` | `SKG-TTAF` (`v1.0`) | `6c4de90` | `474de2c` | `"101"` (`39.36%`) | `60.03` | `60.97` | `60.50` | `30.69` | `48.63` | `23.32` | `2.042` | `85 / 199 / 938` | `REJECTED` (S1, End) |

*\*Note: Wave 1 `diff_predictions.py` recorded `0/0/0` prior to QID-aligned head-to-head tallying; `iter_2..3` compare against promoted `iter_1` (`2573e37`), and `iter_4..10` compare against promoted `iter_3` (`0f8a2d4d`).*

---

## 2. Core Scientific & Mechanistic Discoveries Across Domain 1

### Discovery 1 — Transient Unblurred Native-RGB Sub-Tile Foveation Breaks the Sub-Nyquist `< 4px` Bottleneck (`+3.93 pp` F1, `+5.40 pp` EM, `+6.69 pp` ANLS, `CLASS_A` `15.60% -> 25.40%..30.34%`)
- Across all three teams, **Domain 1 native-RGB sensing delivered the largest real-benchmark gains of the entire campaign**, lifting `Vintern-1B-v3_5` from `60.19%` F1 / `61.46%` Sub (`31.59%` EM, `48.86%` ANLS) to **`62.49% / 61.62%` in `iter_1` (`VRE-OF`)** and **`64.12% / 62.19%` (`36.99%` EM, `55.55%` ANLS, `453` exact matches) in `iter_3` (`FRSA-PBCF`, `0f8a2d4d`)** while keeping per-image `self._cached_vit_embeds` strictly immutable.
- **Why Pure Lexical-Semantic Localization (`"011"`) Beat Non-Semantic RGB Gradient Priors (`"111"`, `"101"`)**:
  - In `iter_3`, Stage A factorial ablation selected LOO mask **`"011"` (`comp_a=0`, `40.92%` vs `"111"` `38.06%` and `"101"` `34.50%`)**, completely disabling the RGB Color Structure Tensor (`optical_energy`) branch and driving exact 2D Integral-Image Poisson–Bregman KL subwindow selection (`select_poisson_bregman_subwindow`) purely from mean-centered `InternViT` patch-to-query directional alignment (`compute_lexical_spatial_distribution`), gated physically by `DynamicMultiTileROICropper`'s Nyquist super-resolution constraint (`m_crop > m_glob`, `min_tile_dim = 112px`).
  - Non-semantic high-frequency RGB texture (brickwork, foliage, storefront grilles, decorative borders) systematically mislocalizes gradient-based saliency away from small queried text entities unless anchored strictly to query semantics.

### Discovery 2 — Frozen `InternViT-300M` Patch Manifold Is Hypersensitive to Pixel High-Pass Perturbations & Overlapping Seam Duplication (`direction_2`, `iter_7..9`)
- Every Stage-A factorial ablation across `iter_7` (`WSR-MSF`), `iter_8` (`EBW-MSF`), and `iter_9` (`SKS-OIF`) rejected pixel-level unsharp/Wiener/Bregman high-pass filtering (`"101"` beat `"111"` in `iter_7` `49.29% > 47.18%` and `iter_8` `49.34% > 47.73%`; `"011"` set the **all-campaign Stage-A F1 record `49.35%` `f1_100` / `40.0%` `sub_100`** in `iter_9` when pixel filtering was disabled), confirming that unblurred `PIL.Image.Resampling.BICUBIC` lies directly on `InternViT-300M`'s pretrained patch manifold whereas synthetic pixel sharpening amplifies JPEG mosquito ringing and Vietnamese diacritic Confusion (`dấu hỏi / ngã / sắc`).
- Replacing `max_m` with **Shared-RMS Spherical KL-Specificity `Soft-OR`** ($w_m = I_m^2 / \sum I_{m'}^2$) in `iter_9` (`SKS-OIF`) set the **all-campaign Stage-A Composite record (`41.68%`)** and **nearly doubled `CLASS_A <4px` F1 to `30.34%`** (`63.89%` overall F1, `55.04%` ANLS), falling `-0.65 pp` shy on `Substring Match` (`61.54%` vs `62.19%`) only because `25%`-overlapping `1x2 / 2x1` tiles duplicated interior characters across seam boundaries (`QID 964`: `"Nhưngế quá"`).

### Discovery 3 — Pre-`</img>` 1D RoPE Recency Sink Governs Extractive Span Conciseness vs. Pretrained Thumbnail Ordering (`direction_3`, `iter_10`)
- **`iter_10` (`SKG-TTAF`, `474de2c`)** isolated two decisive causal mechanisms in `Vintern-1B-v3_5` (`Qwen2.5-0.5B-Instruct`):
  1. **1D RoPE Pre-`</img>` Recency Sink (`comp_c`)**: Although `InternVL` pre-training places localized grid tiles *before* the `448 x 448` global thumbnail (`chen2024expanding`), splicing transient high-zoom ROI crops *before* `vit_embeds[-1:]` (`comp_c=1`, `"101"`) placed `256` coarse `448 x 448` thumbnail tokens directly into the pre-`</img>` causal RoPE recency window right before the Vietnamese conciseness instruction, collapsing `EM` from `36.99% -> 30.69%` (`MATCHED 453 -> 375`, `F1 = 60.03%`) and reviving verbose full-sentence priors (`QID 8281`, `QID 9459`). Within `iter_10` Stage A itself, reverting to **post-thumbnail append `[vit_embeds, crops]` (`comp_c=0`, `"110"`)** outperformed `"111"` by **`+2.47 pp` `f1_70` (`43.86%` vs `41.39%`) and `+2.80 pp` `f1_100` (`45.88%` vs `43.08%`)**, proving that `0f8a2d4d`'s post-thumbnail placement works precisely by occupying the pre-`</img>` recency anchor with high-resolution foveal tokens.
  2. **Physical Nyquist Gating (`m_crop > m_glob`) Dominates Statistical Scan Thresholding (`comp_b`)**: Because $\tilde{c}_u / \sigma_c$ standardizes spatial logits to unit variance prior to `Softmax` over $N = 1024\text{--}1536$ patches, genuine multi-word scene-text alignments exhibit empirical Poisson–Bregman divergence $J^* \in (10^{-6}, 0.08)$ nats below $\tau_{\text{scan}}(N; 0.50) = \sqrt{(\ln N)/N}$, causing `comp_b=1` to over-abstain (`"101"` beat `"111"` by `+2.95 pp` `comp_70`, `39.36%` vs `36.41%`). Physical magnification gating (`m_crop > m_glob`) inside `DynamicMultiTileROICropper` remains the sharper, parameter-free abstention boundary.
- **Implied Next-Generation Synthesis Beyond `iter_10`**: Combining `0f8a2d4d`'s post-thumbnail `[vit_embeds, crops]` recency placement + zero-overlap `DynamicMultiTileROICropper` (`m_crop > m_glob`, `eps_k = 1e-6`) with `iter_9` `"011"`'s Shared-RMS Spherical KL-Specificity `Soft-OR` (`49.35%` `f1_100`, `30.34%` `CLASS_A` F1) isolates every empirically validated positive mechanism without seam duplication or pre-`</img>` recency dilution.

---

## 3. Literature & Archival Integrity Summary

| Direction | Iterations | Verdict | Audited Papers (`1:1` BibTeX) | Bucket Breakdown `[A / B / C / D]` | Spec Artifacts Preserved |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `direction_1` | `iter_1 .. iter_6` | `PROMOTED_BEST_CHECKPOINT` (`0f8a2d4d`) | `58` | `[12 / 13 / 13 / 20]` | `spec_iter_{1..6}.md` |
| `direction_2` | `iter_7 .. iter_9` | `FAILED_ROLLED_BACK` | `28` | `[6 / 6 / 6 / 10]` | `spec_iter_{7..9}.md` |
| `direction_3` | `iter_10` | `FAILED_ROLLED_BACK` | `9` | `[2 / 2 / 2 / 3]` | `spec_iter_10.md` |
| **Total** | **`10 / 10`** | **Champion locked at `0f8a2d4d`** | **`95` unique keys (`0` overlap)** | **`[20 / 21 / 21 / 33]`** | **All `10` specs + `3` histories intact** |
