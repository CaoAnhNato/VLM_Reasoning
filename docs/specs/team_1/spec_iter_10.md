# Specification (`spec_iter_10.md`) — Spherical KL-Specificity Gated & Thumbnail-Terminal Active Foveation (`SKG-TTAF`)

> **Note**: Do NOT embed a `spec_sha256` field inside this file. The SHA-256 digest is computed externally over the UTF-8 LF-normalized bytes of this file (`sha256(bytes.replace(b"\r\n", b"\n"))`) and bound on line 2 of `src/nevfe/core/engine.py`.

## 1. Header
- **team_id**: `team_1`
- **direction_id**: `direction_3`
- **iteration**: `10`
- **spec_version**: `v1.0`
- **base_best_commit**: `0f8a2d4d2bedbb5654c514f1666d9f2748b00bc5`

## 2. Backbone Tensor Invariants
- **Vision Encoder (`InternViT-300M`)**: Unwarped native-RGB patch embeddings `vit_embeds` of shape `[N_glob, 256, 896]` (`bfloat16`), cached immutably per image in `self._cached_vit_embeds`. Under `dynamic_preprocess_vintern(..., use_thumbnail=True)`, when `N_glob == 1` the single tile `vit_embeds[0]` is the `448 x 448` global image view; when `N_glob > 1`, `vit_embeds[:N_glob - 1]` are the `B = N_glob - 1` spatial grid tiles (`g_x * g_y == B`) and `vit_embeds[-1]` is the terminal `448 x 448` global thumbnail view (`chen2024expanding`).
- **Transient Per-Question Active Sensing Contract (Domain 1 Pure Semantic Pivot)**:
  - Prunes `100%` of `0f8a2d4d`'s rejected RGB Color Structure Tensor (`optical_energy` / `CST`) branch and `100%` of `direction_2`'s pixel high-pass / `25%`-overlap square-seam perturbations (`0` pixel filtering, `0` Sobel/Hessian/CST gradient priors).
  - Retains `0f8a2d4d`'s proven `DynamicMultiTileROICropper` native-RGB cropping (`min_tile_dim = max(1, force_image_size // 4)`, `max_tiles = 3`, strict Nyquist super-resolution gate `M_crop > M_glob`, unblurred `PIL.Image.Resampling.BICUBIC` + `self.transform`) producing transient `extra_vit_embeds` of shape `[N_crop, 256, 896]` (`bfloat16`) or `None` without ever mutating `self._cached_vit_embeds`.
  - `NEVFEPipeline.run` continues returning `stage1_bbox = [0, 0, w, h]`.
- **Language Decoder (`Qwen2.5-0.5B-Instruct`)**: `24` transformer layers, `14` query heads, `2` KV heads (`7:1` GQA group ratio), hidden dimension `896`, dynamic vocabulary size `vocab_size` (never hardcoded in `src/nevfe/core/engine.py`).
- **Two-Stream Contract & Cross-Team Isolation**: Positive visual-conditioned sequence $\tilde{V}^+ = \mathrm{assemble\_foveated\_visual\_sequence}(V_{\text{glob}}, V_{\text{crop}}, \mathtt{comp\_c})$ (`[N_eff, 256, 896]`) and synchronized Mean-Field Visual Reference sequence $V^0 = \mathbf{1}_{N_{\text{eff}} P_{\text{tile}}} \bar{v}^\top$ computed over $\tilde{V}^+$ (`compute_mean_field_reference_embeddings(effective_vit)`) enter LLM Layer 0 with identical sequence length and `<img>...</img>` boundaries. Zero overlap with Team 2 post-`lm_head` multi-step trajectory rollout (`LP-BEIT`) or Team 3 intra-decoder hidden-state readout (`OG-DMFR`).

## 3. Open-Loop Calibration Bank Observation
- **30-Sample CPU Calibration Benchmark (`data/local_calibration_bank/`, `CUDA_VISIBLE_DEVICES=""`)**:
  - Guarded open-loop evaluation (`scripts/local/run_guarded.py`) across all `30` `is_calibration: true` samples (`CLASS_A: 9`, `CLASS_B: 10`, `CLASS_C: 4`, `CLASS_D: 6`, `CLASS_E: 1`; global tile counts `N_glob in {1: 9, 3: 10, 5: 11}`) achieved `finite_ratio = 1.0000` (`30/30`), `simplex_kl_mean = 0.390094 nats`, `top1_prior_shift_rate = 0.4000` (`12/30`), and total wall latency `wall_ms = 199.25 ms` (`< 500 ms` budget).
  - On spatial norm lattices derived from the `30` calibration tensors at default `gamma_param = 0.50`, the multiscale Poisson–Bregman scan gate (`Eq. 2`) abstained (`J^* <= tau_scan -> j_star = 0.0`) on `100%` (`7/7`) of diffuse / large-text `CLASS_D` (`0/6`) and `CLASS_E` (`0/1`) samples while passing localized `CLASS_A..CLASS_C` concentrations (`6/23`).
- **Synthetic Multi-Resolution Null vs. Fine-Stroke Calibration (`S^{895}`, `N in {256, 512, 1024, 1536}`)**:
  - **Stopword Hijacking (`comp_a`)**: With `1` localized `2x3` sign token + `7` diffuse distractor tokens, Shared-RMS Spherical KL-Specificity (`comp_a=1`) concentrated `89.55%` posterior mass on the true `2x3` target vs `65.57%` under unweighted `max_m` (`comp_a=0`, `+23.98 pp`), matching `iter_9` `"011"`'s Stage-A campaign record (`f1_100 = 49.35%`, `sub_100 = 40.0%`).
  - **Multiscale Scan Abstention (`comp_b`)**: Clustered null scenes without localized query alignment still yield finite-lattice scan maxima $J_{\text{null}}^*$ with medians `[0.1348, 0.0906, 0.0526, 0.0499]` nats across `16x16`, `16x32`, `32x32`, `32x48` — `50,000x` above `0f8a2d4d`'s scale-blind `1e-6` trigger (`~100%` forced cropping). At `gamma_param = 0.50`, $\tau_{\text{scan}}(N; 0.50) = \sqrt{(\ln N)/N} \in \{0.1472, 0.1104, 0.0823, 0.0691\}$ abstains on `53.7%–61.3%` of diffuse null scenes while passing `93.8%–100.0%` of weak `+0.12`-cosine `2x3` signs and `100.0%` of `+0.16`-cosine `2x2` (`CLASS_A <4px`) signs (`q10 = 1.55..2.12 nats`).

## 4. Numbered Governing Equations & Boundary Conditions

### 4.1 Unified Information-Directed Active Sensing Principle (`SKG-TTAF`)
Let $\Omega = \{0,\dots,H_g-1\} \times \{0,\dots,W_g-1\}$ ($N = H_g W_g$) denote the unwarped spatial patch lattice formed by reshaping the `B` spatial tiles (`B = 1` if `N_glob == 1` else `N_glob - 1`) of `vit_embeds[:B]` from `[g_y, g_x, p_s, p_s, D]` (`p_s = sqrt(P_tile)`) via `.permute(0, 2, 1, 3, 4).reshape(H_g, W_g, D)`, and let $Q \in \mathbb{R}^{M \times D}$ denote prompt-centered Layer-0 query token embeddings (`q_raw - mean(prompt_embeds)`). `SKG-TTAF` selects at most one high-zoom native-RGB subwindow and splices its transient tiles into the `InternVL` visual token stream via a single three-stage information-theoretic operator (`Eqs. 1–3`) paired with locked `IG-VFED` decoding (`Eq. 4`):

### 4.2 (Eq. 1) Spherical von Mises–Fisher KL-Specificity Spatial Posterior (`compute_semantic_specificity_posterior`)
Given `patch_embeds` $\in \mathbb{R}^{H_g \times W_g \times D}$ and optional `query_embeds` $\in \mathbb{R}^{M \times D}$ (with fixed numerical tolerances $\tau_v = 10^{-6}$, $\epsilon = 10^{-12}$, and uniform lattice prior $u_0(u) = 1 / \max(1, N)$):
1. **Rank, Empty, and IEEE-754 Sanitization Guards**:
   - Raise `ValueError` if `patch_embeds.dim() != 3`, or if `query_embeds is not None` and `query_embeds.dim() == 2` with `query_embeds.shape[1] != D`.
   - Return `u_0` (`shape = [H_g, W_g]`, `dtype = float32`, `device = patch_embeds.device`) immediately if `query_embeds is None`, `query_embeds.dim() != 2`, `M == 0`, `D == 0`, or `N == 0`.
   - Promote to `float32`, sanitize `NaN`/`±Inf` to `0.0`, and normalize global/row dynamic range against `±FLT_MAX` overflow prior to Euclidean norm evaluation:
     $$V_{\text{safe}} = \frac{\mathrm{nan\_to\_num}(\mathrm{float32}(V), 0)}{\max(\|V_{\text{raw}}\|_\infty, 1.0)} \in \mathbb{R}^{N \times D}, \qquad Q_{\text{safe}, m} = \frac{\mathrm{nan\_to\_num}(\mathrm{float32}(Q_m), 0)}{\max(\|Q_{\text{raw}, m}\|_\infty, 1.0)} \in \mathbb{R}^{D}.$$
2. **Mean-Centered Hyperspherical Projection (`banerjee2005clustering`)**:
   Center spatial patches $\Delta v_u = V_{\text{safe}, u} - \frac{1}{N}\sum_{u'=1}^N V_{\text{safe}, u'}$ and project onto $\mathbb{S}^{D-1}$ with dimension-scaled subnormal guard $\tau_D = \tau_v \sqrt{\max(1, D)}$:
   $$\hat{v}_u = \begin{cases} \dfrac{\Delta v_u}{\max(\|\Delta v_u\|_2, \tau_D)} & \|\Delta v_u\|_2 > \tau_D \\ \mathbf{0} & \text{otherwise} \end{cases}, \qquad \hat{e}_m = \begin{cases} \dfrac{Q_{\text{safe}, m}}{\max(\|Q_{\text{safe}, m}\|_2, \tau_v)} & \|Q_{\text{safe}, m}\|_2 > \tau_v \\ \mathbf{0} & \text{otherwise} \end{cases},$$
   forming directional cosine matrix $S = \hat{V} \hat{E}^\top \in [-1, 1]^{N \times M}$.
3. **Query Token Aggregation (`comp_a`; `yuksekgonul2023when`)**:
   - **If `comp_a == False` (LOO `(0, 1, 1)`)**: Unweighted bag-of-words maximum $c_u = \max_{1 \le m \le M} S_{u, m}$.
   - **If `comp_a == True` (`SKG-TTAF`)**: Column-center $\tilde{S}_{u,m} = S_{u,m} - \frac{1}{N}\sum_{u'=1}^N S_{u',m}$ and compute the **shared matrix RMS** $\sigma_S = \sqrt{\max\!\big(0, \frac{1}{N M}\sum_{u=1}^N \sum_{m=1}^M \tilde{S}_{u,m}^2\big)}$. If $\sigma_S \le \tau_v$, return $u_0$. Otherwise standardize $Z_{u,m} = \tilde{S}_{u,m} / \max(\sigma_S, \tau_v)$, form token-wise vMF spatial posteriors $q_m(u) = \mathrm{Softmax}_u(Z_{u,m})$, and compute each query token's spatial KL specificity from $u_0$:
     $$I_m = \max\!\left(0.0, \; \sum_{u=1}^N q_m(u)\Big(\ln\max(q_m(u), \epsilon) + \ln N\Big)\right), \qquad w_m = \begin{cases} \dfrac{I_m^2}{\sum_{m'=1}^M I_{m'}^2} & \sum_{m'=1}^M I_{m'}^2 > \epsilon \\ \dfrac{1}{M} & \text{otherwise} \end{cases}.$$
     Aggregate via shift-safe weighted log-sum-exp (`Soft-OR`) with $z_u^* = \max_{1 \le m \le M} Z_{u,m}$:
     $$c_u = z_u^* + \ln\max\!\left(\epsilon, \; \sum_{m=1}^M w_m \exp(Z_{u,m} - z_u^*)\right).$$
4. **Simplex Projection**:
   Let $\tilde{c}_u = c_u - \frac{1}{N}\sum_{u'=1}^N c_{u'}$ and $\sigma_c = \sqrt{\max\!\big(0, \frac{1}{N}\sum_{u=1}^N \tilde{c}_u^2\big)}$. If $\sigma_c \le \tau_v$, return $u_0$; otherwise return $P(u) = \mathrm{Softmax}_u\big(\tilde{c}_u / \max(\sigma_c, \tau_v)\big)$ reshaped to `[H_g, W_g]`.

### 4.3 (Eq. 2) Multiscale Poisson–Bregman Scan-Statistic Active Subwindow Gate (`select_active_foveal_subwindow`)
Given `posterior_2d` $\in \mathbb{R}^{H_g \times W_g}$, optional `gamma_param` $\in [0.0, 1.0]$ (defaulting to `AlgorithmConfig().gamma_param` when `None` or non-finite, clamped via $\gamma_{\text{eff}} = \min(1.0, \max(0.0, \gamma))$), `comp_b: bool = True`, $\epsilon_k = 10^{-6}$, and $\epsilon = 10^{-12}$:
1. **Sanitization & Degenerate Lattice Guards**:
   - Raise `ValueError` if `posterior_2d.dim() != 2`.
   - Sanitize $p_0 = \max(0, \mathrm{nan\_to\_num}(\mathrm{float32}(\text{posterior\_2d}), 0))$. If $N = H_g W_g == 0$, return `((0, 0, H_g, W_g), 0.0, 0.0, 0.0)`.
   - Rescale $p_1 = p_0 / \max(\max(p_0), 1.0)$ and normalize $P = p_1 / \sum p_1$ if $\sum p_1 > \epsilon$ else $u_0$. If $N \le 1$, return `((0, 0, H_g, W_g), 0.0, 1.0, 1.0)`.
2. **Exact $O(H_g^2 W_g^2)$ 2D Integral-Image Poisson–Bregman Scan**:
   Construct padded summed-area table $\mathrm{SAT} \in \mathbb{R}^{(H_g+1)\times(W_g+1)}$ and evaluate every rectangle $B = [i_1, i_2) \times [j_1, j_2)$ ($0 \le i_1 < i_2 \le H_g$, $0 \le j_1 < j_2 \le W_g$):
   $$P(B) = \mathrm{clip}(\mathrm{SAT}_{i_2, j_2} - \mathrm{SAT}_{i_1, j_2} - \mathrm{SAT}_{i_2, j_1} + \mathrm{SAT}_{i_1, j_1}, \; 0.0, \; 1.0), \qquad A(B) = \frac{(i_2 - i_1)(j_2 - j_1)}{N}.$$
   On the compact concentrated admissibility set $\mathcal{B}_{\text{adm}} = \{B : P(B) > A(B) + \epsilon_k \;\wedge\; \epsilon_k \le A(B) \le 0.5\}$, with $P_{\text{cl}} = \mathrm{clip}(P(B), \epsilon_k, 1.0)$ and $A_{\text{cl}} = \mathrm{clip}(A(B), \epsilon_k, 1.0)$:
   $$J_{\text{Pois}}(B) = \begin{cases} \max\!\left(0.0, \; \mathrm{nan\_to\_num}\!\left(P_{\text{cl}}\ln\dfrac{P_{\text{cl}}}{A_{\text{cl}}} - (P_{\text{cl}} - A_{\text{cl}}), \; 0.0\right)\right) & B \in \mathcal{B}_{\text{adm}} \\ 0.0 & \text{otherwise} \end{cases},$$
   with maximizer $B^* = \arg\max_B J_{\text{Pois}}(B)$ and peak information gain $J^* = J_{\text{Pois}}(B^*)$.
3. **Multiscale Scan-Statistic Abstention Gate (`comp_b`; `butucea2013detection`; `russo2016information`)**:
   Define the finite-lattice null concentration boundary:
   $$\tau_{\text{scan}}(N; \gamma_{\text{eff}}) = \begin{cases} \max\!\left(\epsilon_k, \; 2.0 \, \gamma_{\text{eff}} \sqrt{\dfrac{\ln N}{N}}\right) & \text{if } \mathtt{comp\_b} = \text{True} \\ \epsilon_k & \text{if } \mathtt{comp\_b} = \text{False} \end{cases}.$$
   - If $J^* \le \tau_{\text{scan}}(N; \gamma_{\text{eff}})$: **abstain** and return `((0, 0, H_g, W_g), 0.0, 1.0, 1.0)`.
   - If $J^* > \tau_{\text{scan}}(N; \gamma_{\text{eff}})$: return `(B^*, J^*, P(B^*), A(B^*))`.

### 4.4 (Eq. 3) Thumbnail-Terminal `InternVL` Pretrained Tile Splice (`assemble_foveated_visual_sequence`)
Given global `vit_embeds` $\in \mathbb{R}^{N_{\text{glob}} \times P_{\text{tile}} \times D}$ and optional transient crop features `extra_vit_embeds` $\in \mathbb{R}^{N_{\text{crop}} \times P_{\text{tile}} \times D}$ (`comp_c: bool = True`):
- Raise `ValueError` if `vit_embeds.dim() != 3`, or if `extra_vit_embeds is not None` and (`extra_vit_embeds.dim() != 3` or `extra_vit_embeds.shape[1:] != vit_embeds.shape[1:]`).
- If `extra_vit_embeds is None` or `N_crop == 0` or `N_glob == 0`: return `vit_embeds` unmodified (`0`-ULP bit-identity).
- Otherwise cast `crops = extra_vit_embeds.to(device=vit_embeds.device, dtype=vit_embeds.dtype)` without mutating `vit_embeds`, and splice (`chen2024expanding`; `chen2024fastv`):
  $$\tilde{V} = \begin{cases} \mathrm{cat}\big([V_{\text{glob}}, \; \mathrm{crops}], \;\mathrm{dim}=0\big) & \text{if } \mathtt{comp\_c} = \text{False} \\ \mathrm{cat}\big([\mathrm{crops}, \; V_{\text{glob}}], \;\mathrm{dim}=0\big) & \text{if } \mathtt{comp\_c} = \text{True} \;\wedge\; N_{\text{glob}} = 1 \\ \mathrm{cat}\big([V_{\text{glob}}[:-1], \; \mathrm{crops}, \; V_{\text{glob}}[-1:]], \;\mathrm{dim}=0\big) & \text{if } \mathtt{comp\_c} = \text{True} \;\wedge\; N_{\text{glob}} > 1 \end{cases}.$$
  Whenever `comp_c == True`, $\tilde{V}[-1]$ is strictly bit-identical to $V_{\text{glob}}[-1]$ (the `448 x 448` global view immediately preceding `</img>`).

### 4.5 (Eq. 4) Adapter Wiring & Locked `IG-VFED` Decoding (`NEVFEEngine` & `VinternNEVFEAdapter`)
- **`NEVFEEngine.foveate(patch_embeds, query_embeds=None)`**: Evaluates `posterior_2d = compute_semantic_specificity_posterior(patch_embeds, query_embeds, comp_a=self.comp_a)` (`Eq. 1`) followed by `box, j_star, _, _ = select_active_foveal_subwindow(posterior_2d, gamma_param=self.gamma_param, comp_b=self.comp_b)` (`Eq. 2`), returning `(box, j_star, posterior_2d)`. Export `compute_semantic_specificity_posterior`, `select_active_foveal_subwindow`, and `assemble_foveated_visual_sequence` in `src/nevfe/core/__init__.py`.
- **`VinternNEVFEAdapter` Pure-Semantic Wiring**:
  1. Replace `_extract_lattice_and_energy` with pure `_extract_lattice(image, vit_embeds) -> Tuple[torch.Tensor, int, int]` returning `(patch_embeds, img_w, img_h)` with `img_w, img_h = image.size` (`0` RGB `CST` gradient computation).
  2. In `extract_foveated_roi_crops(image, vit_embeds, question)`: compute `patch_embeds, img_w, img_h = self._extract_lattice(image, vit_embeds)` and `query_embeds = self._extract_query_embeds(question, vit_embeds.device)`; call `box, j_star, _ = self.engine.foveate(patch_embeds, query_embeds=query_embeds)`; if `j_star <= 1e-6`, return `None`; otherwise project `box = (i1, j1, i2, j2)` over `(h_g, w_g) = patch_embeds.shape[:2]` to `(x1, y1, x2, y2)` and invoke `0f8a2d4d`'s unchanged `encode_transient_roi_crops(image, [(x1, y1, x2, y2)], max_tiles=3, min_magnification=m_glob)`.
  3. In both `build_multimodal_embeds` and `build_mfvr_embeds`: construct `effective_vit = assemble_foveated_visual_sequence(vit_embeds, extra_vit_embeds, comp_c=self.engine.comp_c)` (`Eq. 3`) so `p_t^+` and `p_t^0` remain length- and boundary-synchronized.
- **Locked Two-Stream `IG-VFED` Decoding (`t = 1` vs `t > 1`)**:
  `nevfe_step_logits` and `NEVFEEngine.decode_step` remain identical to `0f8a2d4d` ($\alpha_t = \lambda\sqrt{1 - \mathrm{BC}_t^2} + \mathbb{I}[t=1 \wedge \mathtt{comp\_c}]\lambda\mathrm{BC}_t$, $\beta_t = \beta\,\mathrm{BC}_t$, `steered_logits = where(p_plus >= beta_t * max_p_plus, lp + alpha_t * (lp - lz), -1e9)`), with `VinternNEVFEAdapter.generate` invoking `nevfe_step_logits(..., step_idx=step, comp_a=True, comp_b=True, comp_c=True)` locked across all steps $t \ge 1$ so `ablation_mask` exclusively ablates the three Domain-1 `SKG-TTAF` mechanisms (`Eqs. 1–3`) while `NEVFEEngine.decode_step` preserves exact `<= 1e-7` parity across all 8 masks.
- **AST & Literal Discipline**: Unpack `ablation_mask` via 3-tuple unpacking (`ca_cfg, cb_cfg, cc_cfg = base.ablation_mask`), use only `ALLOWED_NUMERIC_LITERALS = {0, 0.0, 0.5, 1, 1.0, -1, 2, 2.0, 1e-6, 1e-8, 1e-12, -1e9}` outside `AlgorithmConfig`, and keep `src/nevfe/core/engine.py` at `~200 <= 250` non-docstring AST statement lines.

## 5. `AlgorithmConfig` Parameter Table (`<= 3` Total Tunable Parameters)
| Field Name | Type | Valid Domain | Default | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `lambda_param` | `float` | `[0.0, 1.5]` | `0.50` | Locked Fisher-Rao geodesic visual contrast gain (`IG-VFED` `Eq. 4`) |
| `beta_param` | `float` | `[0.001, 5.0]` | `0.05` | Locked geodesic candidate plausibility floor (`IG-VFED` `Eq. 4`) |
| `gamma_param` | `float` | `[0.0, 1.0]` | `0.50` | Multiscale Poisson–Bregman scan-statistic abstention gain $\tau_{\text{scan}}(N; \gamma_{\text{eff}}) = \max(\epsilon_k, 2.0\,\gamma_{\text{eff}}\sqrt{(\ln N)/N})$ (`Eq. 2`) |

*(Count excludes `ablation_mask: Tuple[int, ...]`. Total tunable fields in `AlgorithmConfig` = `3 <= 3`.)*

## 6. Full + Leave-One-Out Ablation Mask Table
| Mask (`ablation_mask`) | Active Components | Hypothesis Tested |
| :--- | :--- | :--- |
| `(1, 1, 1)` | Full `SKG-TTAF` (`comp_a=1, comp_b=1, comp_c=1`) | Combined Shared-RMS KL-specificity `Soft-OR` (`Eq. 1`) + multiscale scan abstention gate (`Eq. 2`) + thumbnail-terminal `InternVL` tile splice (`Eq. 3`) |
| `(0, 1, 1)` | Leave-One-Out (`comp_a=0`) | Isolates `comp_a`: reverts query aggregation to unweighted `max_m` (`0f8a2d4d` lexical baseline without `CST`) while keeping multiscale scan abstention and thumbnail-terminal splicing |
| `(1, 0, 1)` | Leave-One-Out (`comp_b=0`) | Isolates `comp_b`: reverts scan abstention threshold from $\tau_{\text{scan}}(N; \gamma_{\text{eff}})$ to scale-blind $\epsilon_k = 10^{-6}$ while keeping KL-specificity `Soft-OR` and thumbnail-terminal splicing |
| `(1, 1, 0)` | Leave-One-Out (`comp_c=0`) | Isolates `comp_c`: reverts tile ordering from thumbnail-terminal splice to post-thumbnail append `[vit_embeds, crops]` while keeping KL-specificity `Soft-OR` and multiscale scan abstention |

## 7. Structural Independence Matrix
| BibTeX Key | Bucket [A/B/C/D] | Extracted Principle | Structural Difference vs. Prior SOTA & Other Teams |
| :--- | :--- | :--- | :--- |
| `mathew2022infographicvqa` | `A` | Dense visual-textual layouts and numerical span extraction where non-semantic visual cues mislocalize onto decorative layout borders | Grounds pruning `0f8a2d4d`'s RGB `optical_energy` (`CST`) prior in favor of pure query-conditioned semantic active sensing |
| `chen2024expanding` | `A` | `InternVL` dynamic high-resolution tiling invariant placing localized spatial tiles first and the `448 x 448` global thumbnail last right before `</img>` | Directly motivates `Eq. 3` thumbnail-terminal splicing (`[vit_embeds[:-1], crops, vit_embeds[-1:]]`) with zero retraining |
| `cha2024honeybee` | `B` | Locality-preserving visual projection motivation for fine spatial comprehension | Foil only (never chained): Honeybee replaces the frozen MLP projector with trainable C/D-Abstractors; `SKG-TTAF` is 100% zero-training on frozen `Vintern-1B-v3_5` |
| `mckinzie2024mm1` | `B` | Resolution scaling dominates MLLM OCR/VQA performance over connector architecture | Foil only (never chained): MM1 scales resolution unconditionally during multi-billion-token pre-training; `SKG-TTAF` performs selective query-conditioned test-time zoom with null-scan abstention |
| `yuksekgonul2023when` | `C` | Contrastive/retrieval VLMs behave like bags-of-words where diffuse relational/stopword tokens match irrespective of composition | Mechanistically grounds replacing unweighted `max_m` with squared spatial KL-specificity token weights $w_m \propto I_m^2$ (`Eq. 1`) |
| `chen2024fastv` | `C` | Causal LVLM decoders aggregate visual context onto late-sequence anchor tokens in shallow layers | Mechanistically grounds preserving the terminal `448 x 448` global thumbnail at `effective_vit[-1]` right before `</img>` (`Eq. 3`) |
| `banerjee2005clustering` | `D` | Maximum-entropy von Mises–Fisher directional density on $\mathbb{S}^{D-1}$ (*JMLR*, ISSN `1532-4435`, Q1) | Mathematical primitive for mean-centered hyperspherical cosine projection and shared-RMS concentration $Z_{u,m} = \tilde{S}_{u,m}/\sigma_S$ (`Eq. 1`); zero overlap with Team 3 decoder hidden-state routing |
| `butucea2013detection` | `D` | Minimax scan-statistic detection boundary over sparse submatrices of noisy lattices (*Bernoulli*, ISSN `1350-7265`, Q1) | Mathematical primitive for multiscale finite-lattice scan threshold $\tau_{\text{scan}}(N; \gamma_{\text{eff}}) = \max(\epsilon_k, 2.0\,\gamma_{\text{eff}}\sqrt{(\ln N)/N})$ (`Eq. 2`) |
| `russo2016information` | `D` | Information-theoretic value of active observation acquisition under posterior uncertainty (*JMLR*, ISSN `1532-4435`, Q1) | Mathematical primitive for selective active native-RGB sensing that abstains (`0`-ULP global `vit_embeds`) when subwindow KL divergence fails to exceed null uncertainty; zero overlap with Team 2 post-`lm_head` Bellman rollout |

## 8. Mechanical Revision Changelog (`v1.0 -> v1.3`)
- `v1.0`: Initial `direction_3` (`iter_10`) pure semantic active native-RGB sensing specification (`SKG-TTAF`) derived from `0f8a2d4d2bedbb5654c514f1666d9f2748b00bc5` post-`Action C` rollback.
