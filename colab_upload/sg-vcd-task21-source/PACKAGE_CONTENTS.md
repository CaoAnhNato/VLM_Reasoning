# SG-VCD Task 21 Colab Gate B Source Package Manifest (`PACKAGE_CONTENTS.md`)

**Package Name:** `sg-vcd-task21-source`  
**Target Environment:** Google Colab (Tesla T4 GPU, CUDA 12.x / PyTorch 2.14+)  
**Authorization Context:** Task 21 Conditional Gate B Pilot (Project Owner authorized 2026-11-12)  
**Package Structure:** Standalone workspace root designed for direct extraction into `/content/sg-vcd-task21-source` or `/content/VLM_Reasoning`

---

## 1. Directory Tree & Layout

```text
sg-vcd-task21-source/
├── COLAB_SETUP.md
├── PACKAGE_CONTENTS.md
├── SHA256SUMS.txt
├── colab_setup_check.py
├── requirements-colab.txt
├── data/
│   └── vitextvqa/
│       └── manifest_integration_20qa.json
├── results/
│   └── swintextspotter_auto_annot/
│       └── answer_bboxes_400_verified.json
└── scripts/
    └── benchmark/
        ├── colab_gate_b_preflight.py
        ├── evaluate_vitextvqa.py
        ├── metrics.py
        ├── run_colab_t4_gate_b_pilot.py
        ├── cerv/
        │   ├── __init__.py
        │   └── pipeline.py
        └── sg_vcd/
            ├── __init__.py
            ├── adaptive_propagator.py
            ├── attention_extractor.py
            ├── bbox_selector.py
            ├── dynamic_cd.py
            ├── feature_cache.py
            ├── pipeline.py
            └── roi_cropper.py
```

---

## 2. File Inventory & Specification

### A. Execution Entry Points & Preflight

| Relative Path | Size (Bytes) | Role & Description | Colab Necessity |
|---|---|---|---|
| `scripts/benchmark/run_colab_t4_gate_b_pilot.py` | 11,520 | **Primary Gate B Pilot Runner**. Sequentially executes the approved 3 arms (`ctrl0`, `oracle_roi`, `sg_vcd`), generates consolidated summary JSON/markdown, tracks GPU memory and timings. | **Strictly Required** |
| `scripts/benchmark/colab_gate_b_preflight.py` | 19,368 | **Preflight Readiness Check**. Verifies Tesla T4 GPU, FP16/BF16 tensor math, model weights availability, dataset images, and write permissions before starting pilot. | **Strictly Required** |
| `scripts/benchmark/evaluate_vitextvqa.py` | 94,156 | **Core Evaluation Engine**. Implements model loading, baseline inference, oracle ROI evaluation, SG-VCD integration, P0 R-04 schema output, checkpoint resumption, and metrics recording. | **Strictly Required** |
| `scripts/benchmark/metrics.py` | 23,665 | **Evaluation Metrics**. ViTextVQA official evaluation protocol: exact match (EM), F1 score, normalized text processing, and dual-reporting denominator aggregation. | **Strictly Required** |

### B. SG-VCD Production Modules (`scripts/benchmark/sg_vcd/`)

| Relative Path | Size (Bytes) | Role & Description | Colab Necessity |
|---|---|---|---|
| `scripts/benchmark/sg_vcd/__init__.py` | 3,773 | Package exports for SG-VCD components and version metadata. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/pipeline.py` | 85,946 | **SG-VCD Stage 1 & Stage 2 Orchestrator**. Visual token filtering, contrastive logit adjustment, and generation control. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/dynamic_cd.py` | 16,181 | **Contrastive Decoding Core**. Approved Canonical R-03 signed-logit formula ($z_{\text{final}} = (1+\alpha_t) z_{\text{pos}} - \alpha_t z_{\text{neg}}$) with legacy ReLU-clipped fallback option. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/adaptive_propagator.py` | 67,448 | Dynamic alpha/beta adaptation, token relevance estimation, and propagation rules. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/attention_extractor.py` | 44,698 | Vision-language cross-attention map extraction and spatial relevance aggregation. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/bbox_selector.py` | 24,272 | Heuristic & attention-driven bounding box selection for saliency guidance. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/feature_cache.py` | 24,952 | KV-cache and vision feature retention for high-throughput inference without redundant forward passes. | **Strictly Required** |
| `scripts/benchmark/sg_vcd/roi_cropper.py` | 14,383 | High-resolution Region of Interest (ROI) extraction and preprocessing. | **Strictly Required** |

### C. CERV Baseline Modules (`scripts/benchmark/cerv/`)

| Relative Path | Size (Bytes) | Role & Description | Colab Necessity |
|---|---|---|---|
| `scripts/benchmark/cerv/__init__.py` | 645 | CERV module initialization. | **Strictly Required** |
| `scripts/benchmark/cerv/pipeline.py` | 57,498 | CERV contrastive visual baseline implementation for comparison. | **Strictly Required** |

### D. Frozen Data & Ground-Truth Annotations

| Relative Path | Size (Bytes) | Role & Description | Colab Necessity |
|---|---|---|---|
| `data/vitextvqa/manifest_integration_20qa.json` | 18,179 | **Frozen 20-QA Integration Manifest**. SHA256: `9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a`. Covers exactly 20 QA pairs across 6 distinct images (`22.jpg`, `30.jpg`, `68.jpg`, `76.jpg`, `111.jpg`, `126.jpg`). | **Strictly Required** |
| `results/swintextspotter_auto_annot/answer_bboxes_400_verified.json` | 3,078,648 | **Verified GT Bounding Box Snapshot**. Contains SwinTextSpotter verified OCR bounding boxes for 400 questions, fully covering all 20 manifest items required for the Oracle ROI Diagnostic arm. | **Strictly Required** |

### E. Colab Setup & Environment Configuration

| Relative Path | Size (Bytes) | Role & Description | Colab Necessity |
|---|---|---|---|
| `colab_setup_check.py` | 4,579 | **Colab Pre-run Helper & Image Symlinker**. Recursively locates extracted ViTextVQA images on Google Drive, validates hashes/sizes, creates symlinks in `data/vitextvqa/st_images/`, and validates Python package readiness. | **Setup Helper** |
| `requirements-colab.txt` | 519 | Minimal pip dependency manifest pinned to Colab environment (does not upgrade or touch preinstalled `torch` / `torchvision` / `cuda`). | **Setup Helper** |
| `COLAB_SETUP.md` | 3,911 | Complete step-by-step Colab setup and execution instructions for the Project Owner. | **Documentation** |
| `PACKAGE_CONTENTS.md` | 7,883 | Detailed manifest of package files, functions, and roles. | **Documentation** |
| `SHA256SUMS.txt` | 2,075 | Cryptographic SHA256 checksums for every file in this package. | **Verification** |

---

## 3. Manifest Integrity Verification

- **Frozen Manifest Path:** `data/vitextvqa/manifest_integration_20qa.json`
- **Exact File Size:** 18,179 bytes
- **Exact SHA256 Checksum:** `9e9dc05dd87c2fb2de254cc1022dad41d5e745f8070560ddd98c648fab5b269a`
- **Total QA Entries:** 20
- **Distinct Image Count:** 6
- **Required Image Basenames:**
  1. `22.jpg` (Archive: `Cao_Anh_001_100`)
  2. `30.jpg` (Archive: `Cao_Anh_001_100`)
  3. `68.jpg` (Archive: `Cao_Anh_001_100`)
  4. `76.jpg` (Archive: `Cao_Anh_001_100`)
  5. `111.jpg` (Archive: `Trieu_101_200`)
  6. `126.jpg` (Archive: `Trieu_101_200`)
- **GT Coverage:** 20 / 20 QA items confirmed present in `results/swintextspotter_auto_annot/answer_bboxes_400_verified.json`.

---

## 4. Execution Workflow on Google Colab

1. **Extract source package:**
   ```bash
   mkdir -p /content/sg-vcd-task21-source
   unzip -q /content/drive/MyDrive/Document\ AI/Data/sg-vcd-task21-source.zip -d /content/sg-vcd-task21-source
   cd /content/sg-vcd-task21-source
   ```

2. **Install minimal dependencies:**
   ```bash
   pip install -r requirements-colab.txt
   ```

3. **Verify environment and link dataset images from Google Drive:**
   ```bash
   python colab_setup_check.py --images-root "/content/drive/MyDrive/Document AI/Data" --link
   ```

4. **Run preflight diagnostics:**
   ```bash
   python -m scripts.benchmark.colab_gate_b_preflight
   ```

5. **Run 3-arm Gate B Pilot:**
   ```bash
   python -m scripts.benchmark.run_colab_t4_gate_b_pilot
   ```
