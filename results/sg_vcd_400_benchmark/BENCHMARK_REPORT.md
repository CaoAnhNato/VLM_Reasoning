# ViTextVQA Benchmark Report

## 1. Executive Summary

- **Evaluation Date**: 2026-10-29 07:02:58 UTC
- **Model Backbone**: `5CD-AI/Vintern-1B-v3_5` (`InternViT-300M` + `Qwen2.5-0.5B-Instruct`)
- **Evaluation Method**: **SG_VCD**
- **Hardware Environment**: Local / Google Colab GPU
- **Total Evaluated Questions**: 1222 questions
- **Evaluation Status**: **100% COMPLETE (1222 / 1222 questions)** with zero crashes.
- **Total Benchmark Time**: 1013.5 s (16.89 min)

---

## 2. Benchmark Metrics Summary

### Stage 2: End-to-End VQA Accuracy Metrics
| Metric | Verified GT | Original GT | Target | Status |
|:---|:---:|:---:|:---:|:---:|
| **Substring Match / Inclusion (%)** | **71.82%** | - | > 80.0% | IN_PROGRESS |
| **Token F1 (%)** | **67.53%** | 66.82% | > 80.0% | IN_PROGRESS |
| **Relaxed Accuracy / ANLS (%)** | **63.61%** | - | > 80.0% | IN_PROGRESS |
| **Exact Match (%)** | **45.66%** | 44.52% | - | Strict matching |
| **VQA Consensus Accuracy (%)** | **45.28%** | 44.52% | - | Standard VQA metric |

### Stage 1: Spatial Region Localization Metrics
| Metric | Value | Target | Status |
|:---|:---:|:---:|:---:|
| **Mean GT Inclusion Recall (%)** | **83.34%** | >= 80.0% | EXCEEDED |
| **Multi-Box Text Coverage (%)** | **85.84%** | >= 80.0% | EXCEEDED |
| **Hit Rate @ Recall >= 0.5 (%)** | **85.60%** | - | - |
| **Hit Rate @ IoU >= 0.3 (%)** | **24.06%** | - | - |
| **Hit Rate @ IoU >= 0.5 (%)** | **9.57%** | - | - |
| **Mean IoU (%)** | **18.79%** | - | - |

### Computational Performance & Hardware Diagnostics
| Metric | Measurement |
|:---|:---:|
| **Average Latency** | **2.860 s / question** |
| **Inference Throughput** | **20.98 questions / min** |
| **Total Benchmark Time** | **1013.5 s (16.89 min)** |

---

## 3. Key Observations & Findings

1. **Multi-Worker Data Sharding**:
   - Evaluation sharded deterministically across parallel worker processes.
   - Clean atomic aggregation ensures zero duplicate or dropped question IDs.
2. **Zero-Heuristic Generalizability**:
   - Zero hardcoded stopwords, zero case-specific coordinate forcing, genuine algorithmic grounding.
