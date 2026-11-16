# COLAB_SETUP.md — SG-VCD Task 21 Gate B Pilot (Tesla T4)

Scope (Owner-approved, one run only): 20 QA, 3 arms (`ctrl_0`, `oracle_roi`, `sg_vcd`), frozen manifest, `canonical_signed` scoring.
FP16 is primary. No automatic fallback to FP32/INT8. BF16 is only the Owner-authorized exception (`--allow-bf16-exception`) if FP16 hits a stop condition; it never overrides hard resource gates. No 100/400-image runs, no extra arms.

## 1. Project root
Unzip so that the folder containing `scripts/`, `data/`, `results/` is the project root, e.g.:
```python
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/work && cd /content/work && unzip -q -o "/content/drive/MyDrive/Document AI/Data/sg-vcd-task21-source.zip"
PROJECT_ROOT = "/content/work/sg-vcd-task21-source"   # adjust if your ZIP unpacks one level differently
%cd $PROJECT_ROOT
```
No `__init__.py` files are required at `scripts/` or `scripts/benchmark/` (namespace packages, as in the repo).

## 2. Dependencies (do not touch the CUDA torch build)
Colab already has torch `2.14.1+cu130`. Do NOT `pip install torch`, and do NOT use the repo-root `requirements.txt`.
```bash
pip install -r requirements-colab.txt
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must stay +cu130, True, Tesla T4
```
If pip proposes replacing torch, abort and report. Model remote code needs `einops`, `timm`, `sentencepiece` (included).

## 3. Dataset paths (no re-upload)
The evaluator resolves manifest images from `data/vitextvqa/st_images/` (or `data/xanylabeling_workspace/<archive>/`). The 6 images are `22.jpg 30.jpg 68.jpg 76.jpg 111.jpg 126.jpg`. Link them from your already-extracted Drive data (read-only, searched recursively):
```bash
python colab_setup_check.py --images-root "/content/drive/MyDrive/Document AI/Data" --link
```
It verifies the manifest SHA256 (`9e9dc05d…b269a`) and that all 6 images resolve. **Run this before the pilot**: the evaluator silently skips samples whose image is missing. Oracle ROI boxes (`answer_region_bbox`) come from the frozen manifest; the GT snapshot `results/swintextspotter_auto_annot/answer_bboxes_400_verified.json` is shipped as the authoritative reference (not read when `--manifest` is used).

## 4. Model identifier
`5CD-AI/Vintern-1B-v3_5` (default; downloaded from Hugging Face at run time, no weights in this package). Optional token for rate limits: `export HF_TOKEN=...` (set in Colab secrets/env only; never put in files).

## 5. Output directory
Default `results/task21_t4_gate_b_pilot/` under the project root (override with `--output-dir`; use a Drive path to persist). Contains `preflight_report.json`, `smoke_check_1qa_sg_vcd/`, `arm_ctrl_0/`, `arm_oracle_roi/`, `arm_sg_vcd/`, `pilot_consolidated_summary.json`.

## 6. Commands
Preflight only (FP16, JSON; exit 0 = admitted):
```bash
python scripts/benchmark/colab_gate_b_preflight.py --precision fp16 --json
```
Primary pilot (preflight → 1-QA real smoke check → 20-QA × 3 arms; aborts without weights download if preflight fails, and stops if smoke check fails):
```bash
python scripts/benchmark/run_colab_t4_gate_b_pilot.py --precision fp16 --output-dir results/task21_t4_gate_b_pilot
```
Owner-authorized BF16 exception, ONLY after an FP16 stop condition (separate output dir; native BF16 stays reported as REJECT; hard gates still enforced):
```bash
python scripts/benchmark/run_colab_t4_gate_b_pilot.py --precision bf16 --allow-bf16-exception --output-dir results/task21_t4_gate_b_pilot_bf16_exc
```
Note: preflight requires host RAM ≥ 8 GB, physical free VRAM ≥ 6 GB, total VRAM ≥ 8 GB, disk ≥ 15 GB, CUDA available. Do not retry indefinitely; preserve logs/partial outputs and report the first failure.

## 7. Environment variables
None required. Optional: `HF_TOKEN` (see §4). The evaluator sets `KMP_DUPLICATE_LIB_OK=TRUE` itself.
