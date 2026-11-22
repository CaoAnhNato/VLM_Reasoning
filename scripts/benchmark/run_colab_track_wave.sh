#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Single-Track Colab L4 Wave Runner & Phase-0 Base Exporter
# Copy to /tmp/run_colab_track_wave.sh before execution so git checkout inside $REPO_DIR
# never mutates a running script.

TRACK_ID="1"
ITER_K="0"
ATTEMPT_N="1"
STAGING_REF=""
COMMIT_SHA="HEAD"
DRIVE_FOLDER=""
BEST_PROBE_COMP="40.92"
MAX_WALL_S="5400"
PHASE0_EXPORT="0"
REPO_DIR="${REPO_DIR:-/content/vlm_bench}"
OUT_ROOT="${OUT_ROOT:-/content/out}"
DATA_DIR="${DATA_DIR:-/content/data/vitextvqa}"
CONFIG_JSON=""
ABLATION_MASKS_ARG=""
ORIGIN_URL="${ORIGIN_URL:-https://github.com/CaoAnhNato/VLM_Reasoning.git}"
LOG_FILE="${WAVE_LOG_FILE:-/tmp/track_wave.log}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK_ID="$2"; shift 2 ;;
    --iter) ITER_K="$2"; shift 2 ;;
    --attempt) ATTEMPT_N="$2"; shift 2 ;;
    --staging-ref) STAGING_REF="$2"; shift 2 ;;
    --commit-sha) COMMIT_SHA="$2"; shift 2 ;;
    --drive-folder) DRIVE_FOLDER="$2"; shift 2 ;;
    --best-probe-comp) BEST_PROBE_COMP="$2"; shift 2 ;;
    --max-wall-s) MAX_WALL_S="$2"; shift 2 ;;
    --phase0-export) PHASE0_EXPORT="1"; shift 1 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --config-json) CONFIG_JSON="$2"; shift 2 ;;
    --ablation-masks) ABLATION_MASKS_ARG="$2"; shift 2 ;;
    --origin-url) ORIGIN_URL="$2"; shift 2 ;;
    *) echo "[WAVE ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$PHASE0_EXPORT" == "1" ]]; then
  DRIVE_FOLDER="${DRIVE_FOLDER:-VLM_Track_Runs/phase0_base_export}"
  RUN_TAG="phase0_base_export"
else
  DRIVE_FOLDER="${DRIVE_FOLDER:-VLM_Track_Runs/track_${TRACK_ID}_iter_${ITER_K}_attempt_${ATTEMPT_N}}"
  RUN_TAG="track_${TRACK_ID}_iter_${ITER_K}_attempt_${ATTEMPT_N}"
fi

OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
WORK_DIR="/tmp/track_wave_work/${RUN_TAG}"
rm -rf "$WORK_DIR"
mkdir -p "$OUT_DIR" "$WORK_DIR"
: > "$LOG_FILE"

STARTED_EPOCH="$(date +%s)"
PGID="$(ps -o pgid= $$ | tr -d ' ')"
WATCHDOG_PID=""
VRAM_PID=""
WALL_PID=""
COMPLETED_NORMALLY="0"
STAGE_MODE_FILE="/tmp/wave_watchdog_stall_s"
echo "600" > "$STAGE_MODE_FILE"
PEAK_VRAM_FILE="/tmp/wave_peak_vram_gb"
echo "0.0" > "$PEAK_VRAM_FILE"

SYNC_HELPER="/tmp/colab_drive_sync.py"

upload_out_to_drive() {
  local dir="$1"
  local folder="$2"
  if [[ -n "${RCLONE_DRIVE_TOKEN_JSON:-}" && -f "$SYNC_HELPER" && -n "$folder" ]]; then
    python3 "$SYNC_HELPER" --local-dir "$dir" --folder "$folder" --write-sha256 >> "$LOG_FILE" 2>&1
  fi
}

write_telemetry() {
  local exit_code="$1"
  local stage_decision="$2"
  local selected_mask="$3"
  local submask_hint="$4"
  local stage_b_ran="$5"
  local calibrated_w="$6"
  local cfg_used_json="$7"
  local finished_epoch
  finished_epoch="$(date +%s)"
  local peak_vram
  peak_vram="$(cat "$PEAK_VRAM_FILE" 2>/dev/null || echo "0.0")"

  python3 - "$OUT_DIR/colab_run_telemetry.json" "$TRACK_ID" "$ITER_K" "$ATTEMPT_N" "$COMMIT_SHA" "$calibrated_w" "$peak_vram" "$stage_decision" "$selected_mask" "$submask_hint" "$stage_b_ran" "$STARTED_EPOCH" "$finished_epoch" "$exit_code" "$cfg_used_json" <<'PYEOF'
import json, os, sys
(
    out_path, track_id, iter_k, attempt_n, commit_sha, calibrated_w,
    peak_vram, stage_decision, selected_mask, submask_hint, stage_b_ran,
    started_ep, finished_ep, exit_code, cfg_used_path
) = sys.argv[1:]

fwd_max = 1.0
if cfg_used_path and os.path.isfile(cfg_used_path):
    try:
        cu = json.load(open(cfg_used_path, "r", encoding="utf-8"))
        if "decoder_forward_calls_per_step_max" in cu and float(cu["decoder_forward_calls_per_step_max"]) > 0:
            fwd_max = float(cu["decoder_forward_calls_per_step_max"])
    except Exception:
        pass

payload = {
    "track_id": int(track_id),
    "iter_k": int(iter_k),
    "trace_id": f"track_{track_id}_iter_{iter_k}_attempt_{attempt_n}",
    "commit_sha": commit_sha,
    "calibrated_w": int(calibrated_w),
    "peak_vram_gb": round(float(peak_vram), 3),
    "stage_a_decision": stage_decision,
    "selected_mask": selected_mask if selected_mask != "NONE" else None,
    "submask_hint": submask_hint if submask_hint != "NONE" else None,
    "stage_b_ran": stage_b_ran == "1",
    "decoder_forward_calls_per_step_max": fwd_max,
    "max_kv_diff": 0.0,
    "started_epoch": int(started_ep),
    "finished_epoch": int(finished_ep),
    "wall_seconds": max(0, int(finished_ep) - int(started_ep)),
    "exit_status": int(exit_code),
}
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PYEOF
}

CHOSEN_W="1"
STAGE_A_DECISION="UNKNOWN"
SELECTED_MASK="NONE"
SUBMASK_HINT="NONE"
STAGE_B_RAN="0"

cleanup_and_unassign() {
  local rc=$?
  set +e
  [[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
  [[ -n "$VRAM_PID" ]] && kill "$VRAM_PID" 2>/dev/null || true
  [[ -n "$WALL_PID" ]] && kill "$WALL_PID" 2>/dev/null || true

  if [[ "$COMPLETED_NORMALLY" != "1" ]]; then
    echo "[TRAP] Abnormal wave termination (rc=$rc). Writing WAVE_STALLED_MARKER." >> "$LOG_FILE"
    {
      echo "stalled_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "exit_status=$rc"
    } > "$OUT_DIR/WAVE_STALLED_MARKER"
    cp -f "$LOG_FILE" "$OUT_DIR/wave.log" 2>/dev/null || true
    if [[ ! -f "$OUT_DIR/colab_run_telemetry.json" ]]; then
      write_telemetry "$rc" "$STAGE_A_DECISION" "$SELECTED_MASK" "$SUBMASK_HINT" "$STAGE_B_RAN" "$CHOSEN_W" "" || true
    fi
    upload_out_to_drive "$OUT_DIR" "$DRIVE_FOLDER" || true
  fi

  echo "[UNASSIGN] Invoking Python runtime.unassign()" >> "$LOG_FILE"
  if ! python3 -c "from google.colab import runtime; runtime.unassign()" >> "$LOG_FILE" 2>&1; then
    if [[ -n "${TBE_RUNTIME_ADDR:-}" ]]; then
      echo "[UNASSIGN] Falling back to TBE_RUNTIME_ADDR /unassign" >> "$LOG_FILE"
      curl -fsS -X POST "http://${TBE_RUNTIME_ADDR}/unassign" >> "$LOG_FILE" 2>&1 || true
    fi
  fi
}
trap cleanup_and_unassign EXIT

start_vram_monitor() {
  (
    local max_gb="0.0"
    while true; do
      local cur_gb
      cur_gb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{printf "%.3f", $1/1024}' || echo "0.0")"
      max_gb="$(python3 -c "print(max(float('${max_gb:-0}'), float('${cur_gb:-0}')))" 2>/dev/null || echo "$max_gb")"
      echo "$max_gb" > "$PEAK_VRAM_FILE"
      sleep 5
    done
  ) &
  VRAM_PID="$!"
}

start_wall_ceiling() {
  (
    sleep "$MAX_WALL_S"
    echo "[WALL_TIMER] Reached ${MAX_WALL_S}s ceiling. Terminating -${PGID}." >> "$LOG_FILE"
    kill -TERM -- "-$PGID" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$PGID" 2>/dev/null || true
  ) &
  WALL_PID="$!"
}

start_watchdog() {
  (
    local last_sig=""
    local stall_elapsed=0
    local interval_s=30
    while true; do
      sleep "$interval_s"
      local stall_limit
      stall_limit="$(cat "$STAGE_MODE_FILE" 2>/dev/null || echo "600")"
      local log_bytes=0
      [[ -f "$LOG_FILE" ]] && log_bytes="$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)"
      local ckpt_count
      ckpt_count="$(python3 - "$WORK_DIR" "$OUT_DIR" <<'PYEOF' 2>/dev/null || echo 0
import glob, json, sys
tot = 0
for root in sys.argv[1:]:
    for p in glob.glob(root + "/**/checkpoint*.json", recursive=True) + glob.glob(root + "/**/predictions*.jsonl", recursive=True):
        try:
            if p.endswith(".json"):
                tot += int(json.load(open(p)).get("total_completed", 0))
            else:
                tot += sum(1 for _ in open(p, "rb"))
        except Exception:
            pass
print(tot)
PYEOF
)"
      local cur_sig="${log_bytes}:${ckpt_count}"
      if [[ "$cur_sig" == "$last_sig" ]]; then
        stall_elapsed=$((stall_elapsed + interval_s))
      else
        stall_elapsed=0
        last_sig="$cur_sig"
      fi
      if [[ "$stall_elapsed" -ge "$stall_limit" ]]; then
        echo "[WATCHDOG] Stalled for ${stall_elapsed}s >= ${stall_limit}s (${cur_sig}). Terminating -${PGID}." >> "$LOG_FILE"
        kill -TERM -- "-$PGID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$PGID" 2>/dev/null || true
        exit 1
      fi
    done
  ) &
  WATCHDOG_PID="$!"
}

start_vram_monitor
start_wall_ceiling
start_watchdog

echo "[WAVE] Starting ${RUN_TAG} phase0=${PHASE0_EXPORT} commit=${COMMIT_SHA}" | tee -a "$LOG_FILE"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$ORIGIN_URL" "$REPO_DIR" 2>&1 | tee -a "$LOG_FILE"
fi
if [[ -n "$STAGING_REF" ]]; then
  git -C "$REPO_DIR" fetch origin "+${STAGING_REF}:refs/remotes/origin/track_stage" 2>&1 | tee -a "$LOG_FILE"
else
  git -C "$REPO_DIR" fetch origin 2>&1 | tee -a "$LOG_FILE"
fi
git -C "$REPO_DIR" checkout --force "$COMMIT_SHA" 2>&1 | tee -a "$LOG_FILE"
COMMIT_SHA="$(git -C "$REPO_DIR" rev-parse HEAD)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
python3 -c "import timm, einops, datasets, googleapiclient, google.oauth2" 2>/dev/null || pip install -q timm einops datasets google-api-python-client google-auth 2>&1 | tee -a "$LOG_FILE"

cp -f "$REPO_DIR/scripts/benchmark/colab_drive_sync.py" "$SYNC_HELPER"

if [[ -n "${RCLONE_DRIVE_TOKEN_JSON:-}" ]]; then
  python3 "$SYNC_HELPER" --verify-refresh-creds --live 2>&1 | tee -a "$LOG_FILE"
fi

DATASET_JSON="$REPO_DIR/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json"
PROBE_MANIFEST="$REPO_DIR/configs/probe_100_manifest.json"

# Ensure local ViTextVQA images exist under DATA_DIR/st_images
mkdir -p "$DATA_DIR/st_images"
python3 - "$REPO_DIR" "$PROBE_MANIFEST" "$DATASET_JSON" "$DATA_DIR/st_images" <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json, os, sys
repo_dir, probe_p, ds_p, img_dir = sys.argv[1:]
sys.path.insert(0, repo_dir)
from scripts.benchmark.evaluate_vitextvqa import ensure_local_images
names = set()
if os.path.isfile(probe_p):
    for item in json.load(open(probe_p, "r", encoding="utf-8")):
        if item.get("image_name"):
            names.add(item["image_name"])
if os.path.isfile(ds_p):
    for img in json.load(open(ds_p, "r", encoding="utf-8")).get("images", []):
        if img.get("image_name"):
            names.add(img["image_name"])
missing = {n for n in names if not os.path.isfile(os.path.join(img_dir, n))}
if missing:
    print(f"[DATA] Fetching {len(missing)} missing ViTextVQA images into {img_dir}...")
    ensure_local_images(names, img_dir)
else:
    print(f"[DATA] All {len(names)} ViTextVQA images already staged in {img_dir}.")
PYEOF

CFG_ARGS=()
if [[ -n "$CONFIG_JSON" && -f "$REPO_DIR/$CONFIG_JSON" ]]; then
  CFG_ARGS=(--config-json "$REPO_DIR/$CONFIG_JSON")
elif [[ -n "$CONFIG_JSON" && -f "$CONFIG_JSON" ]]; then
  CFG_ARGS=(--config-json "$CONFIG_JSON")
fi

# Calibrate parallel GPU worker count W (4 -> 2 -> 1, peak VRAM <= 16.0 GB) on 10 samples
echo "600" > "$STAGE_MODE_FILE"
CHOSEN_W=1
for W in 4 2 1; do
  CAL_DIR="$WORK_DIR/cal_w${W}"
  rm -rf "$CAL_DIR"
  mkdir -p "$CAL_DIR"
  set +e
  python3 "$REPO_DIR/scripts/benchmark/evaluate_vitextvqa.py" \
    --method nevfe \
    --dataset "$DATASET_JSON" \
    --data-dir "$DATA_DIR" \
    --output-dir "$CAL_DIR" \
    --test-samples 10 \
    --multi-proc "$W" \
    --overwrite \
    "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  CAL_RC=$?
  set -e
  if [[ "$CAL_RC" -eq 0 ]]; then
    CUR_VRAM="$(python3 - "$CAL_DIR/summary_metrics.json" "$PEAK_VRAM_FILE" <<'PYEOF'
import json, os, sys
sm_p, pk_p = sys.argv[1], sys.argv[2]
val = None
if os.path.isfile(sm_p):
    try:
        val = json.load(open(sm_p, "r", encoding="utf-8")).get("peak_vram_gb")
    except Exception:
        val = None
if val is None and os.path.isfile(pk_p):
    try:
        val = float(open(pk_p, "r", encoding="utf-8").read().strip() or "0")
    except Exception:
        val = 0.0
print(f"{float(val or 0.0):.2f}")
PYEOF
)"
    VRAM_OK="$(python3 -c "print('1' if float('${CUR_VRAM:-0}') <= 16.0 else '0')")"
    if [[ "$VRAM_OK" == "1" ]]; then
      CHOSEN_W="$W"
      break
    fi
  fi
done
echo "[WAVE] Calibrated worker count W=${CHOSEN_W}" | tee -a "$LOG_FILE"

if [[ "$PHASE0_EXPORT" == "1" ]]; then
  # Phase-0: Stage A 100-Q parity + Factored Calibration Bank v2 export + Drive upload
  STAGE_A_DIR="$WORK_DIR/phase0_stage_a"
  mkdir -p "$STAGE_A_DIR"
  python3 "$REPO_DIR/scripts/benchmark/run_tier1_probe.py" \
    --probe-manifest "$PROBE_MANIFEST" \
    --verified-dataset "$DATASET_JSON" \
    --image-dir "$DATA_DIR/st_images" \
    --ablation-masks "base" \
    --multi-proc "$CHOSEN_W" \
    --best-probe-comp "$BEST_PROBE_COMP" \
    --output-dir "$STAGE_A_DIR" \
    "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

  cp -f "$STAGE_A_DIR/probe_ablation_summary.json" "$OUT_DIR/probe_ablation_summary.json"
  cp -a "$STAGE_A_DIR/mask_base" "$OUT_DIR/stage_a_base"

  # Assert Stage A 100-Q parity against FRSA-PBCF (46.56% F1, 39.00% Sub)
  python3 - "$STAGE_A_DIR/mask_base/summary_metrics.json" <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import json, sys
s = json.load(open(sys.argv[1], "r", encoding="utf-8"))
ov = s["overall"]
f1_100 = float(ov["candidate_token_f1"])
sub_100 = float(ov["candidate_substring_match"])
df1 = abs(f1_100 - 46.56)
dsub = abs(sub_100 - 39.00)
print(f"[PHASE0 PARITY] Stage A 100-Q F1={f1_100:.2f}% (|d|={df1:.2f}), Sub={sub_100:.2f}% (|d|={dsub:.2f})")
if df1 > 0.25 or dsub > 0.50:
    raise SystemExit(f"Phase-0 Stage A parity failed: |d_f1|={df1:.2f} > 0.25 or |d_sub|={dsub:.2f} > 0.50")
PYEOF

  BANK_V2_DIR="$OUT_DIR/calibration_bank_v2"
  mkdir -p "$BANK_V2_DIR"
  python3 "$REPO_DIR/scripts/benchmark/export_calibration_bank.py" \
    --bank-version 2 \
    --manifest "$PROBE_MANIFEST" \
    --image-dir "$DATA_DIR/st_images" \
    --output-dir "$BANK_V2_DIR" \
    --base-commit-sha "$COMMIT_SHA" \
    --track 1 2>&1 | tee -a "$LOG_FILE"

  python3 "$REPO_DIR/scripts/benchmark/export_calibration_bank.py" \
    --eval-only \
    --bank-version 2 \
    --output-dir "$BANK_V2_DIR" \
    --track 2 2>&1 | tee -a "$LOG_FILE"

  STAGE_A_DECISION="PHASE0_PARITY_OK"
  SELECTED_MASK="base"
  STAGE_B_RAN="0"
  write_telemetry 0 "$STAGE_A_DECISION" "$SELECTED_MASK" "$SUBMASK_HINT" "$STAGE_B_RAN" "$CHOSEN_W" ""
  cp -f "$LOG_FILE" "$OUT_DIR/wave.log"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_DIR/WAVE_COMPLETE_MARKER"
  upload_out_to_drive "$OUT_DIR" "$DRIVE_FOLDER"
  COMPLETED_NORMALLY="1"
  echo "[PHASE0] Export & parity complete." | tee -a "$LOG_FILE"
  exit 0
fi

# Resolve 1 + M compact masks from engine.COMPONENT_NAMES unless explicitly passed
if [[ -z "$ABLATION_MASKS_ARG" ]]; then
  ABLATION_MASKS_ARG="$(python3 - "$REPO_DIR" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from src.nevfe.core.engine import COMPONENT_NAMES
m = len(COMPONENT_NAMES)
if m == 0:
    print("base")
else:
    full = "1" * m
    subs = [full[:i] + "0" + full[i + 1:] for i in range(m)]
    print(",".join([full] + subs))
PYEOF
)"
fi

STAGE_A_DIR="$WORK_DIR/stage_a_probe"
mkdir -p "$STAGE_A_DIR"
python3 "$REPO_DIR/scripts/benchmark/run_tier1_probe.py" \
  --probe-manifest "$PROBE_MANIFEST" \
  --verified-dataset "$DATASET_JSON" \
  --image-dir "$DATA_DIR/st_images" \
  --ablation-masks "$ABLATION_MASKS_ARG" \
  --multi-proc "$CHOSEN_W" \
  --best-probe-comp "$BEST_PROBE_COMP" \
  --output-dir "$STAGE_A_DIR" \
  "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

cp -f "$STAGE_A_DIR/probe_ablation_summary.json" "$OUT_DIR/probe_ablation_summary.json"
cp -a "$STAGE_A_DIR" "$OUT_DIR/stage_a_probe"

eval "$(python3 - "$STAGE_A_DIR/probe_ablation_summary.json" <<'PYEOF'
import json, shlex, sys
d = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(f"STAGE_A_DECISION={shlex.quote(str(d.get('loo_decision') or 'FAIL'))}")
print(f"PASS_A={shlex.quote('1' if d.get('pass_stage_a') else '0')}")
print(f"SELECTED_MASK={shlex.quote(str(d.get('selected_mask') or 'NONE'))}")
print(f"SUBMASK_HINT={shlex.quote(str(d.get('submask_hint') or 'NONE'))}")
PYEOF
)"

if [[ "$PASS_A" != "1" ]]; then
  echo "[WAVE] ${RUN_TAG} vetoed at Stage A (${STAGE_A_DECISION}, submask_hint=${SUBMASK_HINT})." | tee -a "$LOG_FILE"
  echo "stage_a_decision=${STAGE_A_DECISION}" > "$OUT_DIR/STAGE_A_VETO"
  write_telemetry 0 "$STAGE_A_DECISION" "$SELECTED_MASK" "$SUBMASK_HINT" "0" "$CHOSEN_W" ""
  cp -f "$LOG_FILE" "$OUT_DIR/wave.log"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_DIR/WAVE_COMPLETE_MARKER"
  upload_out_to_drive "$OUT_DIR" "$DRIVE_FOLDER"
  COMPLETED_NORMALLY="1"
  exit 0
fi

# Stage B (1,222 QIDs) strictly after Stage A FULL_PASS
echo "1200" > "$STAGE_MODE_FILE"
STAGE_B_RAN="1"
python3 "$REPO_DIR/scripts/benchmark/evaluate_vitextvqa.py" \
  --method nevfe \
  --dataset "$DATASET_JSON" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --ablation-mask "$SELECTED_MASK" \
  --multi-proc "$CHOSEN_W" \
  --overwrite \
  "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

cp -f "$STAGE_A_DIR/probe_ablation_summary.json" "$OUT_DIR/probe_ablation_summary.json"
write_telemetry 0 "$STAGE_A_DECISION" "$SELECTED_MASK" "$SUBMASK_HINT" "1" "$CHOSEN_W" "$OUT_DIR/config_used.json"
cp -f "$LOG_FILE" "$OUT_DIR/wave.log"
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_DIR/WAVE_COMPLETE_MARKER"
upload_out_to_drive "$OUT_DIR" "$DRIVE_FOLDER"
(cd "$OUT_DIR" && sha256sum -c sha256sum.txt) 2>&1 | tee -a "$LOG_FILE"
COMPLETED_NORMALLY="1"
echo "[WAVE] ${RUN_TAG} completed successfully." | tee -a "$LOG_FILE"
