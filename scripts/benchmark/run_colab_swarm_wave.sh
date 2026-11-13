#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Colab L4 Parallel Swarm Wave Runner
# Copy to /tmp/run_colab_swarm_wave.sh before launching so git checkout inside $REPO_DIR never mutates a running script.
# NOTE: No runtime.unassign() calls or traps belong inside this script; colab-runner manages Colab runtime teardown.

WAVE_MANIFEST=""
REPO_DIR="${REPO_DIR:-/content/vlm_bench}"
OUT_ROOT="${OUT_ROOT:-/content/out}"
DRIVE_FOLDER_PREFIX="${DRIVE_FOLDER_PREFIX:-VLM_Swarm_Runs}"
DRIVE_RESULTS="${DRIVE_RESULTS:-}"
DATA_DIR="${DATA_DIR:-/content/data/vitextvqa}"
DATASET_JSON=""
PROBE_MANIFEST=""
LOG_FILE="${WAVE_LOG_FILE:-/tmp/wave.log}"
WATCHDOG_INTERVAL_S="${WATCHDOG_INTERVAL_S:-60}"
WATCHDOG_STALL_S="${WATCHDOG_STALL_S:-1800}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wave-manifest) WAVE_MANIFEST="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --drive-folder-prefix) DRIVE_FOLDER_PREFIX="$2"; shift 2 ;;
    --drive-results) DRIVE_RESULTS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --dataset) DATASET_JSON="$2"; shift 2 ;;
    --probe-manifest) PROBE_MANIFEST="$2"; shift 2 ;;
    --smoke-only) SMOKE_ONLY="1"; shift 1 ;;
    *) echo "[WAVE ERROR] Unknown argument: $1" >&2; exit 2 ;;
  esac
done

DATASET_JSON="${DATASET_JSON:-$REPO_DIR/results/swintextspotter_auto_annot/answer_bboxes_400_verified.json}"
PROBE_MANIFEST="${PROBE_MANIFEST:-$REPO_DIR/configs/probe_100_manifest.json}"

if [[ -z "$WAVE_MANIFEST" || ! -f "$WAVE_MANIFEST" ]]; then
  echo "[WAVE ERROR] --wave-manifest missing or not found: $WAVE_MANIFEST" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
[[ -n "$DRIVE_RESULTS" ]] && mkdir -p "$DRIVE_RESULTS"
rm -f /tmp/WAVE_COMPLETE_MARKER /tmp/WAVE_STALLED_MARKER
touch "$LOG_FILE"
PGID="$(ps -o pgid= $$ | tr -d ' ')"
WATCHDOG_PID=""
SYNC_HELPER="/tmp/colab_drive_sync.py"
if [[ -f "$REPO_DIR/scripts/benchmark/colab_drive_sync.py" ]]; then
  cp -f "$REPO_DIR/scripts/benchmark/colab_drive_sync.py" "$SYNC_HELPER"
fi

sync_to_drive() {
  local src="$1"
  local remote_sub="$2"
  if [[ -n "${RCLONE_DRIVE_TOKEN_JSON:-}" && -f "$SYNC_HELPER" ]]; then
    python3 "$SYNC_HELPER" --local-dir "$src" --folder "${DRIVE_FOLDER_PREFIX}/${remote_sub}" --write-sha256 2>&1 | tee -a "$LOG_FILE"
  fi
  if [[ -n "$DRIVE_RESULTS" ]]; then
    local dst="$DRIVE_RESULTS/$remote_sub"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --inplace "$src/" "$dst/"
    else
      cp -a "$src/." "$dst/"
    fi
  fi
}

start_watchdog() {
  local work_root="$1"
  (
    local last_sig=""
    local stall_elapsed=0
    while true; do
      sleep "$WATCHDOG_INTERVAL_S"
      local log_bytes=0
      [[ -f "$LOG_FILE" ]] && log_bytes="$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)"
      local ckpt_count
      ckpt_count="$(python3 -c '
import glob, json, sys
tot = 0
for p in glob.glob(sys.argv[1] + "/**/checkpoint*.json", recursive=True) + glob.glob(sys.argv[2] + "/**/checkpoint*.json", recursive=True):
    try:
        tot += int(json.load(open(p)).get("total_completed", 0))
    except Exception:
        pass
print(tot)
' "$work_root" "$OUT_ROOT" 2>/dev/null || echo 0)"
      local cur_sig="${log_bytes}:${ckpt_count}"
      if [[ "$cur_sig" == "$last_sig" ]]; then
        stall_elapsed=$((stall_elapsed + WATCHDOG_INTERVAL_S))
      else
        stall_elapsed=0
        last_sig="$cur_sig"
      fi
      if [[ "$stall_elapsed" -ge "$WATCHDOG_STALL_S" ]]; then
        echo "[WATCHDOG] Stalled for ${stall_elapsed}s (${cur_sig}). Terminating process group -${PGID}." >> "$LOG_FILE"
        mkdir -p "$OUT_ROOT/_stalled_partial"
        echo "stalled_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_ROOT/_stalled_partial/WAVE_STALLED_MARKER"
        echo "stalled_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/WAVE_STALLED_MARKER
        cp -f "$LOG_FILE" "$OUT_ROOT/_stalled_partial/wave.log" 2>/dev/null || true
        sync_to_drive "$OUT_ROOT/_stalled_partial" "_stalled_partial" || true
        kill -TERM -- "-$PGID" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-$PGID" 2>/dev/null || true
        exit 1
      fi
    done
  ) &
  WATCHDOG_PID="$!"
}

cleanup_watchdog() {
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap cleanup_watchdog EXIT

WORK_ROOT="/tmp/swarm_wave_work"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
start_watchdog "$WORK_ROOT"

N_ITEMS="$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$WAVE_MANIFEST")"
echo "[WAVE] Starting Colab L4 swarm wave with $N_ITEMS queue items (PGID=$PGID)" | tee -a "$LOG_FILE"

for idx in $(seq 0 $((N_ITEMS - 1))); do
  eval "$(python3 -c '
import json, shlex, sys
raw = json.load(open(sys.argv[1]))
items = raw["queue"] if isinstance(raw, dict) and "queue" in raw else raw
item = items[int(sys.argv[2])]
for k in ("team_id", "iter_k", "impl_sha", "config_json", "ablation_masks", "baseline_probe_json"):
    val = "" if item.get(k) is None else str(item.get(k))
    print(f"ITEM_{k.upper()}={shlex.quote(val)}")
' "$WAVE_MANIFEST" "$idx")"

  NORM_TEAM="${ITEM_TEAM_ID#team_}"
  NORM_TEAM="${NORM_TEAM#t}"
  RUN_TAG="team_${NORM_TEAM}_iter_${ITEM_ITER_K}"
  RUN_DIR="$WORK_ROOT/$RUN_TAG"
  SMOKE_DIR="$RUN_DIR/smoke_3"
  STAGE_A_DIR="$RUN_DIR/stage_a_probe"
  STAGE_B_DIR="$OUT_ROOT/$RUN_TAG"
  rm -rf "$STAGE_B_DIR"
  mkdir -p "$SMOKE_DIR" "$STAGE_A_DIR" "$STAGE_B_DIR"

  echo "[WAVE] === Queue [$idx] ${RUN_TAG} sha=${ITEM_IMPL_SHA} ===" | tee -a "$LOG_FILE"
  if ! git -C "$REPO_DIR" cat-file -e "${ITEM_IMPL_SHA}^{commit}" 2>/dev/null; then
    git -C "$REPO_DIR" fetch origin 2>&1 | tee -a "$LOG_FILE" || true
  fi
  git -C "$REPO_DIR" checkout -f "$ITEM_IMPL_SHA" 2>&1 | tee -a "$LOG_FILE"

  CFG_ARGS=()
  if [[ -n "$ITEM_CONFIG_JSON" && -f "$REPO_DIR/$ITEM_CONFIG_JSON" ]]; then
    CFG_ARGS=(--config-json "$REPO_DIR/$ITEM_CONFIG_JSON")
  elif [[ -n "$ITEM_CONFIG_JSON" && -f "$ITEM_CONFIG_JSON" ]]; then
    CFG_ARGS=(--config-json "$ITEM_CONFIG_JSON")
  fi

  BASE_PROBE_ARGS=()
  if [[ -n "$ITEM_BASELINE_PROBE_JSON" && -f "$REPO_DIR/$ITEM_BASELINE_PROBE_JSON" ]]; then
    BASE_PROBE_ARGS=(--baseline-probe-json "$REPO_DIR/$ITEM_BASELINE_PROBE_JSON")
  fi

  # 1. Live 3-sample smoke check before Stage A
  python3 "$REPO_DIR/scripts/benchmark/evaluate_vitextvqa.py" \
    --method nevfe \
    --dataset "$DATASET_JSON" \
    --data-dir "$DATA_DIR" \
    --output-dir "$SMOKE_DIR" \
    --test \
    --overwrite \
    "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

  # 2. Stage A 100-Q probe across ablation masks (1x model load per worker)
  MASKS="${ITEM_ABLATION_MASKS:-111,011,101,110}"
  python3 "$REPO_DIR/scripts/benchmark/run_tier1_probe.py" \
    --probe-manifest "$PROBE_MANIFEST" \
    --verified-dataset "$DATASET_JSON" \
    --image-dir "$DATA_DIR/st_images" \
    --ablation-masks "$MASKS" \
    --multi-proc 4 \
    --output-dir "$STAGE_A_DIR" \
    "${CFG_ARGS[@]}" \
    "${BASE_PROBE_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

  cp -f "$STAGE_A_DIR/probe_ablation_summary.json" "$STAGE_B_DIR/probe_ablation_summary.json"
  cp -a "$STAGE_A_DIR" "$STAGE_B_DIR/stage_a_probe"

  PASS_A="$(python3 -c 'import json, sys; print("1" if json.load(open(sys.argv[1])).get("pass_stage_a") else "0")' "$STAGE_A_DIR/probe_ablation_summary.json")"
  BEST_MASK="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("best_mask") or "111")' "$STAGE_A_DIR/probe_ablation_summary.json")"

  if [[ "$PASS_A" != "1" ]]; then
    echo "[WAVE] ${RUN_TAG} VETOED at Stage A." | tee -a "$LOG_FILE"
    echo "stage_a_veto=1" > "$STAGE_B_DIR/STAGE_A_VETO"
    echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STAGE_B_DIR/WAVE_COMPLETE_MARKER"
    sync_to_drive "$STAGE_B_DIR" "$RUN_TAG"
    continue
  fi

  # 3. 20-sample multi-worker VRAM calibration loop (W=4 -> 2 -> 1, peak <= 16.0 GB)
  CHOSEN_W=1
  for W in 4 2 1; do
    PRETEST_DIR="$RUN_DIR/pretest_w${W}"
    mkdir -p "$PRETEST_DIR"
    set +e
    python3 "$REPO_DIR/scripts/benchmark/evaluate_vitextvqa.py" \
      --method nevfe \
      --dataset "$DATASET_JSON" \
      --data-dir "$DATA_DIR" \
      --output-dir "$PRETEST_DIR" \
      --ablation-mask "$BEST_MASK" \
      --test-samples 20 \
      --multi-proc "$W" \
      --overwrite \
      "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
    RC=$?
    set -e
    if [[ "$RC" -eq 0 ]]; then
      VRAM_GB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{printf "%.2f", $1/1024}' || echo "0.0")"
      VRAM_OK="$(python3 -c "print('1' if float('${VRAM_GB:-0}') <= 16.0 else '0')")"
      if [[ "$VRAM_OK" == "1" ]]; then
        CHOSEN_W="$W"
        break
      fi
    fi
  done

  if [[ "$SMOKE_ONLY" == "1" ]]; then
    echo "[WAVE] --smoke-only active; copying pretest W=${CHOSEN_W} outputs as Stage B smoke." | tee -a "$LOG_FILE"
    cp -a "$RUN_DIR/pretest_w${CHOSEN_W}/." "$STAGE_B_DIR/"
  else
    python3 "$REPO_DIR/scripts/benchmark/evaluate_vitextvqa.py" \
      --method nevfe \
      --dataset "$DATASET_JSON" \
      --data-dir "$DATA_DIR" \
      --output-dir "$STAGE_B_DIR" \
      --ablation-mask "$BEST_MASK" \
      --multi-proc "$CHOSEN_W" \
      --overwrite \
      "${CFG_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  fi

  cp -f "$STAGE_A_DIR/probe_ablation_summary.json" "$STAGE_B_DIR/probe_ablation_summary.json"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STAGE_B_DIR/WAVE_COMPLETE_MARKER"
  (cd "$STAGE_B_DIR" && sha256sum -c sha256sum.txt) 2>&1 | tee -a "$LOG_FILE"
  sync_to_drive "$STAGE_B_DIR" "$RUN_TAG"
done

echo "wave_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /tmp/WAVE_COMPLETE_MARKER
[[ -n "$DRIVE_RESULTS" ]] && echo "wave_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$DRIVE_RESULTS/WAVE_COMPLETE_MARKER"
echo "[WAVE] All $N_ITEMS items completed successfully." | tee -a "$LOG_FILE"
