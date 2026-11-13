#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-Only Multi-Signal Zero-Progress Liveness Inspector (scripts/audit/check_swarm_liveness.py).

Classifies each 'working' worker deterministically:
- ACTIVE_PROGRESS: any signal advances (delta > 0), resetting zero_delta_since
- MONITORING: delta == 0 for < 1800s
- LEVEL_1_SUBPROCESS_HANG: delta == 0 for >= 1800s with a live child_pid
- LEVEL_2_SESSION_FREEZE: delta == 0 for >= 3600s without a live child_pid
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

LEVEL_1_THRESHOLD_S = 1800.0
LEVEL_2_THRESHOLD_S = 3600.0
DEFAULT_HEARTBEAT_DIR = os.path.join(
    os.path.expanduser("~"), ".gemini", "vlm_swarm_state", "heartbeats"
)


def _read_transcript_signal(transcript_path: Optional[str]) -> Tuple[int, int, Optional[int]]:
    if not transcript_path or not os.path.isfile(transcript_path):
        return 0, 0, None
    st_size = os.path.getsize(transcript_path)
    max_step = 0
    pending_cmd_step: Optional[int] = None
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            step = int(rec.get("step_index", 0))
            if step > max_step:
                max_step = step
            if rec.get("tool_calls"):
                for tc in rec["tool_calls"]:
                    if tc.get("name") in ("run_command", "ctx_shell") and rec.get("status") != "DONE":
                        pending_cmd_step = step
    return max_step, st_size, pending_cmd_step


def _read_heartbeat_signal(
    worker_name: str,
    heartbeat_dir: str,
) -> Tuple[int, Optional[int]]:
    if not os.path.isdir(heartbeat_dir):
        return 0, None
    max_bytes = 0
    live_child_pid: Optional[int] = None
    for fname in sorted(os.listdir(heartbeat_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(heartbeat_dir, fname)
        data = json.load(open(fpath, "r", encoding="utf-8"))
        if data.get("worker_id") != worker_name:
            continue
        b = int(data.get("bytes_emitted", 0))
        if b > max_bytes:
            max_bytes = b
        cpid = data.get("child_pid")
        if isinstance(cpid, int) and cpid > 0 and psutil.pid_exists(cpid):
            live_child_pid = cpid
    return max_bytes, live_child_pid


def _read_worktree_mtime(worktree_path: Optional[str]) -> float:
    if not worktree_path or not os.path.isdir(worktree_path):
        return 0.0
    max_m = os.path.getmtime(worktree_path)
    for root, dirs, files in os.walk(worktree_path):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for fname in files:
            max_m = max(max_m, os.path.getmtime(os.path.join(root, fname)))
    return round(max_m, 3)


def _read_git_branch_sha(worktree_path: Optional[str], branch: Optional[str]) -> str:
    if not worktree_path or not branch or not os.path.isdir(worktree_path):
        return ""
    proc = subprocess.run(
        ["git", "-C", worktree_path, "rev-parse", "--verify", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _read_colab_qids(checkpoint_path: Optional[str]) -> int:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0
    data = json.load(open(checkpoint_path, "r", encoding="utf-8"))
    return int(data.get("total_completed", len(data.get("completed_sample_keys", []))))


def evaluate_liveness(
    ledger: Dict[str, Any],
    snapshot: Dict[str, Any],
    heartbeat_dir: str = DEFAULT_HEARTBEAT_DIR,
    now_epoch: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    now = float(now_epoch if now_epoch is not None else time.time())
    prev_workers = snapshot.get("workers", {})
    next_workers: Dict[str, Any] = {}
    statuses: Dict[str, Any] = {}

    for name, meta in ledger.get("workers", {}).items():
        if meta.get("state", "working") != "working":
            continue
        step_idx, st_size, pending_step = _read_transcript_signal(meta.get("transcript_path"))
        hb_bytes, live_child_pid = _read_heartbeat_signal(name, heartbeat_dir)
        wt_mtime = _read_worktree_mtime(meta.get("worktree_path"))
        branch_sha = _read_git_branch_sha(meta.get("worktree_path"), meta.get("branch"))
        colab_qids = _read_colab_qids(meta.get("colab_checkpoint_path"))

        cur_sig = {
            "transcript_step_index": step_idx,
            "transcript_st_size": st_size,
            "heartbeat_bytes_emitted": hb_bytes,
            "worktree_max_mtime": wt_mtime,
            "git_branch_sha": branch_sha,
            "colab_completed_qids": colab_qids,
        }
        prev = prev_workers.get(name)
        if prev is None:
            zero_since = now
            delta_gt_zero = True
        else:
            prev_sig = prev.get("signals", {})
            delta_gt_zero = any(cur_sig[k] != prev_sig.get(k) for k in cur_sig)
            zero_since = now if delta_gt_zero else float(prev.get("zero_delta_since", now))

        elapsed_zero = max(0.0, now - zero_since)
        child_pid = meta.get("child_pid") if live_child_pid is None else live_child_pid
        if isinstance(child_pid, int) and not psutil.pid_exists(child_pid):
            child_pid = None

        if delta_gt_zero:
            classification = "ACTIVE_PROGRESS"
        elif elapsed_zero >= LEVEL_1_THRESHOLD_S and child_pid is not None:
            classification = "LEVEL_1_SUBPROCESS_HANG"
        elif elapsed_zero >= LEVEL_2_THRESHOLD_S and child_pid is None:
            classification = "LEVEL_2_SESSION_FREEZE"
        else:
            classification = "MONITORING"

        worker_entry = {
            "classification": classification,
            "zero_delta_since": zero_since,
            "zero_delta_seconds": round(elapsed_zero, 2),
            "child_pid": child_pid,
            "pending_cmd_step": pending_step,
            "signals": cur_sig,
        }
        next_workers[name] = worker_entry
        statuses[name] = worker_entry

    new_snapshot = {"updated_at": now, "workers": next_workers}
    report = {"checked_at": now, "workers": statuses}
    return report, new_snapshot


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only multi-signal zero-progress liveness check.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--heartbeat-dir", default=DEFAULT_HEARTBEAT_DIR)
    parser.add_argument("--now-epoch", type=float, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    ledger = json.load(open(args.ledger, "r", encoding="utf-8"))
    snapshot = (
        json.load(open(args.snapshot, "r", encoding="utf-8"))
        if os.path.isfile(args.snapshot)
        else {"workers": {}}
    )
    report, new_snapshot = evaluate_liveness(
        ledger=ledger,
        snapshot=snapshot,
        heartbeat_dir=args.heartbeat_dir,
        now_epoch=args.now_epoch,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.snapshot)), exist_ok=True)
    tmp_s = f"{args.snapshot}.tmp.{os.getpid()}"
    with open(tmp_s, "w", encoding="utf-8") as sf:
        json.dump(new_snapshot, sf, ensure_ascii=False, indent=2)
    os.replace(tmp_s, args.snapshot)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as of:
            json.dump(report, of, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
