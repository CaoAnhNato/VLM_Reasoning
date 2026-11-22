#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated 0-Token Stdlib Workflow Helper CLI (scripts/workflows/track_ops.py).

Owns config validation, preflight checks, crash-safe FSM state & CU accounting,
epistemic ledger compaction (< 3 KB), remote staging refs, symmetric rollback,
core-paper provenance verification, and Markdown report generation.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, PROJECT_ROOT)

from scripts.local.run_guarded import acquire_os_lock

DEFAULT_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".gemini", "colab_l4.mutex")
DEFAULT_ACTIVE_TRACK_PATH = os.path.join(os.path.expanduser("~"), ".gemini", "colab_active_track.json")
REQUIRED_TRACK_KEYS = {
    "track_id",
    "track_name",
    "notebooklm_notebook_id",
    "single_step_lock",
    "require_zero_kv_diff",
}
VALID_FSM_STATES = {
    "PREFLIGHT",
    "RESEARCH",
    "IMPLEMENT_AUDIT",
    "COLAB_LAUNCH",
    "COLAB_POLLING",
    "RATCHET_DECIDE",
    "MINI_BOOST",
    "HALTED_FOR_BOOST",
    "PAUSED_INFRA",
    "DONE",
}
MAX_BANK_BYTES = 200 * 1024 * 1024
EPISTEMIC_PREAMBLE_CAP_BYTES = 3072


def atomic_write_text(path: str, text: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace("\r\n", "\n"))
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha256_file(path: str, normalize_lf: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if normalize_lf:
            h.update(f.read().replace(b"\r\n", b"\n"))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest().lower()


def _run_git(repo_root: str, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", repo_root, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return proc


def _state_path(repo_root: str, track_id: int) -> str:
    return os.path.join(repo_root, ".workflow_state", f"track_{int(track_id)}", "state.json")


def _ledger_path(repo_root: str, track_id: int) -> str:
    return os.path.join(repo_root, "reports", f"track_{int(track_id)}", "epistemic_ledger.json")


def load_merged_config(track_id: int, repo_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    """Load and validate workflow_defaults.json + strict 5-field track_<id>.json via stdlib."""
    if int(track_id) not in (1, 2):
        raise ValueError(f"track_id must be 1 or 2, got {track_id}")
    defaults_path = os.path.join(repo_root, "configs", "tracks", "workflow_defaults.json")
    track_path = os.path.join(repo_root, "configs", "tracks", f"track_{int(track_id)}.json")
    if not os.path.isfile(defaults_path):
        raise FileNotFoundError(f"Missing workflow_defaults.json: {defaults_path}")
    if not os.path.isfile(track_path):
        raise FileNotFoundError(f"Missing track config: {track_path}")

    with open(defaults_path, "r", encoding="utf-8") as df:
        defaults = json.load(df)
    with open(track_path, "r", encoding="utf-8") as tf:
        track_cfg = json.load(tf)

    if not isinstance(defaults, dict) or not isinstance(track_cfg, dict):
        raise ValueError("workflow_defaults.json and track config must be JSON objects")
    if set(track_cfg.keys()) != REQUIRED_TRACK_KEYS:
        raise ValueError(
            f"track_{int(track_id)}.json must contain strictly {sorted(REQUIRED_TRACK_KEYS)}, got {sorted(track_cfg.keys())}"
        )
    if int(track_cfg["track_id"]) != int(track_id):
        raise ValueError(f"track_id mismatch in {track_path}: expected {track_id}, got {track_cfg['track_id']}")
    if not isinstance(track_cfg["track_name"], str) or not track_cfg["track_name"].strip():
        raise ValueError("track_name must be a non-empty string")
    if not isinstance(track_cfg["notebooklm_notebook_id"], str) or not track_cfg["notebooklm_notebook_id"].strip():
        raise ValueError("notebooklm_notebook_id must be a non-empty UUID string")
    if not isinstance(track_cfg["single_step_lock"], bool) or not isinstance(track_cfg["require_zero_kv_diff"], bool):
        raise ValueError("single_step_lock and require_zero_kv_diff must be booleans")

    for req_key in ("base_ref", "baseline_summary_path", "allowed_mutable_paths", "immutable_paths", "budgets", "drive_root"):
        if req_key not in defaults:
            raise ValueError(f"workflow_defaults.json missing required key: {req_key}")

    merged = dict(defaults)
    merged.update(track_cfg)
    return merged


def _validate_cal_bank_v2_stdlib(bank_dir: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    manifest_path = os.path.join(bank_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False, f"Calibration Bank v2 manifest missing: {manifest_path}", None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if str(manifest.get("schema_version")) != "2.0":
            return False, f"Unsupported Calibration Bank schema_version: {manifest.get('schema_version')!r}", None
        if float(manifest.get("max_kv_diff", -1.0)) != 0.0:
            return False, f"Calibration Bank v2 max_kv_diff != 0.0: {manifest.get('max_kv_diff')}", None
        files_map = manifest.get("files")
        if not isinstance(files_map, dict) or "shared_readout.pt" not in files_map:
            return False, "Calibration Bank v2 manifest missing shared_readout.pt", None
        total_bytes = 0
        for rel_name, meta in files_map.items():
            fpath = os.path.join(bank_dir, rel_name)
            if not os.path.isfile(fpath):
                return False, f"Calibration Bank v2 file missing: {rel_name}", None
            sz = os.path.getsize(fpath)
            total_bytes += sz
            if int(meta.get("size_bytes", -1)) != sz:
                return False, f"Calibration Bank v2 size mismatch for {rel_name}", None
            if _sha256_file(fpath, normalize_lf=False) != str(meta.get("sha256", "")).lower():
                return False, f"Calibration Bank v2 SHA-256 mismatch for {rel_name}", None
        if total_bytes > MAX_BANK_BYTES:
            return False, f"Calibration Bank v2 size {total_bytes} exceeds 200 MB limit", None
        return True, "ok", {"num_samples": manifest.get("num_samples", 0), "total_bytes": total_bytes}
    except Exception as exc:
        return False, f"Calibration Bank v2 validation error: {exc}", None


def _default_state(track_id: int, repo_root: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    branch_name = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    base_sha = _run_git(repo_root, ["rev-parse", cfg["base_ref"]]).stdout.strip()
    base_sum_path = os.path.join(repo_root, cfg["baseline_summary_path"])
    base_f1, base_sub, base_comp, probe_comp = 64.12, 62.19, 63.16, 40.92
    if os.path.isfile(base_sum_path):
        bs = json.load(open(base_sum_path, "r", encoding="utf-8"))
        b1222 = bs.get("benchmark_1222", {})
        p70 = bs.get("probe_70_heldout", {})
        base_f1 = float(b1222.get("token_f1_pct", base_f1))
        base_sub = float(b1222.get("substring_match_pct", base_sub))
        base_comp = float(b1222.get("composite_pct", base_comp))
        probe_comp = float(p70.get("composite_pct", probe_comp))
    return {
        "schema_version": "1.0",
        "track_id": int(track_id),
        "branch_name": branch_name,
        "base_commit_sha": base_sha,
        "worktree_root": os.path.abspath(repo_root),
        "fsm_state": "PREFLIGHT",
        "iteration": 0,
        "consecutive_no_gain": 0,
        "best_commit": base_sha,
        "best_f1": base_f1,
        "best_sub": base_sub,
        "best_comp": base_comp,
        "best_probe_70_comp": probe_comp,
        "halted_head_sha": None,
        "consumed_boost_digests": [],
        "active_wave": None,
        "cu_ledger": [],
        "cumulative_track_cu": 0.0,
        "event_log": [],
    }


def init_or_load_state(
    track_id: int,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        if os.path.isfile(sp):
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        st = _default_state(track_id, repo_root, cfg)
        atomic_write_json(sp, st)
        return st


def transition_state(
    track_id: int,
    to_state: str,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
    trace_id: Optional[str] = None,
    workflow_run_id: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    if to_state not in VALID_FSM_STATES:
        raise ValueError(f"Invalid FSM state: {to_state!r}")
    cfg = load_merged_config(track_id, repo_root)
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        if os.path.isfile(sp):
            st = json.load(open(sp, "r", encoding="utf-8"))
        else:
            st = _default_state(track_id, repo_root, cfg)
        prev = st.get("fsm_state")
        st["fsm_state"] = to_state
        if updates:
            for k, v in updates.items():
                st[k] = v
        event = {
            "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "from": prev,
            "to": to_state,
            "trace_id": trace_id,
            "workflow_run_id": workflow_run_id,
            "note": note,
        }
        log_list = list(st.get("event_log") or [])
        log_list.append(event)
        st["event_log"] = log_list[-200:]
        atomic_write_json(sp, st)
        return st


def set_active_wave(
    track_id: int,
    wave_id: str,
    run_folder_id: str,
    added_cell_ids: List[str],
    trace_id: str,
    launched_at: Optional[float] = None,
    ttl_s: Optional[int] = None,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
    active_track_path: str = DEFAULT_ACTIVE_TRACK_PATH,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    eff_ttl = int(ttl_s if ttl_s is not None else cfg["budgets"]["active_wave_ttl_s"])
    now_ts = float(launched_at if launched_at is not None else time.time())
    wave_obj = {
        "track_id": int(track_id),
        "trace_id": trace_id,
        "wave_id": wave_id,
        "run_folder_id": run_folder_id,
        "added_cell_ids": list(added_cell_ids),
        "launched_at": now_ts,
        "ttl_s": eff_ttl,
    }
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        if os.path.isfile(active_track_path):
            cur_global = json.load(open(active_track_path, "r", encoding="utf-8"))
            if cur_global and int(cur_global.get("track_id", -1)) != int(track_id):
                raise RuntimeError(
                    f"Colab L4 mutex conflict: track {cur_global.get('track_id')} holds active wave {cur_global.get('wave_id')}"
                )
        st = json.load(open(sp, "r", encoding="utf-8")) if os.path.isfile(sp) else _default_state(track_id, repo_root, cfg)
        st["active_wave"] = wave_obj
        atomic_write_json(sp, st)
        atomic_write_json(active_track_path, wave_obj)
        return st


def clear_active_wave(
    track_id: int,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
    active_track_path: str = DEFAULT_ACTIVE_TRACK_PATH,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        st = json.load(open(sp, "r", encoding="utf-8")) if os.path.isfile(sp) else _default_state(track_id, repo_root, cfg)
        st["active_wave"] = None
        atomic_write_json(sp, st)
        if os.path.isfile(active_track_path):
            try:
                cur_global = json.load(open(active_track_path, "r", encoding="utf-8"))
                if int(cur_global.get("track_id", -1)) == int(track_id):
                    os.remove(active_track_path)
            except Exception:
                os.remove(active_track_path)
        return st


def record_cu_usage(
    track_id: int,
    trace_id: str,
    iter_k: int,
    connect_ts: float,
    unassign_ts: float,
    unassign_confirmed: bool = True,
    teardown_mode: str = "remote_trap+dom_delete",
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    rate = float(cfg["budgets"]["cu_rate_per_hr"])
    max_cu = float(cfg["budgets"]["max_cu"])
    wall_s = max(0.0, float(unassign_ts) - float(connect_ts))
    est_cu = round((wall_s / 3600.0) * rate, 4)
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        st = json.load(open(sp, "r", encoding="utf-8")) if os.path.isfile(sp) else _default_state(track_id, repo_root, cfg)
        ledger = list(st.get("cu_ledger") or [])
        ledger.append(
            {
                "trace_id": trace_id,
                "iter_k": int(iter_k),
                "connect_ts": float(connect_ts),
                "unassign_ts": float(unassign_ts),
                "wall_seconds": round(wall_s, 2),
                "cu_rate_per_hr": rate,
                "estimated_cu": est_cu,
                "unassign_confirmed": bool(unassign_confirmed),
                "teardown_mode": teardown_mode,
            }
        )
        st["cu_ledger"] = ledger
        st["cumulative_track_cu"] = round(sum(float(x.get("estimated_cu", 0.0)) for x in ledger), 4)
        st["remaining_cu"] = round(max(0.0, max_cu - st["cumulative_track_cu"]), 4)
        atomic_write_json(sp, st)
        return st


def append_epistemic_entry(
    track_id: int,
    iter_k: int,
    outcome: str,
    hypothesis: str,
    dead_end_rule: str,
    detail: str,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
) -> Dict[str, Any]:
    lp = _ledger_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        if os.path.isfile(lp):
            data = json.load(open(lp, "r", encoding="utf-8"))
        else:
            data = {"schema_version": "1.0", "track_id": int(track_id), "entries": []}
        entries = list(data.get("entries") or [])
        entries.append(
            {
                "iter_k": int(iter_k),
                "outcome": str(outcome),
                "hypothesis": str(hypothesis).strip(),
                "dead_end_rule": str(dead_end_rule).strip(),
                "detail": str(detail).strip(),
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        data["entries"] = entries
        atomic_write_json(lp, data)
        return data


def render_epistemic_preamble(
    track_id: int,
    repo_root: str = PROJECT_ROOT,
    max_bytes: int = EPISTEMIC_PREAMBLE_CAP_BYTES,
) -> str:
    lp = _ledger_path(repo_root, track_id)
    if not os.path.isfile(lp):
        return "## Epistemic Ledger Preamble\n- No prior iterations recorded.\n"
    data = json.load(open(lp, "r", encoding="utf-8"))
    entries = data.get("entries") or []
    if not entries:
        return "## Epistemic Ledger Preamble\n- No prior iterations recorded.\n"

    rules: List[str] = []
    seen_rules = set()
    for e in entries:
        r = str(e.get("dead_end_rule") or "").strip()
        if r and r not in seen_rules:
            seen_rules.add(r)
            rules.append(f"- {r[:140]}")

    recent = entries[-3:]
    recent_lines: List[str] = []
    for e in recent:
        recent_lines.append(
            f"- [iter {e.get('iter_k')} | {e.get('outcome')}] {str(e.get('hypothesis', ''))[:160]} -> {str(e.get('detail', ''))[:220]}"
        )

    def _assemble(r_list: List[str], rec_list: List[str]) -> str:
        parts = ["## Epistemic Ledger Preamble", "### Dead-End Rules"]
        parts.extend(r_list or ["- (none)"])
        parts.append("### Most Recent Iterations")
        parts.extend(rec_list or ["- (none)"])
        return "\n".join(parts) + "\n"

    rendered = _assemble(rules, recent_lines)
    while len(rendered.encode("utf-8")) >= max_bytes and len(rules) > 1:
        rules.pop(0)
        rendered = _assemble(rules, recent_lines)
    while len(rendered.encode("utf-8")) >= max_bytes and len(recent_lines) > 1:
        recent_lines.pop(0)
        rendered = _assemble(rules, recent_lines)
    if len(rendered.encode("utf-8")) >= max_bytes:
        rendered = rendered.encode("utf-8")[: max_bytes - 4].decode("utf-8", errors="ignore").rstrip("\n") + "\n"
    return rendered


def verify_post_boost(
    track_id: int,
    boost_notes_path: Optional[str] = None,
    repo_root: str = PROJECT_ROOT,
    lock_path: str = DEFAULT_LOCK_PATH,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    sp = _state_path(repo_root, track_id)
    with acquire_os_lock(lock_path):
        st = json.load(open(sp, "r", encoding="utf-8")) if os.path.isfile(sp) else _default_state(track_id, repo_root, cfg)
        if st.get("fsm_state") != "HALTED_FOR_BOOST":
            raise RuntimeError(f"mode='post_boost' requires fsm_state == HALTED_FOR_BOOST, got {st.get('fsm_state')!r}")

        consumed = set(st.get("consumed_boost_digests") or [])
        head_sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
        halted_sha = st.get("halted_head_sha")

        digest: Optional[str] = None
        directive_summary = ""
        if boost_notes_path:
            abs_notes = boost_notes_path if os.path.isabs(boost_notes_path) else os.path.join(repo_root, boost_notes_path)
            if not os.path.isfile(abs_notes):
                raise FileNotFoundError(f"Boost resolution file not found: {abs_notes}")
            txt = open(abs_notes, "r", encoding="utf-8").read()
            if "## Diagnosis" not in txt or "## Next Direction" not in txt:
                raise ValueError(f"Boost resolution file must contain '## Diagnosis' and '## Next Direction': {abs_notes}")
            digest = _sha256_file(abs_notes, normalize_lf=True)
            if digest in consumed:
                raise RuntimeError(f"Boost resolution digest already consumed (replay prevented): {digest}")
            directive_summary = " ".join(txt.splitlines())[:220]
        elif halted_sha and head_sha != halted_sha:
            anc = _run_git(repo_root, ["merge-base", "--is-ancestor", halted_sha, head_sha], check=False)
            if anc.returncode != 0:
                raise RuntimeError("Current HEAD does not descend from halted_head_sha")
            digest = f"commit:{head_sha}"
            if digest in consumed:
                raise RuntimeError(f"Post-halt commit {head_sha} already consumed for post_boost")
            directive_summary = _run_git(repo_root, ["log", "-1", "--pretty=%s", head_sha]).stdout.strip()[:220]
        else:
            raise ValueError("mode='post_boost' requires --boost-notes-path or a new post-halt commit on HEAD")

        st["consumed_boost_digests"] = list(st.get("consumed_boost_digests") or []) + [digest]
        st["consecutive_no_gain"] = 0
        st["fsm_state"] = "PREFLIGHT"
        atomic_write_json(sp, st)

    append_epistemic_entry(
        track_id=track_id,
        iter_k=int(st.get("iteration", 0)),
        outcome="POST_BOOST_DIRECTIVE",
        hypothesis="Manual /boost resolution",
        dead_end_rule="",
        detail=directive_summary,
        repo_root=repo_root,
        lock_path=lock_path,
    )
    return {"verified": True, "digest": digest, "state": st}


def run_preflight(
    track_id: int,
    mode: str = "run",
    boost_notes_path: Optional[str] = None,
    repo_root: str = PROJECT_ROOT,
    cal_bank_dir: Optional[str] = None,
    lock_path: str = DEFAULT_LOCK_PATH,
    active_track_path: str = DEFAULT_ACTIVE_TRACK_PATH,
) -> Dict[str, Any]:
    remediations: List[str] = []
    checks: Dict[str, Any] = {}
    cfg: Optional[Dict[str, Any]] = None

    try:
        cfg = load_merged_config(track_id, repo_root)
        checks["config"] = {"passed": True, "track_name": cfg["track_name"]}
    except Exception as exc:
        checks["config"] = {"passed": False, "error": str(exc)}
        remediations.append(f"Fix configs/tracks configuration: {exc}")

    # Git ancestry & clean tree
    if cfg:
        anc = _run_git(repo_root, ["merge-base", "--is-ancestor", cfg["base_ref"], "HEAD"], check=False)
        anc_ok = anc.returncode == 0
        checks["git_ancestry"] = {"passed": anc_ok, "base_ref": cfg["base_ref"]}
        if not anc_ok:
            remediations.append(f"Ensure HEAD descends from {cfg['base_ref']}")

        porcelain = _run_git(repo_root, ["status", "--porcelain"], check=False).stdout.strip()
        clean_ok = porcelain == ""
        checks["git_clean"] = {"passed": clean_ok, "dirty_entries": porcelain.splitlines()[:10] if porcelain else []}
        if not clean_ok and mode != "status":
            remediations.append("Commit or clean working tree modifications before running workflow")

        # Immutable pinned SHA-256 verification
        imm_errors: List[str] = []
        for rel_p, exp_sha in (cfg.get("pinned_sha256") or {}).items():
            full_p = os.path.join(repo_root, rel_p)
            if not os.path.isfile(full_p):
                imm_errors.append(f"Missing immutable file: {rel_p}")
            else:
                act = _sha256_file(full_p, normalize_lf=True)
                if act != str(exp_sha).lower():
                    imm_errors.append(f"SHA-256 mismatch for {rel_p}: expected={exp_sha} actual={act}")
        checks["immutable_files"] = {"passed": len(imm_errors) == 0, "errors": imm_errors}
        if imm_errors:
            remediations.extend(imm_errors)

    # Calibration Bank v2 SHA-256 check
    bank_rel = cal_bank_dir or (cfg.get("cal_bank_dir") if cfg else "data/calibration_bank_v2")
    bank_abs = bank_rel if os.path.isabs(bank_rel) else os.path.join(repo_root, bank_rel)
    bank_ok, bank_msg, bank_meta = _validate_cal_bank_v2_stdlib(bank_abs)
    checks["cal_bank_v2"] = {"passed": bank_ok, "message": bank_msg, "meta": bank_meta, "path": bank_abs}
    if not bank_ok and mode != "status":
        remediations.append(f"Stage valid Calibration Bank v2 at {bank_abs}: {bank_msg}")

    # Environment hygiene
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    py_enc = os.environ.get("PYTHONIOENCODING", "")
    env_ok = (cuda_env == "") and (py_enc.lower() == "utf-8")
    checks["host_env"] = {"passed": env_ok, "CUDA_VISIBLE_DEVICES": cuda_env, "PYTHONIOENCODING": py_enc}
    if not env_ok and mode != "status":
        remediations.append("Set CUDA_VISIBLE_DEVICES=\"\" and PYTHONIOENCODING=utf-8 before launching")

    state = init_or_load_state(track_id, repo_root=repo_root, lock_path=lock_path) if cfg else {}
    resume_action = "START_ITERATION"
    halt_refused = False

    if cfg and state:
        # Check cross-track Colab L4 conflict
        if os.path.isfile(active_track_path):
            try:
                g_wave = json.load(open(active_track_path, "r", encoding="utf-8"))
                g_tid = int(g_wave.get("track_id", -1))
                if g_tid not in (-1, int(track_id)):
                    resume_action = "CROSS_TRACK_CONFLICT"
                    checks["cross_track_mutex"] = {"passed": False, "active_track": g_wave}
                    remediations.append(
                        f"Colab L4 active on Track {g_tid} (wave_id={g_wave.get('wave_id')}); finish or clear it before Track {track_id}."
                    )
            except Exception:
                pass
        if "cross_track_mutex" not in checks:
            checks["cross_track_mutex"] = {"passed": True}

        # Check active_wave crash resume vs expired TTL
        aw = state.get("active_wave")
        if isinstance(aw, dict) and aw.get("wave_id"):
            age_s = time.time() - float(aw.get("launched_at", 0.0))
            ttl_s = float(aw.get("ttl_s", cfg["budgets"]["active_wave_ttl_s"]))
            if age_s < ttl_s:
                if resume_action != "CROSS_TRACK_CONFLICT":
                    resume_action = "COLAB_POLLING"
                checks["active_wave"] = {"passed": True, "status": "UNEXPIRED", "age_s": round(age_s, 1), "ttl_s": ttl_s}
            else:
                if resume_action != "CROSS_TRACK_CONFLICT":
                    resume_action = "ORPHAN_EXPIRED"
                checks["active_wave"] = {"passed": False, "status": "ORPHAN_EXPIRED", "age_s": round(age_s, 1), "ttl_s": ttl_s}
                remediations.append(
                    f"Active wave {aw.get('wave_id')} exceeded ttl_s={ttl_s}s; confirm L4 teardown and clear-active-wave."
                )

        # HALTED_FOR_BOOST handshake
        if state.get("fsm_state") == "HALTED_FOR_BOOST":
            iter_k = int(state.get("iteration", 0))
            notes_hint = f"reports/track_{int(track_id)}/iter_{iter_k}/BOOST_RESOLUTION_iter_{iter_k}.md"
            if mode == "run":
                halt_refused = True
                resume_action = "HALTED_FOR_BOOST_REFUSED"
                checks["boost_handshake"] = {"passed": False, "fsm_state": "HALTED_FOR_BOOST"}
                remediations.append(
                    f"Track {track_id} is HALTED_FOR_BOOST. Invoke /run-workflow with "
                    f'Args={{"track": {int(track_id)}, "mode": "post_boost", "boost_notes_path": "{notes_hint}"}}'
                )
            elif mode == "post_boost":
                try:
                    res = verify_post_boost(
                        track_id=track_id,
                        boost_notes_path=boost_notes_path,
                        repo_root=repo_root,
                        lock_path=lock_path,
                    )
                    state = res["state"]
                    resume_action = "POST_BOOST_VERIFIED"
                    checks["boost_handshake"] = {"passed": True, "digest": res["digest"]}
                except Exception as exc:
                    checks["boost_handshake"] = {"passed": False, "error": str(exc)}
                    remediations.append(f"Provide valid boost resolution for mode=post_boost: {exc}")

    # Budget check
    if cfg and state and mode in ("run", "post_boost"):
        budgets = cfg["budgets"]
        rem_cu = round(float(budgets["max_cu"]) - float(state.get("cumulative_track_cu", 0.0)), 4)
        it = int(state.get("iteration", 0))
        if it >= int(budgets["max_iterations"]) or rem_cu < float(budgets["min_stage_b_cu"]):
            checks["budget"] = {"passed": False, "iteration": it, "remaining_cu": rem_cu}
            if resume_action not in ("COLAB_POLLING", "ORPHAN_EXPIRED", "CROSS_TRACK_CONFLICT", "HALTED_FOR_BOOST_REFUSED"):
                resume_action = "BUDGET_EXHAUSTED"
            remediations.append(f"Track {track_id} reached budget/iteration cap (iter={it}, remaining_cu={rem_cu}).")
        else:
            checks["budget"] = {"passed": True, "iteration": it, "remaining_cu": rem_cu}

    if mode == "status":
        ready = checks.get("config", {}).get("passed", False)
    else:
        ready = all(c.get("passed", False) for c in checks.values()) and not halt_refused

    return {
        "ready": bool(ready),
        "track_id": int(track_id),
        "mode": mode,
        "resume_action": resume_action,
        "halt_refused": halt_refused,
        "config": cfg,
        "state": state,
        "checks": checks,
        "remediations": remediations,
    }


def verify_git_scope(
    track_id: int,
    base_sha: str,
    target_sha: str = "HEAD",
    repo_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    allowed = set(cfg.get("allowed_mutable_paths") or [])
    immutable = set(cfg.get("immutable_paths") or [])
    diff_files = [
        line.strip().replace("\\", "/")
        for line in _run_git(repo_root, ["diff", "--name-only", base_sha, target_sha]).stdout.splitlines()
        if line.strip()
    ]
    untracked = [
        line.strip().replace("\\", "/")
        for line in _run_git(repo_root, ["ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
        if line.strip()
    ]
    all_touched = sorted(set(diff_files + untracked))
    violations: List[str] = []
    for rel_p in all_touched:
        if rel_p in immutable:
            violations.append(f"Immutable file touched: {rel_p}")
        elif rel_p.startswith("docs/specs/"):
            violations.append(f"Implementer modified docs/specs/: {rel_p}")
        elif rel_p not in allowed:
            violations.append(f"File outside allowed_mutable_paths: {rel_p}")
    return {"passed": len(violations) == 0, "touched_files": all_touched, "violations": violations}


def commit_spec(track_id: int, iter_k: int, spec_path: str, repo_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    rel_spec = os.path.relpath(spec_path, repo_root).replace("\\", "/") if os.path.isabs(spec_path) else spec_path.replace("\\", "/")
    _run_git(repo_root, ["add", "--", rel_spec])
    st = _run_git(repo_root, ["diff", "--cached", "--name-only"]).stdout.strip()
    if st:
        _run_git(repo_root, ["commit", "-m", f"spec(track-{int(track_id)}): iter_{int(iter_k)}"])
    sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
    return {"spec_sha": sha, "spec_path": rel_spec}


def commit_impl(
    track_id: int,
    iter_k: int,
    spec_sha: Optional[str] = None,
    repo_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    cfg = load_merged_config(track_id, repo_root)
    allowed = list(cfg.get("allowed_mutable_paths") or [])
    if spec_sha:
        scope = verify_git_scope(track_id, spec_sha, "HEAD", repo_root=repo_root)
        if not scope["passed"]:
            raise RuntimeError(f"Scope violation before commit-impl: {scope['violations']}")
    for rel_p in allowed:
        if os.path.exists(os.path.join(repo_root, rel_p)):
            _run_git(repo_root, ["add", "--", rel_p])
    st = _run_git(repo_root, ["diff", "--cached", "--name-only"]).stdout.strip()
    if st:
        _run_git(repo_root, ["commit", "-m", f"impl(track-{int(track_id)}): iter_{int(iter_k)}"])
    sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
    return {"impl_sha": sha}


def push_staging_ref(track_id: int, repo_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    ref = f"refs/heads/colab-staging/track-{int(track_id)}"
    _run_git(repo_root, ["push", "--force", "origin", f"HEAD:{ref}"])
    sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
    return {"pushed": True, "staging_ref": ref, "commit_sha": sha}


def prune_staging_ref(track_id: int, repo_root: str = PROJECT_ROOT) -> Dict[str, Any]:
    ref = f"refs/heads/colab-staging/track-{int(track_id)}"
    proc = _run_git(repo_root, ["push", "origin", "--delete", f"colab-staging/track-{int(track_id)}"], check=False)
    return {"pruned": proc.returncode == 0, "staging_ref": ref, "stderr": proc.stderr.strip()}


def rollback_clean(
    track_id: int,
    best_commit: str,
    iter_k: int,
    repo_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Symmetric rollback scoped to src/nevfe/ and tests/ while preserving specs, reports, and metrics tests."""
    added = [
        line.strip().replace("\\", "/")
        for line in _run_git(
            repo_root,
            ["diff", "--name-only", "--diff-filter=A", f"{best_commit}..HEAD", "--", "src/nevfe/", "tests/"],
        ).stdout.splitlines()
        if line.strip()
    ]
    protected = {"tests/test_metrics_relaxation.py", "src/nevfe/core/frozen_foveation.py"}
    to_remove = [p for p in added if p not in protected]
    if to_remove:
        _run_git(repo_root, ["rm", "-f", "--ignore-unmatch", "--", *to_remove])

    _run_git(repo_root, ["checkout", best_commit, "--", "src/nevfe/", "tests/"])
    _run_git(repo_root, ["clean", "-fd", "--", "src/nevfe/", "tests/"])

    staged = _run_git(repo_root, ["diff", "--cached", "--name-only"]).stdout.strip()
    if staged:
        short_best = _run_git(repo_root, ["rev-parse", "--short", best_commit]).stdout.strip()
        _run_git(repo_root, ["commit", "-m", f"rollback(track-{int(track_id)}): iter_{int(iter_k)} -> {short_best}"])

    rem_status = _run_git(repo_root, ["status", "--porcelain", "--", "src/nevfe/", "tests/"]).stdout.strip()
    rem_diff = _run_git(repo_root, ["diff", best_commit, "--", "src/nevfe/", "tests/"]).stdout.strip()
    if rem_status:
        raise RuntimeError(f"Working tree not clean after rollback-clean: {rem_status}")
    if rem_diff:
        raise RuntimeError("Non-empty diff against best_commit on src/nevfe/ or tests/ after rollback-clean")

    head_sha = _run_git(repo_root, ["rev-parse", "HEAD"]).stdout.strip()
    return {"rolled_back": True, "best_commit": best_commit, "head_sha": head_sha, "removed_added_files": to_remove}


def verify_papers_provenance(
    verification_json: str,
    cache_dir: str,
    repo_root: str = PROJECT_ROOT,
) -> Dict[str, Any]:
    if not os.path.isfile(verification_json):
        return {"passed": False, "needs_waiver": False, "errors": [f"Missing verification JSON: {verification_json}"]}
    data = json.load(open(verification_json, "r", encoding="utf-8"))
    entries = data.get("papers") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not (1 <= len(entries) <= 3):
        return {"passed": False, "needs_waiver": False, "errors": ["Expected 1..3 core paper entries"]}

    errors: List[str] = []
    needs_waiver = False
    for idx, entry in enumerate(entries):
        pid = str(entry.get("paper_id") or entry.get("id") or f"paper_{idx}")
        status = str(entry.get("status", ""))
        if status == "UNAVAILABLE_NEEDS_WAIVER":
            waiver = str(entry.get("waiver_path") or "").strip()
            w_abs = waiver if os.path.isabs(waiver) else os.path.join(repo_root, waiver) if waiver else ""
            if not waiver or not os.path.isfile(w_abs) or os.path.getsize(w_abs) == 0:
                needs_waiver = True
                errors.append(f"{pid} UNAVAILABLE_NEEDS_WAIVER without signed waiver_path")
            continue
        if status not in {"VERIFIED_TEX", "VERIFIED_PDF"}:
            errors.append(f"Unsupported paper status {status!r} for {pid}")
            continue

        cached_raw = str(entry.get("cached_path") or "").strip()
        if not cached_raw:
            errors.append(f"Missing cached_path for {pid}")
            continue
        cached_abs = cached_raw if os.path.isabs(cached_raw) else os.path.join(repo_root, cached_raw)
        if not os.path.isfile(cached_abs):
            cached_abs = os.path.join(cache_dir, os.path.basename(cached_raw))
        if not os.path.isfile(cached_abs):
            errors.append(f"Cached source missing for {pid}: {cached_raw}")
            continue

        expected_sha = str(entry.get("source_sha256") or "").lower().strip()
        raw_bytes = open(cached_abs, "rb").read()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest().lower()
        if not expected_sha or actual_sha != expected_sha:
            errors.append(f"Source SHA-256 mismatch for {pid}: expected={expected_sha} actual={actual_sha}")

        norm_text = " ".join(raw_bytes.decode("utf-8", errors="replace").lower().split())
        section = " ".join(str(entry.get("section") or entry.get("target_section") or "").lower().split())
        anchor = " ".join(str(entry.get("anchor") or entry.get("equation_anchor") or "").lower().split())
        if not section or section not in norm_text:
            errors.append(f"Section {section!r} missing from cached source for {pid}")
        if anchor and anchor not in norm_text:
            errors.append(f"Anchor {anchor!r} missing from cached source for {pid}")

    return {"passed": len(errors) == 0, "needs_waiver": needs_waiver, "paper_count": len(entries), "errors": errors}


def generate_boost_handoff_report(
    track_id: int,
    iter_k: int,
    repo_root: str = PROJECT_ROOT,
    forensic_md_path: Optional[str] = None,
    probe_summary_path: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    cfg = load_merged_config(track_id, repo_root)
    st = init_or_load_state(track_id, repo_root=repo_root)
    budgets = cfg["budgets"]
    rem_cu = round(float(budgets["max_cu"]) - float(st.get("cumulative_track_cu", 0.0)), 4)
    rem_iters = max(0, int(budgets["max_iterations"]) - int(st.get("iteration", iter_k)))
    dest = out_path or os.path.join(
        repo_root, "reports", f"track_{int(track_id)}", f"iter_{int(iter_k)}", f"BOOST_HANDOFF_iter_{int(iter_k)}.md"
    )
    res_hint = f"reports/track_{int(track_id)}/iter_{int(iter_k)}/BOOST_RESOLUTION_iter_{int(iter_k)}.md"

    forensic_excerpt = open(forensic_md_path, "r", encoding="utf-8").read()[:4000] if (forensic_md_path and os.path.isfile(forensic_md_path)) else "_No forensic_delta.md available._"
    probe_txt = open(probe_summary_path, "r", encoding="utf-8").read()[:2000] if (probe_summary_path and os.path.isfile(probe_summary_path)) else "_No probe_ablation_summary.json available._"
    preamble = render_epistemic_preamble(track_id, repo_root=repo_root)

    content = "\n".join(
        [
            f"# Manual `/boost` Handoff Dossier — Track {int(track_id)} (`{cfg['track_name']}`), Iteration {int(iter_k)}",
            "",
            "## Scoreboard & Budget",
            f"- **Best Commit**: `{st.get('best_commit')}`",
            f"- **Best Stage B**: F1=`{st.get('best_f1')}%`, Sub=`{st.get('best_sub')}%`, Comp=`{st.get('best_comp')}%`",
            f"- **Best Stage A 70-Q Comp**: `{st.get('best_probe_70_comp')}%`",
            f"- **Consecutive No-Gain**: `{st.get('consecutive_no_gain')}` (Circuit Breaker Triggered)",
            f"- **Remaining Budget**: `{rem_iters}` iterations, `{rem_cu:.2f} CU` (cumulative `{st.get('cumulative_track_cu', 0.0):.2f} / {budgets['max_cu']} CU`)",
            "",
            preamble.strip(),
            "",
            "## Stage A LOO / Diagnostics",
            "```json",
            probe_txt.strip(),
            "```",
            "",
            "## Latest Forensic Delta",
            forensic_excerpt.strip(),
            "",
            "## Copy-Pasteable `/boost` Prompt & Resume Command",
            "1. Invoke `/boost` with:",
            f"   > Analyze Track {int(track_id)} ({cfg['track_name']}) plateau at iter {int(iter_k)} (best Comp {st.get('best_comp')}%, Stage A {st.get('best_probe_70_comp')}%), inspect `{dest}`, and write `## Diagnosis` + `## Next Direction` into `{res_hint}`.",
            "2. Resume workflow via `/run-workflow` with:",
            f'   `Args = {{"track": {int(track_id)}, "mode": "post_boost", "boost_notes_path": "{res_hint}"}}`',
            "",
        ]
    )
    atomic_write_text(dest, content)
    return dest


def generate_final_report(
    track_id: int,
    repo_root: str = PROJECT_ROOT,
    out_path: Optional[str] = None,
) -> str:
    cfg = load_merged_config(track_id, repo_root)
    st = init_or_load_state(track_id, repo_root=repo_root)
    best_commit = str(st.get("best_commit"))
    clean_diff = _run_git(repo_root, ["diff", best_commit, "--", "src/nevfe/", "tests/"], check=False).stdout.strip()
    parity_status = "ZERO_DIFF_VERIFIED" if clean_diff == "" else "NON_EMPTY_DIFF_WARNING"
    dest = out_path or os.path.join(repo_root, "reports", f"track_{int(track_id)}", f"TRACK_{int(track_id)}_FINAL_REPORT.md")

    lp = _ledger_path(repo_root, track_id)
    ledger_entries = json.load(open(lp, "r", encoding="utf-8")).get("entries", []) if os.path.isfile(lp) else []
    cu_rows = st.get("cu_ledger") or []

    lines = [
        f"# Track {int(track_id)} Final Research Report (`{cfg['track_name']}`)",
        "",
        "## Best Checkpoint Summary",
        f"- **Track**: `{track_id}` (`{cfg['track_name']}`)",
        f"- **FSM Final State**: `{st.get('fsm_state')}`",
        f"- **Iterations Completed**: `{st.get('iteration')}` / `{cfg['budgets']['max_iterations']}`",
        f"- **Best Commit**: `{best_commit}` (`{parity_status}`)",
        f"- **Stage B Metrics**: Token F1=`{st.get('best_f1')}%`, Substring Match=`{st.get('best_sub')}%`, Composite=`{st.get('best_comp')}%`",
        f"- **Stage A Best 70-Q Comp**: `{st.get('best_probe_70_comp')}%`",
        f"- **Total Colab CU Used**: `{st.get('cumulative_track_cu', 0.0):.4f}` / `{cfg['budgets']['max_cu']}` CU",
        "",
        "## Colab Wave CU Ledger",
    ]
    if cu_rows:
        for row in cu_rows:
            lines.append(
                f"- `{row.get('trace_id')}` (iter {row.get('iter_k')}): {row.get('wall_seconds')}s -> {row.get('estimated_cu')} CU (`{row.get('teardown_mode')}`)"
            )
    else:
        lines.append("- No Colab waves recorded.")

    lines.extend(["", "## Epistemic History & Verified Outcomes"])
    if ledger_entries:
        for e in ledger_entries:
            lines.append(f"- Iter {e.get('iter_k')} [{e.get('outcome')}]: {e.get('hypothesis')} | {e.get('detail')}")
    else:
        lines.append("- No epistemic entries recorded.")

    atomic_write_text(dest, "\n".join(lines) + "\n")
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidated VLM Track Research Workflow helper CLI.")
    sub = parser.add_subparsers(dest="group", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--track", type=int, choices=[1, 2], required=True)
    p_pre.add_argument("--mode", choices=["run", "post_boost", "preflight", "status"], default="run")
    p_pre.add_argument("--boost-notes-path", default=None)
    p_pre.add_argument("--repo-root", default=PROJECT_ROOT)
    p_pre.add_argument("--cal-bank-dir", default=None)
    p_pre.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    p_pre.add_argument("--active-track-file", default=DEFAULT_ACTIVE_TRACK_PATH)

    p_st = sub.add_parser("state")
    st_sub = p_st.add_subparsers(dest="action", required=True)

    for act in ("init-or-load", "status"):
        sp_a = st_sub.add_parser(act)
        sp_a.add_argument("--track", type=int, choices=[1, 2], required=True)
        sp_a.add_argument("--repo-root", default=PROJECT_ROOT)
        sp_a.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)

    sp_tr = st_sub.add_parser("transition")
    sp_tr.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_tr.add_argument("--to-state", required=True)
    sp_tr.add_argument("--trace-id", default=None)
    sp_tr.add_argument("--workflow-run-id", default=None)
    sp_tr.add_argument("--updates-json", default=None)
    sp_tr.add_argument("--note", default="")
    sp_tr.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_tr.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)

    sp_sw = st_sub.add_parser("set-active-wave")
    sp_sw.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_sw.add_argument("--wave-id", required=True)
    sp_sw.add_argument("--run-folder-id", required=True)
    sp_sw.add_argument("--added-cell-ids-json", default="[]")
    sp_sw.add_argument("--trace-id", required=True)
    sp_sw.add_argument("--launched-at", type=float, default=None)
    sp_sw.add_argument("--ttl-s", type=int, default=None)
    sp_sw.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_sw.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    sp_sw.add_argument("--active-track-file", default=DEFAULT_ACTIVE_TRACK_PATH)

    sp_cw = st_sub.add_parser("clear-active-wave")
    sp_cw.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_cw.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_cw.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    sp_cw.add_argument("--active-track-file", default=DEFAULT_ACTIVE_TRACK_PATH)

    sp_cu = st_sub.add_parser("record-cu")
    sp_cu.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_cu.add_argument("--trace-id", required=True)
    sp_cu.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_cu.add_argument("--connect-ts", type=float, required=True)
    sp_cu.add_argument("--unassign-ts", type=float, required=True)
    sp_cu.add_argument("--unassign-confirmed", action="store_true", default=False)
    sp_cu.add_argument("--teardown-mode", default="remote_trap+dom_delete")
    sp_cu.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_cu.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)

    sp_ae = st_sub.add_parser("append-epistemic")
    sp_ae.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_ae.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_ae.add_argument("--outcome", required=True)
    sp_ae.add_argument("--hypothesis", required=True)
    sp_ae.add_argument("--dead-end-rule", default="")
    sp_ae.add_argument("--detail", required=True)
    sp_ae.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_ae.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)

    sp_rp = st_sub.add_parser("render-epistemic-preamble")
    sp_rp.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_rp.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_rp.add_argument("--out-path", default=None)

    sp_vb = st_sub.add_parser("verify-post-boost")
    sp_vb.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_vb.add_argument("--boost-notes-path", default=None)
    sp_vb.add_argument("--repo-root", default=PROJECT_ROOT)
    sp_vb.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)

    p_git = sub.add_parser("git")
    git_sub = p_git.add_subparsers(dest="action", required=True)

    sp_cs = git_sub.add_parser("commit-spec")
    sp_cs.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_cs.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_cs.add_argument("--spec-path", required=True)
    sp_cs.add_argument("--repo-root", default=PROJECT_ROOT)

    sp_ci = git_sub.add_parser("commit-impl")
    sp_ci.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_ci.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_ci.add_argument("--spec-sha", default=None)
    sp_ci.add_argument("--repo-root", default=PROJECT_ROOT)

    sp_vs = git_sub.add_parser("verify-scope")
    sp_vs.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_vs.add_argument("--base-sha", required=True)
    sp_vs.add_argument("--target-sha", default="HEAD")
    sp_vs.add_argument("--repo-root", default=PROJECT_ROOT)

    for act in ("push-staging", "prune-staging"):
        sp_ps = git_sub.add_parser(act)
        sp_ps.add_argument("--track", type=int, choices=[1, 2], required=True)
        sp_ps.add_argument("--repo-root", default=PROJECT_ROOT)

    sp_rb = git_sub.add_parser("rollback-clean")
    sp_rb.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_rb.add_argument("--best-commit", required=True)
    sp_rb.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_rb.add_argument("--repo-root", default=PROJECT_ROOT)

    p_pap = sub.add_parser("papers")
    pap_sub = p_pap.add_subparsers(dest="action", required=True)
    sp_pv = pap_sub.add_parser("verify")
    sp_pv.add_argument("--verification-json", required=True)
    sp_pv.add_argument("--cache-dir", required=True)
    sp_pv.add_argument("--repo-root", default=PROJECT_ROOT)

    p_rep = sub.add_parser("report")
    rep_sub = p_rep.add_subparsers(dest="action", required=True)
    sp_bh = rep_sub.add_parser("boost-handoff")
    sp_bh.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_bh.add_argument("--iter", dest="iter_k", type=int, required=True)
    sp_bh.add_argument("--forensic-md", default=None)
    sp_bh.add_argument("--probe-summary", default=None)
    sp_bh.add_argument("--out-path", default=None)
    sp_bh.add_argument("--repo-root", default=PROJECT_ROOT)

    sp_fr = rep_sub.add_parser("final-report")
    sp_fr.add_argument("--track", type=int, choices=[1, 2], required=True)
    sp_fr.add_argument("--out-path", default=None)
    sp_fr.add_argument("--repo-root", default=PROJECT_ROOT)

    args = parser.parse_args(argv)

    if args.group == "preflight":
        out = run_preflight(
            track_id=args.track,
            mode=args.mode,
            boost_notes_path=args.boost_notes_path,
            repo_root=args.repo_root,
            cal_bank_dir=args.cal_bank_dir,
            lock_path=args.lock_path,
            active_track_path=args.active_track_file,
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if out["ready"] else (2 if out.get("halt_refused") else 1)

    if args.group == "state":
        if args.action in ("init-or-load", "status"):
            out = init_or_load_state(args.track, repo_root=args.repo_root, lock_path=args.lock_path)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "transition":
            upd = json.loads(args.updates_json) if args.updates_json else None
            out = transition_state(
                args.track,
                args.to_state,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
                trace_id=args.trace_id,
                workflow_run_id=args.workflow_run_id,
                updates=upd,
                note=args.note,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "set-active-wave":
            out = set_active_wave(
                track_id=args.track,
                wave_id=args.wave_id,
                run_folder_id=args.run_folder_id,
                added_cell_ids=json.loads(args.added_cell_ids_json),
                trace_id=args.trace_id,
                launched_at=args.launched_at,
                ttl_s=args.ttl_s,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
                active_track_path=args.active_track_file,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "clear-active-wave":
            out = clear_active_wave(
                track_id=args.track,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
                active_track_path=args.active_track_file,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "record-cu":
            out = record_cu_usage(
                track_id=args.track,
                trace_id=args.trace_id,
                iter_k=args.iter_k,
                connect_ts=args.connect_ts,
                unassign_ts=args.unassign_ts,
                unassign_confirmed=args.unassign_confirmed,
                teardown_mode=args.teardown_mode,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "append-epistemic":
            out = append_epistemic_entry(
                track_id=args.track,
                iter_k=args.iter_k,
                outcome=args.outcome,
                hypothesis=args.hypothesis,
                dead_end_rule=args.dead_end_rule,
                detail=args.detail,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "render-epistemic-preamble":
            txt = render_epistemic_preamble(args.track, repo_root=args.repo_root)
            if args.out_path:
                atomic_write_text(args.out_path, txt)
            print(json.dumps({"bytes": len(txt.encode("utf-8")), "preamble": txt}, ensure_ascii=False))
            return 0
        if args.action == "verify-post-boost":
            out = verify_post_boost(
                args.track,
                boost_notes_path=args.boost_notes_path,
                repo_root=args.repo_root,
                lock_path=args.lock_path,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0

    if args.group == "git":
        if args.action == "commit-spec":
            out = commit_spec(args.track, args.iter_k, args.spec_path, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "commit-impl":
            out = commit_impl(args.track, args.iter_k, spec_sha=args.spec_sha, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "verify-scope":
            out = verify_git_scope(args.track, args.base_sha, target_sha=args.target_sha, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0 if out["passed"] else 1
        if args.action == "push-staging":
            out = push_staging_ref(args.track, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "prune-staging":
            out = prune_staging_ref(args.track, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
        if args.action == "rollback-clean":
            out = rollback_clean(args.track, args.best_commit, args.iter_k, repo_root=args.repo_root)
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0

    if args.group == "papers" and args.action == "verify":
        out = verify_papers_provenance(args.verification_json, args.cache_dir, repo_root=args.repo_root)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if out["passed"] else 1

    if args.group == "report":
        if args.action == "boost-handoff":
            path = generate_boost_handoff_report(
                track_id=args.track,
                iter_k=args.iter_k,
                repo_root=args.repo_root,
                forensic_md_path=args.forensic_md,
                probe_summary_path=args.probe_summary,
                out_path=args.out_path,
            )
            print(json.dumps({"path": path}, ensure_ascii=False))
            return 0
        if args.action == "final-report":
            path = generate_final_report(
                track_id=args.track,
                repo_root=args.repo_root,
                out_path=args.out_path,
            )
            print(json.dumps({"path": path}, ensure_ascii=False))
            return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
