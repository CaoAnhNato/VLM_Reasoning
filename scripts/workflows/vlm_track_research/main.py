# -*- coding: utf-8 -*-
"""Thin /run-workflow orchestrator for training-free VLM track research (stdlib only)."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(WORKSPACES[0]).resolve() if WORKSPACES else Path.cwd().resolve()
OPS = str(ROOT / "scripts" / "workflows" / "track_ops.py")
AUDITOR = str(ROOT / "scripts" / "audit" / "run_auditor_checks.py")
DIFF_CLI = str(ROOT / "scripts" / "benchmark" / "diff_predictions.py")
SYNC_CLI = str(ROOT / "scripts" / "benchmark" / "colab_drive_sync.py")
PROMPTS_DIR = ROOT / "scripts" / "workflows" / "vlm_track_research" / "prompts"
GDRIVE_ACCOUNT = os.environ.get("VLM_GDRIVE_ACCOUNT", "doantranlacanh@gmail.com")


def _obj(**props) -> dict:
    return {"type": "object", "required": list(props), "properties": props}


def _arr(item: dict = None) -> dict:
    return {"type": "array", "items": item or {"type": "string"}}


SCHEMAS = {
    "teardown": _obj(cells_deleted={"type": "integer"}, fallback_unassign_called={"type": "boolean"}, unassign_ts={"type": "number"}),
    "research_t1": _obj(
        hypothesis_summary={"type": "string"},
        core_papers=_arr(_obj(id={"type": "string"}, source_type={"type": "string"}, target_section={"type": "string"})),
        target_equations=_arr(),
    ),
    "paper_extractor": _obj(verification_json_path={"type": "string"}, verified_primitives_path={"type": "string"}, statuses=_arr()),
    "research_t2": _obj(spec_path={"type": "string"}, component_names=_arr(), num_new_scalars={"type": "integer"}),
    "implementer": _obj(changed_files=_arr(), candidate_test_path={"type": "string"}, summary={"type": "string"}),
    "colab_launch": _obj(wave_id={"type": "string"}, run_folder_id={"type": "string"}, added_cell_ids=_arr(), connect_ts={"type": "number"}),
    "mini_boost": _obj(contradiction={"type": "string"}, root_cause={"type": "string"}, dead_end_rule={"type": "string"}, next_direction_hint={"type": "string"}),
}


def _resolve_py() -> str:
    cands = [os.environ.get("VLM_WORKFLOW_PYTHON", ""), r"C:\ProgramData\anaconda3\python.exe", sys.executable]
    for p in cands:
        if p and os.path.isfile(p):
            probe = subprocess.run([p, "-c", "import psutil, pytest"], capture_output=True, check=False)
            if probe.returncode == 0:
                return p
    return sys.executable


PY = _resolve_py()


def _sub_env() -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_cli(script: str, cli_args: list, check: bool = True) -> tuple:
    proc = subprocess.run([PY, script, *cli_args], cwd=str(ROOT), env=_sub_env(), capture_output=True, text=True, encoding="utf-8", check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{Path(script).name} {' '.join(cli_args)} failed (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}")
    return proc.returncode, proc.stdout, proc.stderr


def _run_json(script: str, cli_args: list, check: bool = True) -> tuple:
    rc, out, err = _run_cli(script, cli_args, check=check)
    try:
        return rc, json.loads(out), err
    except json.JSONDecodeError:
        return rc, {"raw_stdout": out, "raw_stderr": err}, err


def _load_prompt(name: str, **fmt) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").format(**fmt)


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().lower()


async def _pause_infra(track: int, note: str, trace_id: str = None) -> dict:
    _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "PAUSED_INFRA", "--note", note, *(["--trace-id", trace_id] if trace_id else [])], check=False)
    log(f"[PAUSED_INFRA] {note}")
    return {"status": "PAUSED_INFRA", "note": note}


async def _poll_and_teardown(track: int, iter_k: int, state: dict, cfg: dict) -> tuple:
    aw = state["active_wave"]
    trace_id, wave_id, folder_id = aw["trace_id"], aw["wave_id"], aw["run_folder_id"]
    cells = json.dumps(aw.get("added_cell_ids") or [])
    cur_cells = str(ROOT / ".workflow_state" / f"track_{track}" / f"cells_end_{iter_k}.json")
    stalled, complete = False, False
    while not (complete or stalled):
        if os.environ.get("RCLONE_DRIVE_TOKEN_JSON"):
            rc, data, _ = _run_json(SYNC_CLI, ["--poll-markers", "--folder-id", folder_id], check=False)
            if rc == 0 and isinstance(data, dict) and "complete" in data:
                complete, stalled = bool(data["complete"]), bool(data["stalled"])
        if not (complete or stalled):
            poll_cmd = f"$env:AGY_ACCOUNT='{GDRIVE_ACCOUNT}'; gdrive readonly ls {folder_id} --max 0 --json"
            raw = await call_tool("run_command", CommandLine=poll_cmd, WaitMsBeforeAsync=10000, toolSummary="Poll Colab Drive markers")
            s_idx, e_idx = raw.find("["), raw.rfind("]")
            items = json.loads(raw[s_idx : e_idx + 1]) if (s_idx != -1 and e_idx > s_idx) else []
            names = {f.get("name", "") for f in items if isinstance(f, dict)}
            complete = "WAVE_COMPLETE_MARKER" in names
            stalled = "WAVE_STALLED_MARKER" in names
        if complete or stalled:
            break
        if time.time() - float(aw.get("launched_at", 0)) >= float(aw.get("ttl_s", 6000)):
            stalled = True
            break
        time.sleep(150)

    td_prompt = _load_prompt("colab_teardown.md", track_id=track, wave_id=wave_id, added_cell_ids_json=cells, current_cells_path=cur_cells, fallback_unassign=str(stalled).lower())
    td = await agent(td_prompt, SCHEMAS["teardown"], role="Colab Teardown", type_name="self", model="flash")
    _run_json(OPS, ["state", "clear-active-wave", "--track", str(track)])
    _run_json(OPS, ["state", "record-cu", "--track", str(track), "--trace-id", trace_id, "--iter", str(iter_k), "--connect-ts", str(aw["launched_at"]), "--unassign-ts", str(td["unassign_ts"]), "--unassign-confirmed", "--teardown-mode", "fallback" if td["fallback_unassign_called"] else "remote_trap+dom_delete"])

    out_dir = ROOT / "results" / "track_runs" / f"track_{track}_iter_{iter_k}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_ps = (
        f"$ErrorActionPreference='Stop'; $env:AGY_ACCOUNT='{GDRIVE_ACCOUNT}'; "
        "function Invoke-Retry([scriptblock]$Fn){ for($i=0;$i -lt 3;$i++){ try{ return & $Fn }catch{ if($i -eq 2){throw}; Start-Sleep -Seconds ([int][Math]::Pow(2,$i)) } } }; "
        "function Sync-Folder($fid,$dst){ New-Item -ItemType Directory -Force -Path $dst | Out-Null; "
        "$raw = Invoke-Retry { $o = & gdrive readonly ls $fid --max 0 --json; if($LASTEXITCODE -ne 0){throw 'gdrive ls failed'}; $o -join [Environment]::NewLine }; "
        "$s=$raw.IndexOf('['); $e=$raw.LastIndexOf(']'); if($s -lt 0 -or $e -le $s){return}; "
        "foreach($it in @($raw.Substring($s,$e-$s+1) | ConvertFrom-Json)){ "
        "if($it.mimeType -eq 'application/vnd.google-apps.folder'){ Sync-Folder $it.id (Join-Path $dst $it.name) } "
        "elseif($it.mimeType -notlike 'application/vnd.google-apps.*'){ Invoke-Retry { & gdrive readonly download $it.id --out (Join-Path $dst $it.name); if($LASTEXITCODE -ne 0){throw 'gdrive download failed'} } } } }; "
        f"Sync-Folder '{folder_id}' '{out_dir}'"
    )
    await call_tool("run_command", CommandLine=sync_ps, WaitMsBeforeAsync=10000, toolSummary="Pull Colab track artifacts")
    return stalled, out_dir, cur_cells


async def _finalize_wave(
    track: int, iter_k: int, trace_id: str, run_id: str, state: dict, cfg: dict,
    out_dir: Path, init_cells: str, cur_cells: str, ver_json: str, spec_rel: str,
    impl_sha: str, hypothesis: str, preamble: str,
) -> dict | None:
    phase(f"Iter {iter_k}: RATCHET_DECIDE")
    _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "RATCHET_DECIDE", "--trace-id", trace_id, "--workflow-run-id", run_id])
    au_args = ["--mode", "post-colab", "--track", str(track), "--iter", str(iter_k), "--results-dir", str(out_dir)]
    if init_cells and cur_cells and os.path.isfile(init_cells) and os.path.isfile(cur_cells):
        au_args += ["--initial-cells-json", init_cells, "--current-cells-json", cur_cells]
    c8_rc, c8_res, _ = _run_json(AUDITOR, au_args, check=False)
    probe_path = out_dir / "probe_ablation_summary.json"
    probe_sum = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
    promoted, d_rep = False, {}
    if c8_rc == 0 and c8_res.get("passed") and probe_sum.get("pass_stage_a") and (out_dir / "predictions.jsonl").exists():
        diff_args = ["--candidate-preds", str(out_dir / "predictions.jsonl"), "--candidate-summary", str(out_dir / "summary_metrics.json"), "--best-f1", str(state["best_f1"]), "--best-sub", str(state["best_sub"]), "--probe-summary", str(probe_path), "--colab-telemetry", str(out_dir / "colab_run_telemetry.json"), "--out-json", str(out_dir / "forensic_delta.json"), "--out-md", str(out_dir / "forensic_delta.md")]
        if ver_json and os.path.isfile(ver_json):
            diff_args += ["--provenance-json", ver_json]
        _run_cli(DIFF_CLI, diff_args)
        d_rep = json.loads((out_dir / "forensic_delta.json").read_text(encoding="utf-8"))
        promoted = bool(d_rep.get("ratchet", {}).get("promoted", False))
    if promoted:
        r_info = d_rep["ratchet"]
        _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "RATCHET_DECIDE", "--updates-json", json.dumps({"iteration": iter_k, "consecutive_no_gain": 0, "best_commit": impl_sha, "best_f1": r_info["cand_f1"], "best_sub": r_info["cand_sub"], "best_comp": r_info["cand_comp"], "best_probe_70_comp": probe_sum.get("full_comp_70", state["best_probe_70_comp"])})])
        _run_json(OPS, ["state", "append-epistemic", "--track", str(track), "--iter", str(iter_k), "--outcome", "PROMOTED", "--hypothesis", hypothesis, "--dead-end-rule", "", "--detail", f"Promoted {impl_sha[:7]}: Comp={r_info['cand_comp']}% (d={r_info['d_comp']:+.2f})"])
        return None
    _run_json(OPS, ["git", "rollback-clean", "--track", str(track), "--best-commit", state["best_commit"], "--iter", str(iter_k)])
    new_ng = int(state.get("consecutive_no_gain", 0)) + 1
    _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "RATCHET_DECIDE", "--updates-json", json.dumps({"iteration": iter_k, "consecutive_no_gain": new_ng})])
    fail_msg = f"StageA={probe_sum.get('loo_decision')} hint={probe_sum.get('submask_hint')} ratchet={d_rep.get('ratchet', {}).get('reasons', [])}"
    _run_json(OPS, ["state", "append-epistemic", "--track", str(track), "--iter", str(iter_k), "--outcome", "NO_GAIN", "--hypothesis", hypothesis, "--dead-end-rule", fail_msg[:140], "--detail", fail_msg])
    if new_ng < int(cfg["budgets"]["no_gain_halt_threshold"]):
        if new_ng == 1:
            phase(f"Iter {iter_k}: MINI_BOOST")
            _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "MINI_BOOST", "--trace-id", trace_id, "--workflow-run-id", run_id])
            f_path = out_dir / "forensic_delta.md"
            spec_exc = (ROOT / spec_rel).read_text(encoding="utf-8")[:4000] if (ROOT / spec_rel).is_file() else ""
            try:
                mb = await agent(_load_prompt("mini_boost_triz.md", track_id=track, track_name=cfg["track_name"], iter_k=iter_k, next_iter_k=iter_k + 1, epistemic_preamble=preamble, spec_path=spec_rel, spec_excerpt=spec_exc, forensic_md_path=str(f_path), forensic_excerpt=f_path.read_text(encoding="utf-8")[:4000] if f_path.exists() else fail_msg), SCHEMAS["mini_boost"], role=f"Mini-Boost Iter {iter_k}", type_name="self", model="inherit")
            except Exception as exc:
                return await _pause_infra(track, f"Mini-Boost agent fault: {exc}", trace_id)
            _run_json(OPS, ["state", "append-epistemic", "--track", str(track), "--iter", str(iter_k), "--outcome", "MINI_BOOST_TRIZ", "--hypothesis", mb["contradiction"], "--dead-end-rule", mb["dead_end_rule"], "--detail", f"{mb['root_cause']} | Next: {mb['next_direction_hint']}"])
        return None
    head_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "HALTED_FOR_BOOST", "--updates-json", json.dumps({"halted_head_sha": head_sha})])
    _run_json(OPS, ["report", "boost-handoff", "--track", str(track), "--iter", str(iter_k), "--forensic-md", str(out_dir / "forensic_delta.md"), "--probe-summary", str(out_dir / "probe_ablation_summary.json")])
    return {"status": "HALTED_FOR_BOOST", "iter": iter_k}


async def main() -> dict:
    track = int(args["track"])
    mode = str(args.get("mode", "run"))
    max_burst = int(args.get("max_burst_iters", 2))
    boost_notes = args.get("boost_notes_path")
    run_id = f"{CONVERSATION_ID}:{STEP_INDEX}"

    phase("PREFLIGHT")
    pre_args = ["preflight", "--track", str(track), "--mode", mode]
    if boost_notes:
        pre_args += ["--boost-notes-path", str(boost_notes)]
    rc, pre, err = _run_json(OPS, pre_args, check=False)
    print(json.dumps(pre, indent=2, ensure_ascii=False))
    if mode in ("preflight", "status"):
        log(f"Preflight ready={pre.get('ready')} action={pre.get('resume_action')}")
        if mode == "preflight" and not pre.get("ready"):
            raise RuntimeError(f"Preflight failed: {pre.get('remediations')}")
        return pre

    if pre.get("resume_action") == "ORPHAN_EXPIRED":
        try:
            aw = pre["state"]["active_wave"]
            cur_cells = str(ROOT / ".workflow_state" / f"track_{track}" / "cells_orphan.json")
            await agent(_load_prompt("colab_teardown.md", track_id=track, wave_id=aw["wave_id"], added_cell_ids_json=json.dumps(aw.get("added_cell_ids") or []), current_cells_path=cur_cells, fallback_unassign="true"), SCHEMAS["teardown"], role="Orphan Teardown", type_name="self", model="flash")
            _run_json(OPS, ["state", "clear-active-wave", "--track", str(track)])
        except Exception as exc:
            return await _pause_infra(track, f"Orphan teardown fault: {exc}")
        return await _pause_infra(track, "Expired active_wave cleaned up; re-run mode=run.")

    if not pre.get("ready"):
        if pre.get("halt_refused"):
            raise RuntimeError("; ".join(pre.get("remediations") or ["Track is HALTED_FOR_BOOST"]))
        return await _pause_infra(track, "; ".join(pre.get("remediations") or [err]))

    cfg, state = pre["config"], pre["state"]
    budgets = cfg["budgets"]

    if pre.get("resume_action") == "COLAB_POLLING":
        phase("COLAB_POLLING (Resume)")
        iter_k = int(state["iteration"]) + 1
        try:
            stalled, out_dir, cur_cells = await _poll_and_teardown(track, iter_k, state, cfg)
        except Exception as exc:
            return await _pause_infra(track, f"Resume polling/teardown fault: {exc}")
        if stalled or (out_dir / "WAVE_STALLED_MARKER").exists():
            return await _pause_infra(track, f"Resumed wave stalled in {out_dir}")
        trace_id = (state.get("active_wave") or {}).get("trace_id") or f"track_{track}_iter_{iter_k}_attempt_1"
        t_dir = ROOT / ".workflow_state" / f"track_{track}" / f"iter_{iter_k}"
        head_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        pream = _run_json(OPS, ["state", "render-epistemic-preamble", "--track", str(track)])[1]["preamble"]
        res = await _finalize_wave(track, iter_k, trace_id, run_id, state, cfg, out_dir, str(t_dir / f"cells_init_{iter_k}.json"), cur_cells, str(t_dir / "core_paper_verification.json"), f"docs/specs/track_{track}/spec_iter_{iter_k}.md", head_sha, f"Resumed wave iter_{iter_k} ({head_sha[:7]})", pream)
        if res is not None:
            return res
        max_burst = max(0, max_burst - 1)

    for _ in range(max_burst):
        state = _run_json(OPS, ["state", "status", "--track", str(track)])[1]
        rem_cu = float(budgets["max_cu"]) - float(state.get("cumulative_track_cu", 0.0))
        if int(state["iteration"]) >= int(budgets["max_iterations"]) or rem_cu < float(budgets["min_stage_b_cu"]):
            break

        iter_k = int(state["iteration"]) + 1
        trace_id = f"track_{track}_iter_{iter_k}_attempt_1"
        track_state_dir = ROOT / ".workflow_state" / f"track_{track}" / f"iter_{iter_k}"
        track_state_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(ROOT / ".workflow_state" / "paper_cache")
        ver_json = str(track_state_dir / "core_paper_verification.json")
        prim_md = str(track_state_dir / f"verified_primitives_iter_{iter_k}.md")
        spec_rel = f"docs/specs/track_{track}/spec_iter_{iter_k}.md"
        forensic_md = str(ROOT / "results" / "track_runs" / f"track_{track}_iter_{iter_k - 1}" / "forensic_delta.md")
        forensic_txt = Path(forensic_md).read_text(encoding="utf-8")[:8192] if os.path.isfile(forensic_md) else "Initial iteration from frozen FRSA-PBCF (011)."

        phase(f"Iter {iter_k}: RESEARCH")
        _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "RESEARCH", "--trace-id", trace_id, "--workflow-run-id", run_id])
        preamble = _run_json(OPS, ["state", "render-epistemic-preamble", "--track", str(track)])[1]["preamble"]
        r = agent(_load_prompt("researcher.md", track_id=track, track_name=cfg["track_name"], base_ref=cfg["base_ref"], max_new_scalars=cfg["max_new_scalars"], single_step_lock=cfg["single_step_lock"], require_zero_kv_diff=cfg["require_zero_kv_diff"], notebooklm_notebook_id=cfg["notebooklm_notebook_id"]), role=f"Researcher Iter {iter_k}", type_name="self", model="inherit")
        try:
            t1 = await r(f"Turn 1 (Iter {iter_k}):\n{preamble}\n\nLatest Forensic Delta:\n{forensic_txt}\n\nSelect 1-3 core papers and target equations.", SCHEMAS["research_t1"])
            ext = await agent(_load_prompt("paper_extractor.md", track_id=track, iter_k=iter_k, cache_dir=cache_dir, core_papers_json=json.dumps(t1["core_papers"], ensure_ascii=False), verified_primitives_path=prim_md, verification_json_path=ver_json), SCHEMAS["paper_extractor"], role=f"Paper Extractor Iter {iter_k}", type_name="self", model="flash")
            p_rc, p_res, _ = _run_json(OPS, ["papers", "verify", "--verification-json", ext["verification_json_path"], "--cache-dir", cache_dir], check=False)
            if p_rc != 0 or not p_res.get("passed"):
                del r
                return await _pause_infra(track, f"Paper provenance failed/needs waiver: {p_res.get('errors')}", trace_id)
            prim_text = Path(ext["verified_primitives_path"]).read_text(encoding="utf-8")[:16000]
            t2 = await r(f"Turn 2 (Iter {iter_k}): Write `{spec_rel}` using `docs/specs/SPEC_TEMPLATE.md` from these verified primitives:\n{prim_text}", SCHEMAS["research_t2"])
            del r
        except Exception as exc:
            return await _pause_infra(track, f"Research MCP/agent error: {exc}", trace_id)

        spec_sha = _run_json(OPS, ["git", "commit-spec", "--track", str(track), "--iter", str(iter_k), "--spec-path", t2["spec_path"]])[1]["spec_sha"]
        spec_lf_sha = _lf_sha256(ROOT / spec_rel)

        phase(f"Iter {iter_k}: IMPLEMENT_AUDIT")
        _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "IMPLEMENT_AUDIT", "--trace-id", trace_id, "--workflow-run-id", run_id])
        tiers = ["flash"] * int(budgets["max_mech_retries_flash"]) + ["inherit"] * int(budgets["max_mech_retries_inherit"])
        audit_ok, prior_errs, impl_sha = False, "None", None
        for idx, tier in enumerate(tiers):
            try:
                await agent(_load_prompt("implementer.md", track_id=track, track_name=cfg["track_name"], iter_k=iter_k, spec_lf_sha256=spec_lf_sha, allowed_mutable_paths=cfg["allowed_mutable_paths"], immutable_paths=cfg["immutable_paths"], max_new_scalars=cfg["max_new_scalars"], single_step_lock=cfg["single_step_lock"], require_zero_kv_diff=cfg["require_zero_kv_diff"], prior_errors=prior_errs), SCHEMAS["implementer"], role=f"Implementer Try {idx + 1}", type_name="self", model=tier)
            except Exception as exc:
                return await _pause_infra(track, f"Implementer error: {exc}", trace_id)
            sc_rc, sc_res, _ = _run_json(OPS, ["git", "verify-scope", "--track", str(track), "--base-sha", spec_sha], check=False)
            if sc_rc != 0 or not sc_res.get("passed"):
                prior_errs = f"Scope violation: {sc_res.get('violations')}"
                continue
            impl_sha = _run_json(OPS, ["git", "commit-impl", "--track", str(track), "--iter", str(iter_k), "--spec-sha", spec_sha])[1]["impl_sha"]
            au_rc, au_res, _ = _run_json(AUDITOR, ["--mode", "pre-colab", "--track", str(track), "--iter", str(iter_k), "--base-ref", cfg["base_ref"], "--spec-sha", spec_sha, "--impl-sha", impl_sha, "--paper-verification-json", ver_json, "--cal-bank-dir", cfg["cal_bank_dir"], "--receipt-dir", str(track_state_dir / "receipts")], check=False)
            if au_rc == 0 and au_res.get("passed"):
                audit_ok = True
                break
            prior_errs = json.dumps(au_res.get("checks", {}), ensure_ascii=False)[:6000]

        if not audit_ok:
            _run_json(OPS, ["git", "rollback-clean", "--track", str(track), "--best-commit", state["best_commit"], "--iter", str(iter_k)])
            _run_json(OPS, ["state", "append-epistemic", "--track", str(track), "--iter", str(iter_k), "--outcome", "MECH_EXHAUSTED", "--hypothesis", t1["hypothesis_summary"], "--dead-end-rule", f"Avoid mechanical failure: {prior_errs[:120]}", "--detail", prior_errs[:240]])
            new_ng = int(state.get("consecutive_no_gain", 0)) + 1
            _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "RATCHET_DECIDE", "--updates-json", json.dumps({"iteration": iter_k, "consecutive_no_gain": new_ng})])
            if new_ng >= int(budgets["no_gain_halt_threshold"]):
                head_sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
                _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "HALTED_FOR_BOOST", "--updates-json", json.dumps({"halted_head_sha": head_sha})])
                _run_json(OPS, ["report", "boost-handoff", "--track", str(track), "--iter", str(iter_k)])
                return {"status": "HALTED_FOR_BOOST", "iter": iter_k}
            continue

        phase(f"Iter {iter_k}: COLAB_LAUNCH & POLLING")
        _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "COLAB_LAUNCH", "--trace-id", trace_id, "--workflow-run-id", run_id])
        st_ref = _run_json(OPS, ["git", "push-staging", "--track", str(track)])[1]["staging_ref"]
        init_cells = str(track_state_dir / f"cells_init_{iter_k}.json")
        drive_folder = f"{cfg['drive_root']}/track_{track}_iter_{iter_k}_attempt_1"
        try:
            ln = await agent(_load_prompt("colab_launch.md", track_id=track, iter_k=iter_k, trace_id=trace_id, initial_cells_path=init_cells, staging_ref=st_ref, commit_sha=impl_sha, drive_folder=drive_folder, best_probe_comp=state["best_probe_70_comp"], attempt_n=1), SCHEMAS["colab_launch"], role=f"Colab Launch Iter {iter_k}", type_name="self", model="flash")
            state = _run_json(OPS, ["state", "set-active-wave", "--track", str(track), "--wave-id", ln["wave_id"], "--run-folder-id", ln["run_folder_id"], "--added-cell-ids-json", json.dumps(ln["added_cell_ids"]), "--trace-id", trace_id, "--launched-at", str(ln["connect_ts"])])[1]
            _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "COLAB_POLLING", "--trace-id", trace_id, "--workflow-run-id", run_id])
            stalled, out_dir, cur_cells = await _poll_and_teardown(track, iter_k, state, cfg)
        except Exception as exc:
            return await _pause_infra(track, f"Colab launch/poll error: {exc}", trace_id)
        if stalled or (out_dir / "WAVE_STALLED_MARKER").exists():
            return await _pause_infra(track, f"Colab wave stalled; partial artifacts in {out_dir}", trace_id)

        res = await _finalize_wave(track, iter_k, trace_id, run_id, state, cfg, out_dir, init_cells, cur_cells, ver_json, spec_rel, impl_sha, t1["hypothesis_summary"], preamble)
        if res is not None:
            return res

    final_st = _run_json(OPS, ["state", "status", "--track", str(track)])[1]
    rem_cu = float(budgets["max_cu"]) - float(final_st.get("cumulative_track_cu", 0.0))
    if int(final_st["iteration"]) >= int(budgets["max_iterations"]) or rem_cu < float(budgets["min_stage_b_cu"]):
        phase("DONE")
        _run_json(OPS, ["git", "prune-staging", "--track", str(track)], check=False)
        _run_json(OPS, ["state", "transition", "--track", str(track), "--to-state", "DONE", "--workflow-run-id", run_id])
        _run_json(OPS, ["report", "final-report", "--track", str(track)])
        return {"status": "DONE", "iteration": final_st["iteration"], "best_commit": final_st["best_commit"]}

    return {"status": "BURST_COMPLETE", "iteration": final_st["iteration"], "best_commit": final_st["best_commit"]}


result = await main()
log(f"[WORKFLOW EXIT] {json.dumps(result, ensure_ascii=False)}")
