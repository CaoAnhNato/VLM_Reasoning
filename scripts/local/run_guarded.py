#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-Only Host OS Resource & Idle-Progress Governor (scripts/local/run_guarded.py).

Serializes local experiments across swarm teams via ~/.gemini/local_exp.lock,
enforces CUDA_VISIBLE_DEVICES="", <= 2 CPU threads, BELOW_NORMAL_PRIORITY_CLASS,
RSS memory cap (--max-rss-mb), and zero-output idle timeout (--idle-timeout-s).
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import psutil

DEFAULT_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".gemini", "local_exp.lock")
DEFAULT_HEARTBEAT_DIR = os.path.join(
    os.path.expanduser("~"), ".gemini", "vlm_swarm_state", "heartbeats"
)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


@contextmanager
def acquire_os_lock(lock_path: str, timeout_s: float = 90.0) -> Iterator[None]:
    """Acquire exclusive OS file lock using Win32 msvcrt.locking or POSIX fcntl.flock."""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    locked = False
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout_s:.1f}s waiting for lock {lock_path}"
                    )
                time.sleep(0.05)
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def _set_below_normal_priority(pid: int) -> str:
    try:
        proc = psutil.Process(pid)
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "BELOW_NORMAL_PRIORITY_CLASS"
        proc.nice(10)
        return "NICE_10"
    except Exception:
        return "UNKNOWN"


def _get_tree_rss_mb(pid: int) -> float:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0
    total_bytes = 0
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    for p in procs:
        try:
            total_bytes += p.memory_info().rss
        except psutil.Error:
            pass
    return total_bytes / (1024.0 * 1024.0)


def _kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
        for ch in children:
            try:
                ch.kill()
            except psutil.Error:
                pass
        root.kill()
    except psutil.Error:
        pass


def _resolve_command(cmd: List[str]) -> List[str]:
    if not cmd:
        raise ValueError("No command supplied after '--'")
    resolved = list(cmd)
    if resolved[0] == "pytest" and shutil.which("pytest") is None:
        resolved = [sys.executable, "-m", "pytest", *resolved[1:]]
    return resolved


def run_guarded_command(
    cmd: List[str],
    max_rss_mb: float = 2560.0,
    idle_timeout_s: float = 1800.0,
    lock_timeout_s: float = 90.0,
    lock_path: str = DEFAULT_LOCK_PATH,
    heartbeat_dir: str = DEFAULT_HEARTBEAT_DIR,
    receipt_path: Optional[str] = None,
    worker_id: Optional[str] = None,
    allow_gpu: bool = False,
) -> Dict[str, Any]:
    resolved_cmd = _resolve_command(cmd)
    os.makedirs(heartbeat_dir, exist_ok=True)

    env = os.environ.copy()
    if not allow_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["NUMEXPR_MAX_THREADS"] = "2"
    env["TORCH_NUM_THREADS"] = "2"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"

    started_wall = time.time()
    with acquire_os_lock(lock_path, timeout_s=lock_timeout_s):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(
            resolved_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        priority_class = _set_below_normal_priority(proc.pid)

        state_lock = threading.Lock()
        io_state: Dict[str, Any] = {
            "last_output_at": time.monotonic(),
            "last_output_epoch": time.time(),
            "bytes_emitted": 0,
        }

        def _pump_stream(pipe, target_stream) -> None:
            try:
                while True:
                    chunk = pipe.read1(4096) if hasattr(pipe, "read1") else pipe.read(4096)
                    if not chunk:
                        break
                    with state_lock:
                        io_state["last_output_at"] = time.monotonic()
                        io_state["last_output_epoch"] = time.time()
                        io_state["bytes_emitted"] += len(chunk)
                    try:
                        if hasattr(target_stream, "buffer"):
                            target_stream.buffer.write(chunk)
                            target_stream.buffer.flush()
                        else:
                            target_stream.write(chunk.decode("utf-8", errors="replace"))
                            target_stream.flush()
                    except Exception:
                        pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=_pump_stream, args=(proc.stdout, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_pump_stream, args=(proc.stderr, sys.stderr), daemon=True)
        t_out.start()
        t_err.start()

        hb_path = os.path.join(heartbeat_dir, f"{os.getpid()}.json")
        resolved_worker = worker_id or env.get("SWARM_WORKER_ID", "local")
        peak_rss_mb = 0.0
        timed_out_idle = False
        killed_reason: Optional[str] = None
        forced_exit_code: Optional[int] = None

        def _write_hb(status: str) -> None:
            with state_lock:
                last_epoch = io_state["last_output_epoch"]
                bytes_emitted = io_state["bytes_emitted"]
            _atomic_write_json(
                hb_path,
                {
                    "pid": os.getpid(),
                    "child_pid": proc.pid,
                    "worker_id": resolved_worker,
                    "cmd": resolved_cmd,
                    "started_at": started_wall,
                    "last_output_at": last_epoch,
                    "bytes_emitted": bytes_emitted,
                    "peak_rss_mb": round(peak_rss_mb, 2),
                    "status": status,
                },
            )

        _write_hb("running")
        poll_interval = min(0.25, max(0.05, float(idle_timeout_s) / 5.0))
        last_hb_write = time.monotonic()

        while True:
            ret = proc.poll()
            rss_mb = _get_tree_rss_mb(proc.pid)
            if rss_mb > peak_rss_mb:
                peak_rss_mb = rss_mb

            with state_lock:
                idle_elapsed = time.monotonic() - io_state["last_output_at"]

            if ret is not None:
                break

            if rss_mb > max_rss_mb:
                killed_reason = "rss_limit"
                forced_exit_code = 137
                _kill_process_tree(proc.pid)
                proc.wait(timeout=5)
                break

            if idle_elapsed >= idle_timeout_s:
                timed_out_idle = True
                killed_reason = "idle_timeout"
                forced_exit_code = 124
                _kill_process_tree(proc.pid)
                proc.wait(timeout=5)
                break

            now_m = time.monotonic()
            if now_m - last_hb_write >= 0.25:
                _write_hb("running")
                last_hb_write = now_m

            time.sleep(poll_interval)

        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        exit_code = forced_exit_code if forced_exit_code is not None else int(proc.returncode or 0)
        _write_hb("exited" if killed_reason is None else f"killed:{killed_reason}")

    wall_s = round(time.time() - started_wall, 4)
    with state_lock:
        total_bytes = io_state["bytes_emitted"]

    receipt_core: Dict[str, Any] = {
        "cmd": resolved_cmd,
        "exit_code": exit_code,
        "timed_out_idle": timed_out_idle,
        "killed_reason": killed_reason,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "wall_s": wall_s,
        "bytes_emitted": total_bytes,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "omp_num_threads": env.get("OMP_NUM_THREADS", ""),
        "priority_class": priority_class,
        "heartbeat_path": hb_path,
    }
    digest = hashlib.sha256(
        json.dumps(receipt_core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    receipt_core["receipt_sha256"] = digest

    if receipt_path:
        _atomic_write_json(receipt_path, receipt_core)

    return receipt_core


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host CPU-only OS resource and idle-timeout governor."
    )
    parser.add_argument("--max-rss-mb", type=float, default=2560.0)
    parser.add_argument("--idle-timeout-s", type=float, default=1800.0)
    parser.add_argument("--lock-timeout-s", type=float, default=90.0)
    parser.add_argument("--lock-path", type=str, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--heartbeat-dir", type=str, default=DEFAULT_HEARTBEAT_DIR)
    parser.add_argument("--receipt", type=str, default=None)
    parser.add_argument("--worker-id", type=str, default=None)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("Command is required after '--'")
    args.command = cmd
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_guarded_command(
        cmd=args.command,
        max_rss_mb=args.max_rss_mb,
        idle_timeout_s=args.idle_timeout_s,
        lock_timeout_s=args.lock_timeout_s,
        lock_path=args.lock_path,
        heartbeat_dir=args.heartbeat_dir,
        receipt_path=args.receipt,
        worker_id=args.worker_id,
        allow_gpu=args.allow_gpu,
    )
    return int(summary["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
