# -*- coding: utf-8 -*-
"""
Unit tests for scripts/local/run_guarded.py (OS Resource & Idle-Timeout Governor).
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_GUARDED = os.path.join(PROJECT_ROOT, "scripts", "local", "run_guarded.py")


class TestRunGuardedGovernor(unittest.TestCase):
    def test_01_env_priority_and_receipt_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_dir = os.path.join(tmp, "hb")
            lock_path = os.path.join(tmp, "test_exp.lock")
            receipt = os.path.join(tmp, "receipt.json")
            probe_code = (
                "import os, psutil; "
                "p = psutil.Process(os.getpid()); "
                "prio = p.nice() if os.name == 'nt' else 'posix'; "
                "print(f\"CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')}|OMP={os.environ.get('OMP_NUM_THREADS')}|PRIO={prio}\")"
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--receipt",
                    receipt,
                    "--",
                    sys.executable,
                    "-c",
                    probe_code,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("CUDA=|OMP=2|", proc.stdout)
            if os.name == "nt":
                import psutil

                self.assertIn(f"PRIO={int(psutil.BELOW_NORMAL_PRIORITY_CLASS)}", proc.stdout)

            data = json.load(open(receipt, "r", encoding="utf-8"))
            self.assertEqual(data["exit_code"], 0)
            self.assertFalse(data["timed_out_idle"])
            self.assertEqual(data["cuda_visible_devices"], "")
            self.assertEqual(data["omp_num_threads"], "2")

            digest_got = data.pop("receipt_sha256")
            expected = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            self.assertEqual(digest_got, expected)

    def test_02_concurrent_lock_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_dir = os.path.join(tmp, "hb")
            lock_path = os.path.join(tmp, "serial.lock")
            log_file = os.path.join(tmp, "events.log")
            child_script = (
                "import sys, time; "
                "tag = sys.argv[1]; log_p = sys.argv[2]; "
                "open(log_p, 'a').write(f'{tag}:start:{time.time():.4f}\\n'); "
                "time.sleep(0.35); "
                "open(log_p, 'a').write(f'{tag}:end:{time.time():.4f}\\n')"
            )
            p1 = subprocess.Popen(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--",
                    sys.executable,
                    "-c",
                    child_script,
                    "A",
                    log_file,
                ]
            )
            time.sleep(0.08)
            p2 = subprocess.Popen(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--",
                    sys.executable,
                    "-c",
                    child_script,
                    "B",
                    log_file,
                ]
            )
            self.assertEqual(p1.wait(timeout=15), 0)
            self.assertEqual(p2.wait(timeout=15), 0)

            lines = [ln.strip().split(":") for ln in open(log_file, "r", encoding="utf-8") if ln.strip()]
            self.assertEqual(len(lines), 4)
            # First process must end before second process starts
            self.assertEqual(lines[0][0], lines[1][0])
            self.assertEqual(lines[1][1], "end")
            self.assertEqual(lines[2][1], "start")
            self.assertLessEqual(float(lines[1][2]), float(lines[2][2]) + 0.02)

    def test_03_rss_memory_cap_kills_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_dir = os.path.join(tmp, "hb")
            lock_path = os.path.join(tmp, "rss.lock")
            receipt = os.path.join(tmp, "rss_receipt.json")
            alloc_code = "import time; buf = bytearray(220 * 1024 * 1024); time.sleep(5)"
            proc = subprocess.run(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--max-rss-mb",
                    "128",
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--receipt",
                    receipt,
                    "--",
                    sys.executable,
                    "-c",
                    alloc_code,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 137)
            data = json.load(open(receipt, "r", encoding="utf-8"))
            self.assertEqual(data["killed_reason"], "rss_limit")

    def test_04_idle_timeout_kills_silent_and_allows_active_printing(self):
        with tempfile.TemporaryDirectory() as tmp:
            hb_dir = os.path.join(tmp, "hb")
            lock_path = os.path.join(tmp, "idle.lock")
            receipt_silent = os.path.join(tmp, "silent.json")
            proc_silent = subprocess.run(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--idle-timeout-s",
                    "1",
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--receipt",
                    receipt_silent,
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(10)",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_silent.returncode, 124)
            silent_data = json.load(open(receipt_silent, "r", encoding="utf-8"))
            self.assertTrue(silent_data["timed_out_idle"])

            receipt_active = os.path.join(tmp, "active.json")
            active_code = (
                "import time; "
                "[print(f'tick {i}', flush=True) or time.sleep(0.3) for i in range(5)]"
            )
            proc_active = subprocess.run(
                [
                    sys.executable,
                    RUN_GUARDED,
                    "--idle-timeout-s",
                    "1",
                    "--lock-path",
                    lock_path,
                    "--heartbeat-dir",
                    hb_dir,
                    "--receipt",
                    receipt_active,
                    "--",
                    sys.executable,
                    "-c",
                    active_code,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_active.returncode, 0, msg=proc_active.stderr)
            active_data = json.load(open(receipt_active, "r", encoding="utf-8"))
            self.assertFalse(active_data["timed_out_idle"])
            hb = json.load(open(active_data["heartbeat_path"], "r", encoding="utf-8"))
            self.assertGreater(hb["bytes_emitted"], 0)


if __name__ == "__main__":
    unittest.main()
