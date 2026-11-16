# -*- coding: utf-8 -*-
"""
Colab Gate B Resource Preflight & Hardware Admission Diagnostic Tool.

Zero model loading, zero network calls, zero package installations.
Inspects physical CUDA VRAM via `torch.cuda.mem_get_info()`, native BF16
Tensor Core capability via `torch.cuda.is_bf16_supported()`, host RAM,
and working volume disk space. Emits structured human-readable diagnostic
and machine-readable JSON output.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple


def get_driver_version() -> Tuple[str, str]:
    """Query NVIDIA driver version via nvidia-smi if available."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0], "MEASURED"
    except Exception:
        pass
    return "UNKNOWN", "NOT_MEASURED"


def probe_fp16_runtime_support(torch_mod: Any, device_index: int, props: Any) -> Tuple[str, str, bool]:
    """
    Verify FP16 support directly in the active CUDA runtime (no lookup-table-only inference).
    Executes tiny FP16 matmul, conv2d, and softmax ops on the target CUDA device (zero weights).
    """
    cc_str = f"{props.major}.{props.minor}"
    has_native_fp16_arith = (props.major > 5) or (props.major == 5 and props.minor >= 3)
    has_fp16_tensor_cores = (props.major >= 7)
    if not has_native_fp16_arith:
        return "FAIL", f"Compute Capability {cc_str} (< 5.3) lacks native FP16 arithmetic", False

    try:
        dev = f"cuda:{device_index}"
        with torch_mod.no_grad():
            a = torch_mod.ones((8, 8), device=dev, dtype=torch_mod.float16)
            b = torch_mod.ones((8, 8), device=dev, dtype=torch_mod.float16) * 0.5
            c = torch_mod.matmul(a, b)
            x2 = torch_mod.ones((1, 3, 14, 14), device=dev, dtype=torch_mod.float16)
            w2 = torch_mod.ones((4, 3, 14, 14), device=dev, dtype=torch_mod.float16) * 0.01
            y2 = torch_mod.nn.functional.conv2d(x2, w2, stride=(14, 14))
            sm = torch_mod.softmax(c, dim=-1)
            ok = bool(
                torch_mod.isfinite(c).all().item()
                and torch_mod.isfinite(y2).all().item()
                and torch_mod.isfinite(sm).all().item()
            )
            del a, b, c, x2, w2, y2, sm
        if not ok:
            return "FAIL", f"Runtime FP16 kernel probe produced non-finite output on CC {cc_str}", False
        tc_note = "FP16 Tensor Cores" if has_fp16_tensor_cores else "native FP16 arithmetic"
        return "PASS", f"Runtime FP16 probe passed ({tc_note}, CC {cc_str})", True
    except Exception as exc:
        return "FAIL", f"Runtime FP16 probe failed on CC {cc_str}: {exc}", False


def audit_hardware_and_environment(
    device_index: int = 0,
    precision: str = "bf16",
    allow_bf16_exception: bool = False,
) -> Dict[str, Any]:
    """Execute complete hardware and runtime environment preflight check for selected precision."""
    prec_norm = str(precision).strip().lower()
    if prec_norm not in ("bf16", "fp16"):
        raise ValueError(f"Unsupported precision {precision!r}. Expected 'bf16' or 'fp16'.")

    data: Dict[str, Any] = {
        "requested_precision": prec_norm,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "pytorch": {
            "version": "UNKNOWN",
            "cuda_runtime_version": "UNKNOWN",
        },
        "host_ram": {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "percent_used": 0.0,
            "status": "UNKNOWN",
        },
        "disk": {
            "total_gb": 0.0,
            "free_gb": 0.0,
            "status": "UNKNOWN",
        },
        "gpu": {
            "cuda_available": False,
            "device_count": 0,
            "driver_version": "UNKNOWN",
            "driver_version_status": "NOT_MEASURED",
            "devices": [],
        },
        "admission": {
            "requested_precision": prec_norm,
            "native_bf16_support": None,
            "verdict": "REJECT",
            "checks": {},
            "blocking_reasons": [],
            "conditional_exception": {
                "authorized": bool(allow_bf16_exception),
                "type": "OWNER_AUTHORIZED_T4_BF16_EXCEPTION" if allow_bf16_exception else None,
                "hard_resource_gates_passed": False,
                "proceed_under_exception": False,
            },
        },
    }

    # 1. Host RAM & Disk
    try:
        import psutil

        vm = psutil.virtual_memory()
        data["host_ram"]["total_gb"] = round(vm.total / (1024**3), 2)
        data["host_ram"]["available_gb"] = round(vm.available / (1024**3), 2)
        data["host_ram"]["percent_used"] = float(vm.percent)
        data["host_ram"]["status"] = "MEASURED"
    except ImportError:
        data["host_ram"]["status"] = "NOT_MEASURED"

    try:
        disk = shutil.disk_usage(".")
        data["disk"]["total_gb"] = round(disk.total / (1024**3), 2)
        data["disk"]["free_gb"] = round(disk.free / (1024**3), 2)
        data["disk"]["status"] = "MEASURED"
    except Exception:
        data["disk"]["status"] = "NOT_MEASURED"

    # 2. PyTorch & CUDA
    try:
        import torch

        data["pytorch"]["version"] = str(torch.__version__)
        data["pytorch"]["cuda_runtime_version"] = (
            str(torch.version.cuda) if torch.version.cuda is not None else "CPU_BUILD"
        )
        cuda_avail = bool(torch.cuda.is_available())
        data["gpu"]["cuda_available"] = cuda_avail
    except ImportError:
        torch = None
        cuda_avail = False

    # 3. Driver Version via nvidia-smi
    drv_ver, drv_status = get_driver_version()
    data["gpu"]["driver_version"] = drv_ver
    data["gpu"]["driver_version_status"] = drv_status

    # 4. Device Inspection
    checks = data["admission"]["checks"]
    reasons = data["admission"]["blocking_reasons"]

    checks["cuda_available"] = {
        "passed": cuda_avail,
        "detail": f"CUDA available: {cuda_avail}",
    }
    if not cuda_avail:
        reasons.append("CUDA is not available in PyTorch environment.")

    if torch is not None and cuda_avail:
        dev_count = torch.cuda.device_count()
        data["gpu"]["device_count"] = dev_count
        target_idx = max(0, min(device_index, dev_count - 1))

        for idx in range(dev_count):
            props = torch.cuda.get_device_properties(idx)
            # Physical device memory query
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                free_gb = round(free_bytes / (1024**3), 2)
                total_gb = round(total_bytes / (1024**3), 2)
                mem_status = "MEASURED"
            except Exception as e:
                free_gb = 0.0
                total_gb = round(props.total_memory / (1024**3), 2)
                mem_status = f"FAILED: {e}"

            # PyTorch internal tracking
            alloc_gb = round(torch.cuda.memory_allocated(idx) / (1024**3), 4)
            res_gb = round(torch.cuda.memory_reserved(idx) / (1024**3), 4)

            # Native BF16 check (rejecting emulation)
            bf16_supported = "UNKNOWN"
            bf16_detail = "API unavailable"
            bf16_bool = False
            if hasattr(torch.cuda, "is_bf16_supported"):
                try:
                    # Explicitly require native hardware Tensor Core support (no emulation)
                    is_native = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                    bf16_bool = is_native
                    bf16_supported = "PASS" if is_native else "FAIL"
                    bf16_detail = "Native hardware Tensor Core BF16" if is_native else "Emulation or unsupported"
                except TypeError:
                    # Fallback if older torch doesn't accept including_emulation
                    # Compute capability >= 8.0 (Ampere/Ada/Hopper) has native BF16 Tensor Cores
                    is_native = (props.major >= 8)
                    bf16_bool = is_native
                    bf16_supported = "PASS" if is_native else "FAIL"
                    bf16_detail = f"Derived from compute capability {props.major}.{props.minor} (>=8.0 required)"
                except Exception as e:
                    bf16_supported = "UNKNOWN"
                    bf16_detail = f"Check failed: {e}"
            else:
                is_native = (props.major >= 8)
                bf16_bool = is_native
                bf16_supported = "PASS" if is_native else "FAIL"
                bf16_detail = f"Derived from compute capability {props.major}.{props.minor}"

            fp16_supported, fp16_detail, fp16_bool = probe_fp16_runtime_support(torch, idx, props)

            dev_info = {
                "index": idx,
                "name": str(props.name),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_gb": total_gb,
                "physical_free_vram_gb": free_gb,
                "pytorch_allocated_gb": alloc_gb,
                "pytorch_reserved_gb": res_gb,
                "memory_metric_status": mem_status,
                "native_bf16_support": bf16_supported,
                "native_bf16_support_bool": bf16_bool,
                "native_bf16_detail": bf16_detail,
                "fp16_hardware_support": fp16_supported,
                "fp16_hardware_support_bool": fp16_bool,
                "fp16_detail": fp16_detail,
                "multi_processor_count": int(props.multi_processor_count),
            }
            data["gpu"]["devices"].append(dev_info)

        # Evaluate target device for admission
        target_dev = data["gpu"]["devices"][target_idx]
        total_vram = target_dev["total_vram_gb"]
        free_vram = target_dev["physical_free_vram_gb"]
        bf16_ok = (target_dev["native_bf16_support"] == "PASS")
        fp16_ok = (target_dev["fp16_hardware_support"] == "PASS")
        data["admission"]["native_bf16_support"] = bf16_ok

        checks["total_vram_ge_8gb"] = {
            "passed": bool(total_vram >= 8.0),
            "detail": f"Total VRAM: {total_vram:.2f} GB (minimum 8.0 GB required)",
        }
        if total_vram < 8.0:
            reasons.append(f"Insufficient total VRAM: {total_vram:.2f} GB < 8.00 GB minimum.")

        checks["physical_free_vram_ge_6gb"] = {
            "passed": bool(free_vram >= 6.0),
            "detail": f"Physical Free VRAM: {free_vram:.2f} GB (minimum 6.0 GB required for pilot)",
        }
        if free_vram < 6.0:
            reasons.append(f"Insufficient physical free VRAM: {free_vram:.2f} GB < 6.00 GB required.")

        if prec_norm == "fp16":
            checks["fp16_hardware_support"] = {
                "passed": fp16_ok,
                "detail": f"FP16 Hardware/Runtime: {target_dev['fp16_hardware_support']} ({target_dev['fp16_detail']})",
            }
            if not fp16_ok:
                reasons.append(f"FP16 runtime probe failed on {target_dev['name']} ({target_dev['compute_capability']}): {target_dev['fp16_detail']}")
        else:
            checks["native_bf16_support"] = {
                "passed": bf16_ok,
                "detail": f"Native BF16: {target_dev['native_bf16_support']} ({target_dev['native_bf16_detail']})",
            }
            if not bf16_ok:
                reasons.append(f"Native hardware BF16 is not supported on {target_dev['name']} ({target_dev['compute_capability']}).")
    else:
        data["admission"]["native_bf16_support"] = False
        checks["total_vram_ge_8gb"] = {"passed": False, "detail": "NOT_MEASURED (No CUDA)"}
        checks["physical_free_vram_ge_6gb"] = {"passed": False, "detail": "NOT_MEASURED (No CUDA)"}
        if prec_norm == "fp16":
            checks["fp16_hardware_support"] = {"passed": False, "detail": "NOT_MEASURED (No CUDA)"}
        else:
            checks["native_bf16_support"] = {"passed": False, "detail": "NOT_MEASURED (No CUDA)"}

    # Host RAM check (>= 8.0 GB)
    host_ram_ok = (data["host_ram"]["total_gb"] >= 8.0) if data["host_ram"]["status"] == "MEASURED" else False
    checks["host_ram_ge_8gb"] = {
        "passed": host_ram_ok,
        "detail": f"Host RAM: {data['host_ram']['total_gb']} GB (minimum 8.0 GB required)",
    }
    if not host_ram_ok and data["host_ram"]["status"] == "MEASURED":
        reasons.append(f"Host RAM {data['host_ram']['total_gb']} GB is below 8.0 GB threshold.")

    # Disk check (>= 15.0 GB free)
    disk_ok = (data["disk"]["free_gb"] >= 15.0) if data["disk"]["status"] == "MEASURED" else False
    checks["disk_free_ge_15gb"] = {
        "passed": disk_ok,
        "detail": f"Working volume free disk: {data['disk']['free_gb']} GB (minimum 15.0 GB required)",
    }
    if not disk_ok and data["disk"]["status"] == "MEASURED":
        reasons.append(f"Free disk space {data['disk']['free_gb']} GB is below 15.0 GB required for weights/cache.")

    # Overall Verdict (BF16 without native hardware support ALWAYS remains REJECT)
    all_passed = all(c["passed"] for c in checks.values())
    data["admission"]["verdict"] = "PASS" if all_passed else "REJECT"

    # Hard resource gates (never overridden by conditional BF16 exception)
    hard_gate_keys = [
        "cuda_available",
        "total_vram_ge_8gb",
        "physical_free_vram_ge_6gb",
        "host_ram_ge_8gb",
        "disk_free_ge_15gb",
    ]
    hard_gates_passed = all(bool(checks.get(k, {}).get("passed")) for k in hard_gate_keys)
    exc_info = data["admission"]["conditional_exception"]
    exc_info["hard_resource_gates_passed"] = hard_gates_passed
    if prec_norm == "bf16" and allow_bf16_exception and hard_gates_passed:
        exc_info["proceed_under_exception"] = True
    else:
        exc_info["proceed_under_exception"] = False

    return data


def format_human_readable_report(data: Dict[str, Any]) -> str:
    """Format structured diagnostic output for console and Colab output cells."""
    lines = [
        "=" * 82,
        " SG-VCD GATE B COLAB RUNTIME & HARDWARE PREFLIGHT AUDIT",
        "=" * 82,
        f" OS: {data['os']['system']} {data['os']['release']} (Build: {data['os']['version']})",
        f" Python: {data['python']['version']} | Executable: {data['python']['executable']}",
        f" PyTorch: {data['pytorch']['version']} | PyTorch CUDA Runtime: {data['pytorch']['cuda_runtime_version']}",
        f" NVIDIA Driver: {data['gpu']['driver_version']} ({data['gpu']['driver_version_status']})",
        "-" * 82,
        f" Host System RAM: {data['host_ram']['total_gb']} GB Total | {data['host_ram']['available_gb']} GB Available ({data['host_ram']['percent_used']}% used) [{data['host_ram']['status']}]",
        f" Working Volume Disk: {data['disk']['total_gb']} GB Total | {data['disk']['free_gb']} GB Free [{data['disk']['status']}]",
        "-" * 82,
        f" CUDA Available: {data['gpu']['cuda_available']} | Device Count: {data['gpu']['device_count']}",
    ]

    if data["gpu"]["devices"]:
        for dev in data["gpu"]["devices"]:
            lines.extend([
                f"   [GPU {dev['index']}] {dev['name']} (Compute Capability: {dev['compute_capability']})",
                f"          Total Physical VRAM:   {dev['total_vram_gb']:.2f} GB",
                f"          Actual Free VRAM:       {dev['physical_free_vram_gb']:.2f} GB ({dev['memory_metric_status']})",
                f"          PyTorch Allocated:     {dev['pytorch_allocated_gb']:.4f} GB (Current process active tensors)",
                f"          PyTorch Reserved:      {dev['pytorch_reserved_gb']:.4f} GB (Current process pooled cache)",
                f"          Native BF16 Support:   {dev['native_bf16_support']} ({dev['native_bf16_detail']})",
                f"          FP16 Runtime Support:  {dev.get('fp16_hardware_support', 'UNKNOWN')} ({dev.get('fp16_detail', 'N/A')})",
            ])
    else:
        lines.append("   [GPU] No CUDA GPU devices detected or CUDA is disabled in this runtime.")

    lines.extend([
        "=" * 82,
        f" ADMISSION VERDICT & INDIVIDUAL CRITERIA (PRECISION: {data.get('requested_precision', 'bf16').upper()})",
        "=" * 82,
    ])

    for k, v in data["admission"]["checks"].items():
        status_tag = "[PASS]" if v["passed"] else "[FAIL]"
        lines.append(f" {status_tag:<8} {k:<28} : {v['detail']}")

    verdict = data["admission"]["verdict"]
    lines.extend([
        "-" * 82,
        f" OVERALL ADMISSION VERDICT: {verdict}",
    ])

    exc_info = data["admission"].get("conditional_exception", {})
    if exc_info.get("authorized"):
        lines.append(
            f" OWNER BF16 EXCEPTION: authorized={exc_info.get('authorized')} | "
            f"hard_gates={exc_info.get('hard_resource_gates_passed')} | "
            f"proceed_under_exception={exc_info.get('proceed_under_exception')}"
        )

    if data["admission"]["blocking_reasons"]:
        lines.append(" BLOCKING REASONS / DISQUALIFYING CONDITIONS:")
        for r in data["admission"]["blocking_reasons"]:
            lines.append(f"   * {r}")
    else:
        dtype_name = "torch.float16" if data.get("requested_precision") == "fp16" else "torch.bfloat16"
        lines.append(f" Status: All admission criteria satisfied for 5CD-AI/Vintern-1B-v3_5 ({dtype_name}).")

    lines.append("=" * 82)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="SG-VCD Gate B Hardware Preflight & Precision Admission Diagnostic Tool."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output instead of human-readable text.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to audit (default: 0).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Target precision mode to validate admission for (bf16 or fp16).",
    )
    parser.add_argument(
        "--allow-bf16-exception",
        action="store_true",
        help="Record Owner-authorized T4 BF16 hardware exception when only native BF16 fails and all hard resource gates pass.",
    )
    args = parser.parse_args()

    audit_data = audit_hardware_and_environment(
        device_index=args.device,
        precision=args.precision,
        allow_bf16_exception=args.allow_bf16_exception,
    )

    if args.json:
        print(json.dumps(audit_data, indent=2))
    else:
        print(format_human_readable_report(audit_data))

    exc = audit_data["admission"].get("conditional_exception", {})
    allowed = (audit_data["admission"]["verdict"] == "PASS") or bool(exc.get("proceed_under_exception", False))
    # Exit with code 0 if PASS or Owner-authorized exception with hard resource gates satisfied
    sys.exit(0 if allowed else 1)


if __name__ == "__main__":
    main()

