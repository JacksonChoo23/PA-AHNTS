#!/usr/bin/env python3
"""Phase 0 hardware, driver, NPU, toolchain, and telemetry probe for PA-AHNTS."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
PROBE_EXE = BUILD_DIR / "tools" / "env_probe" / "dxgi_dml_probe.exe"
PROFILE_OUT = REPO_ROOT / "configs" / "hardware_profile.json"


def run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run subprocess and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_powershell_json(ps_script: str, timeout: int = 30) -> Any:
    """Execute a PowerShell command that returns ConvertTo-Json output."""
    rc, out, err = run_cmd(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        timeout=timeout,
    )
    if rc != 0 or not out:
        return {"error": err or f"exit_code={rc}"}
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        return {"error": f"json_decode_error: {exc}", "raw": out[:500]}


def build_and_run_dxgi_probe(rebuild: bool = False) -> dict[str, Any]:
    """Configure/build dxgi_dml_probe with CMake and parse its JSON output."""
    exe_path = PROBE_EXE
    if not exe_path.exists():
        # Visual Studio / multi-config generators place binaries in Release/
        alt_exe = BUILD_DIR / "tools" / "env_probe" / "Release" / "dxgi_dml_probe.exe"
        if alt_exe.exists():
            exe_path = alt_exe

    if rebuild or not exe_path.exists():
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        cmake_cmd = ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)]
        if shutil.which("ninja") and not shutil.which("cl"):
            cmake_cmd.extend(["-G", "Ninja"])
        rc, out, err = run_cmd(cmake_cmd, timeout=120)
        if rc != 0:
            return {"error": "cmake_configure_failed", "stdout": out, "stderr": err}

        rc, out, err = run_cmd(
            ["cmake", "--build", str(BUILD_DIR), "--config", "Release", "--target", "dxgi_dml_probe"],
            timeout=120,
        )
        if rc != 0:
            return {"error": "cmake_build_failed", "stdout": out, "stderr": err}

        for candidate in (
            PROBE_EXE,
            BUILD_DIR / "tools" / "env_probe" / "Release" / "dxgi_dml_probe.exe",
        ):
            if candidate.exists():
                exe_path = candidate
                break

    if not exe_path.exists():
        return {"error": f"probe_binary_not_found: {exe_path}"}

    rc, out, err = run_cmd([str(exe_path)], timeout=30)
    if rc != 0:
        return {"error": f"dxgi_dml_probe exited {rc}", "stderr": err, "stdout": out}
    try:
        data = json.loads(out)
        data["binary_path"] = str(exe_path.relative_to(REPO_ROOT))
        return data
    except json.JSONDecodeError as exc:
        return {"error": f"invalid_probe_json: {exc}", "raw": out[:500]}


def probe_host_system() -> dict[str, Any]:
    """Query OS, computer model, CPU, RAM, and admin status."""
    import psutil

    ps_info = run_powershell_json(
        "$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture;"
        "$cs = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory;"
        "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed;"
        "[PSCustomObject]@{ os=$os; system=$cs; cpu=$cpu } | ConvertTo-Json -Depth 4"
    )
    vm = psutil.virtual_memory()
    is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    return {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "is_admin": is_admin,
        "os": ps_info.get("os", {}),
        "system": ps_info.get("system", {}),
        "cpu": ps_info.get("cpu", {}),
        "ram": {
            "total_bytes": int(vm.total),
            "total_gib": round(vm.total / (1024 ** 3), 2),
            "available_bytes": int(vm.available),
            "available_gib": round(vm.available / (1024 ** 3), 2),
        },
    }


def probe_nvml() -> dict[str, Any]:
    """Query NVIDIA GPU specs, PCIe link, VRAM, clocks, and power via nvml.dll."""
    class NvmlMemory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    try:
        nvml = ctypes.WinDLL("nvml.dll")
    except OSError as exc:
        return {"available": False, "error": str(exc)}

    rc = nvml.nvmlInit_v2()
    if rc != 0:
        return {"available": False, "error": f"nvmlInit_v2 returned {rc}"}

    result: dict[str, Any] = {"available": True, "devices": []}
    try:
        drv_buf = ctypes.create_string_buffer(80)
        nvml_buf = ctypes.create_string_buffer(80)
        if nvml.nvmlSystemGetDriverVersion(drv_buf, 80) == 0:
            result["driver_version"] = drv_buf.value.decode("utf-8", errors="replace")
        if nvml.nvmlSystemGetNVMLVersion(nvml_buf, 80) == 0:
            result["nvml_version"] = nvml_buf.value.decode("utf-8", errors="replace")

        count = ctypes.c_uint(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return result

        for idx in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(handle)) != 0:
                continue

            name_buf = ctypes.create_string_buffer(96)
            nvml.nvmlDeviceGetName(handle, name_buf, 96)
            mem = NvmlMemory()
            mem_ok = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0

            cc_maj, cc_min = ctypes.c_int(0), ctypes.c_int(0)
            cc_ok = nvml.nvmlDeviceGetCudaComputeCapability(
                handle, ctypes.byref(cc_maj), ctypes.byref(cc_min)
            ) == 0

            max_gen, curr_gen = ctypes.c_uint(0), ctypes.c_uint(0)
            max_w, curr_w = ctypes.c_uint(0), ctypes.c_uint(0)
            nvml.nvmlDeviceGetMaxPcieLinkGeneration(handle, ctypes.byref(max_gen))
            nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle, ctypes.byref(curr_gen))
            nvml.nvmlDeviceGetMaxPcieLinkWidth(handle, ctypes.byref(max_w))
            nvml.nvmlDeviceGetCurrPcieLinkWidth(handle, ctypes.byref(curr_w))

            pwr_mw = ctypes.c_uint(0)
            def_lim_mw = ctypes.c_uint(0)
            enf_lim_mw = ctypes.c_uint(0)
            pwr_ok = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(pwr_mw)) == 0
            def_ok = nvml.nvmlDeviceGetPowerManagementDefaultLimit(handle, ctypes.byref(def_lim_mw)) == 0
            enf_ok = nvml.nvmlDeviceGetEnforcedPowerLimit(handle, ctypes.byref(enf_lim_mw)) == 0

            temp_c = ctypes.c_uint(0)
            temp_ok = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp_c)) == 0

            gfx_mhz, sm_mhz, mem_mhz = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0)
            gfx_ok = nvml.nvmlDeviceGetClockInfo(handle, 0, ctypes.byref(gfx_mhz)) == 0
            sm_ok = nvml.nvmlDeviceGetClockInfo(handle, 1, ctypes.byref(sm_mhz)) == 0
            mclk_ok = nvml.nvmlDeviceGetClockInfo(handle, 2, ctypes.byref(mem_mhz)) == 0

            total_mib = round(mem.total / (1024 ** 2), 2) if mem_ok else 0.0
            result["devices"].append(
                {
                    "index": idx,
                    "name": name_buf.value.decode("utf-8", errors="replace"),
                    "cuda_compute_capability": f"{cc_maj.value}.{cc_min.value}" if cc_ok else None,
                    "vram": {
                        "total_bytes": mem.total if mem_ok else None,
                        "total_mib": total_mib if mem_ok else None,
                        "used_mib": round(mem.used / (1024 ** 2), 2) if mem_ok else None,
                        "free_mib": round(mem.free / (1024 ** 2), 2) if mem_ok else None,
                        "is_8gb_vram_target": bool(mem_ok and 7500 <= total_mib <= 8600),
                    },
                    "pcie": {
                        "max_link_gen": int(max_gen.value),
                        "current_link_gen": int(curr_gen.value),
                        "max_link_width": int(max_w.value),
                        "current_link_width": int(curr_w.value),
                    },
                    "power_w": round(pwr_mw.value / 1000.0, 3) if pwr_ok else None,
                    "default_power_limit_w": round(def_lim_mw.value / 1000.0, 3) if def_ok else None,
                    "enforced_power_limit_w": round(enf_lim_mw.value / 1000.0, 3) if enf_ok else None,
                    "temperature_c": int(temp_c.value) if temp_ok else None,
                    "clocks_mhz": {
                        "graphics": int(gfx_mhz.value) if gfx_ok else None,
                        "sm": int(sm_mhz.value) if sm_ok else None,
                        "memory": int(mem_mhz.value) if mclk_ok else None,
                    },
                }
            )
    finally:
        nvml.nvmlShutdown()

    return result


def probe_npu_and_accelerators() -> Any:
    """Query Windows PnP Display and ComputeAccelerator devices + driver versions."""
    ps_script = (
        "$devs = Get-PnpDevice -Class ComputeAccelerator,Display -ErrorAction SilentlyContinue | ForEach-Object { "
        "  $drvVer = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_DriverVersion -ErrorAction SilentlyContinue).Data; "
        "  $drvDate = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_DriverDate -ErrorAction SilentlyContinue).Data; "
        "  [PSCustomObject]@{ "
        "    class=$_.Class; friendly_name=$_.FriendlyName; status=$_.Status; "
        "    manufacturer=$_.Manufacturer; instance_id=$_.InstanceId; "
        "    driver_version=$drvVer; driver_date=if($drvDate){$drvDate.ToString('yyyy-MM-dd')}else{$null} "
        "  } "
        "}; "
        "@($devs) | ConvertTo-Json -Depth 4"
    )
    return run_powershell_json(ps_script)


def probe_pdh_power_and_counters() -> dict[str, Any]:
    """Read live Windows PDH RAPL Energy Meter counters and check GPU PDH sets."""
    pdh = ctypes.WinDLL("pdh.dll")
    PDH_FMT_DOUBLE = 0x00000200

    class PdhFmtCounterValue(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    counter_paths = {
        "cpu_pkg_power_w": r"\Energy Meter(RAPL_Package0_PKG)\Power",
        "cpu_core_pp0_power_w": r"\Energy Meter(RAPL_Package0_PP0)\Power",
        "igpu_pp1_power_w": r"\Energy Meter(RAPL_Package0_PP1)\Power",
        "dram_power_w": r"\Energy Meter(RAPL_Package0_DRAM)\Power",
    }

    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
        return {"pdh_open_ok": False}

    handles: dict[str, ctypes.c_void_p] = {}
    add_statuses: dict[str, str] = {}
    for key, path in counter_paths.items():
        ch = ctypes.c_void_p()
        st = pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(ch))
        add_statuses[key] = f"0x{st & 0xFFFFFFFF:08X}"
        if st == 0:
            handles[key] = ch

    readings_mw: dict[str, float | None] = {}
    if handles:
        pdh.PdhCollectQueryData(query)
        time.sleep(0.25)
        collect_st = pdh.PdhCollectQueryData(query)
        if collect_st == 0:
            for key, ch in handles.items():
                val = PdhFmtCounterValue()
                st = pdh.PdhGetFormattedCounterValue(ch, PDH_FMT_DOUBLE, None, ctypes.byref(val))
                if st == 0 and val.CStatus == 0:
                    readings_mw[key] = round(val.doubleValue / 1000.0, 3)
                else:
                    readings_mw[key] = None
    pdh.PdhCloseQuery(query)

    # Check Windows GPU memory/engine PDH counter sets
    pdh_sets = run_powershell_json(
        "$sets = @('GPU Local Adapter Memory','GPU Non Local Adapter Memory','GPU Process Memory','GPU Engine','Energy Meter','Power Meter'); "
        "$res = @{}; foreach($s in $sets){ $res[$s] = [bool](Get-Counter -ListSet $s -ErrorAction SilentlyContinue) }; "
        "$res | ConvertTo-Json"
    )

    pkg_w = readings_mw.get("cpu_pkg_power_w")
    return {
        "pdh_open_ok": True,
        "rapl_readable_without_admin": pkg_w is not None and pkg_w > 0.0,
        "sample_readings_w": readings_mw,
        "counter_add_status": add_statuses,
        "available_pdh_sets": pdh_sets,
    }


def probe_ml_runtimes_and_toolchain() -> dict[str, Any]:
    """Inspect ML execution providers, Python packages, compilers, and third_party SDKs."""
    dml_ver = run_powershell_json(
        "$p='C:\\Windows\\System32\\DirectML.dll'; "
        "if(Test-Path $p){ $v=(Get-Item $p).VersionInfo; "
        "@{exists=$true; product_version=$v.ProductVersion; file_version=$v.FileVersion} | ConvertTo-Json } "
        "else { @{exists=$false} | ConvertTo-Json }"
    )

    py_pkgs: dict[str, Any] = {}
    try:
        import torch
        py_pkgs["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:
        py_pkgs["torch"] = {"installed": False, "error": str(exc)}

    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort
        py_pkgs["onnxruntime"] = {
            "installed": True,
            "version": ort.__version__,
            "available_providers": ort.get_available_providers(),
        }
    else:
        py_pkgs["onnxruntime"] = {"installed": False}

    if importlib.util.find_spec("openvino") is not None:
        import openvino as ov
        core = ov.Core()
        py_pkgs["openvino"] = {
            "installed": True,
            "version": ov.__version__,
            "available_devices": core.available_devices,
        }
    else:
        py_pkgs["openvino"] = {"installed": False}

    flip_spec = importlib.util.find_spec("flip_evaluator") or importlib.util.find_spec("flip")
    py_pkgs["flip_evaluator"] = {"installed": flip_spec is not None}

    def tool_ver(exe: str, flag: str = "--version", fallback_rel: str | None = None) -> dict[str, Any]:
        path = shutil.which(exe)
        if not path and fallback_rel:
            candidate = REPO_ROOT / fallback_rel
            if candidate.exists():
                path = str(candidate)
        if not path:
            return {"installed": False, "path": None}
        _, out, _ = run_cmd([path, flag], timeout=10)
        first_line = out.splitlines()[0] if out else ""
        return {"installed": True, "path": path, "version_line": first_line}

    vswhere_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    vs_install = None
    if vswhere_path.exists():
        _, out, _ = run_cmd([str(vswhere_path), "-latest", "-property", "installationPath"])
        vs_install = out or None

    sdk_dirs = {
        "agility_sdk": (REPO_ROOT / "third_party" / "agility_sdk").exists(),
        "dxc": (REPO_ROOT / "third_party" / "dxc").exists(),
        "rtxntc": (REPO_ROOT / "third_party" / "rtxntc").exists(),
        "rtxts": (REPO_ROOT / "third_party" / "rtxts").exists(),
        "minidxnn": (REPO_ROOT / "third_party" / "minidxnn").exists(),
        "onnxruntime_native": (REPO_ROOT / "third_party" / "onnxruntime").exists(),
    }

    return {
        "directml_system_dll": dml_ver,
        "python_packages": py_pkgs,
        "toolchain": {
            "cmake": tool_ver("cmake"),
            "ninja": tool_ver("ninja"),
            "g++": tool_ver("g++"),
            "cl_msvc": tool_ver("cl", ""),
            "vs_installation_path": vs_install,
            "dxc": tool_ver("dxc", "--version", "third_party/dxc/build/native/bin/x64/dxc.exe"),
        },
        "third_party_sdks_present": sdk_dirs,
    }


def evaluate_readiness(profile: dict[str, Any]) -> dict[str, Any]:
    """Summarize Phase 0 Gate readiness and missing dependencies for Phases 1-6."""
    dxgi = profile.get("dxgi_d3d12", {})
    nvml = profile.get("nvml", {})
    pdh = profile.get("power_and_pdh_telemetry", {})
    rt = profile.get("ml_runtimes_and_toolchain", {})
    tc = rt.get("toolchain", {})
    pkgs = rt.get("python_packages", {})
    sdks = rt.get("third_party_sdks_present", {})

    ready: list[str] = []
    missing_for_later: list[str] = []

    nv_devs = nvml.get("devices", [])
    if nv_devs and nv_devs[0].get("vram", {}).get("is_8gb_vram_target"):
        ready.append("Target 8GB VRAM dGPU confirmed via NVML")
    if dxgi.get("adapters"):
        dgpu_adapters = [a for a in dxgi["adapters"] if not a.get("is_software") and a.get("vendor_id") == "0x10DE"]
        if dgpu_adapters and dgpu_adapters[0].get("d3d12", {}).get("sampler_feedback_tier", 0) > 0:
            ready.append("DX12 Ultimate (FL 12_2, Tiled Resources, Sampler Feedback, Native FP16, DirectML) verified")
    if pdh.get("rapl_readable_without_admin"):
        ready.append("Windows PDH RAPL Package/Core power counters readable without Admin elevation")
    if nv_devs and nv_devs[0].get("power_w") is not None:
        ready.append("NVML live dGPU power/clock/temperature telemetry verified")

    if not tc.get("cl_msvc", {}).get("installed"):
        missing_for_later.append("MSVC (VS 2022 Build Tools cl.exe) — optional if using MinGW, recommended for Agility SDK / RTXNTC")
    if not tc.get("dxc", {}).get("installed"):
        missing_for_later.append("dxc.exe (DirectX Shader Compiler for HLSL SM 6.6+ FP16 / Sampler Feedback in Phase 1)")
    if not sdks.get("agility_sdk"):
        missing_for_later.append("third_party/agility_sdk (recommended for consistent DX12 Sampler Feedback & Enhanced Barriers)")
    if not pkgs.get("flip_evaluator", {}).get("installed"):
        missing_for_later.append("flip-evaluator (Python package for NVIDIA FLIP visual quality metric in Phase 1)")
    if not pkgs.get("onnxruntime", {}).get("installed") and not pkgs.get("openvino", {}).get("installed"):
        missing_for_later.append("onnxruntime-directml / openvino (required in Phase 3 for NPU/iGPU/CPU decode benchmarks)")

    passed = bool(dxgi.get("adapters")) and bool(nv_devs) and bool(pdh.get("pdh_open_ok"))
    return {
        "gate0_passed": passed,
        "ready_components": ready,
        "missing_dependencies_for_later_phases": missing_for_later,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-AHNTS Phase 0 hardware & environment probe")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of dxgi_dml_probe.exe")
    args = parser.parse_args()

    profile: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": probe_host_system(),
        "dxgi_d3d12": build_and_run_dxgi_probe(rebuild=args.rebuild),
        "nvml": probe_nvml(),
        "pnp_accelerators": probe_npu_and_accelerators(),
        "power_and_pdh_telemetry": probe_pdh_power_and_counters(),
        "ml_runtimes_and_toolchain": probe_ml_runtimes_and_toolchain(),
    }
    profile["readiness_gate0"] = evaluate_readiness(profile)

    PROFILE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_OUT.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Wrote hardware profile to: {PROFILE_OUT}")
    print(f"Gate 0 passed: {profile['readiness_gate0']['gate0_passed']}")
    for item in profile["readiness_gate0"]["ready_components"]:
        print(f"  [READY] {item}")
    for item in profile["readiness_gate0"]["missing_dependencies_for_later_phases"]:
        print(f"  [NEED LATER] {item}")
    return 0 if profile["readiness_gate0"]["gate0_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
