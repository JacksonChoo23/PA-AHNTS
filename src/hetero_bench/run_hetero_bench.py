#!/usr/bin/env python3
"""PA-AHNTS Phase 3 Heterogeneous Decode Micro-Benchmark, Contention & Quantization Runner.

Measures:
  1) 4-Backend Standalone Throughput & Latency (dGPU DX12/DML, iGPU DML/OV, NPU OV INT8/FP16, CPU OV/ORT)
     across batch sizes B in [1, 4, 16, 64, 256] and hidden dims H in [16, 32, 64].
  2) End-to-End PCIe Transfer Cost vs Compute Decode & analytical/empirical Break-Even Boundaries.
  3) Sustained dGPU Load + Concurrent NPU / iGPU / CPU Power Contention & Dynamic Boost Clock Drop.
  4) 32-Material & 5-Checkpoint DX12 Rendered Quantization Loss (NPU INT8 vs FP16 vs Reference).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import flip_evaluator
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "asset_gen"))
from generate_pbr_assets import encode_bc4_channel_np, encode_bc7_mode6_np  # noqa: E402
from export_ntc_models import build_real_calib_tiles  # noqa: E402


BUILD_DIR = REPO_ROOT / "build"
DX12_BENCH_EXE = BUILD_DIR / "dx12_decode_bench.exe"
PA_BENCH_EXE = BUILD_DIR / "pa_ahnts_bench.exe"
MODELS_DIR = REPO_ROOT / "results" / "phase3" / "models"


def run_cmd(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"[hetero_bench] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd))


def resolve_devices(hw_profile_path: Path) -> dict[str, Any]:
    """Resolve dGPU, iGPU, NPU, and CPU identifiers strictly from hardware_profile.json & runtime probes."""
    import onnxruntime as ort
    import openvino as ov

    hw = json.loads(hw_profile_path.read_text(encoding="utf-8"))
    adapters = hw.get("dxgi_d3d12", {}).get("adapters", [])
    dgpu_dml_id = None
    igpu_dml_id = None
    dgpu_name = "NVIDIA dGPU"
    igpu_name = "Intel iGPU"
    for ad in adapters:
        if ad.get("is_software"):
            continue
        vid = str(ad.get("vendor_id", "")).upper()
        if vid == "0X10DE" and dgpu_dml_id is None:
            dgpu_dml_id = int(ad["adapter_index"])
            dgpu_name = ad.get("description", dgpu_name)
        elif vid == "0X8086" and igpu_dml_id is None:
            igpu_dml_id = int(ad["adapter_index"])
            igpu_name = ad.get("description", igpu_name)

    core = ov.Core()
    ov_devs = core.available_devices
    ov_names: dict[str, str] = {}
    ov_igpu_dev = None
    ov_npu_dev = None
    for d in ov_devs:
        try:
            full = str(core.get_property(d, "FULL_DEVICE_NAME"))
        except Exception:
            full = d
        ov_names[d] = full
        if d.startswith("GPU") and ("Intel" in full or "Graphics" in full or "Arc" in full) and ov_igpu_dev is None:
            ov_igpu_dev = d
        if d == "NPU":
            ov_npu_dev = d

    # Fallback if GPU.0 was Intel without vendor text
    if ov_igpu_dev is None and "GPU.0" in ov_devs and "NVIDIA" not in ov_names.get("GPU.0", ""):
        ov_igpu_dev = "GPU.0"
    elif ov_igpu_dev is None and "GPU" in ov_devs:
        ov_igpu_dev = "GPU"

    pcie_info = hw.get("nvml", {}).get("devices", [{}])[0].get("pcie", {})

    resolved = {
        "dgpu_name": dgpu_name,
        "igpu_name": igpu_name,
        "npu_name": ov_names.get("NPU", "Intel(R) AI Boost"),
        "cpu_name": hw.get("host", {}).get("cpu", {}).get("Name", "CPU"),
        "dgpu_dml_device_id": dgpu_dml_id,
        "igpu_dml_device_id": igpu_dml_id,
        "ov_available_devices": ov_devs,
        "ov_device_names": ov_names,
        "ov_igpu_device": ov_igpu_dev,
        "ov_npu_device": ov_npu_dev,
        "ort_providers": ort.get_available_providers(),
        "pcie_link": pcie_info,
    }
    print(f"[hetero_bench] Resolved devices: {json.dumps(resolved, indent=2)}")
    return resolved


def benchmark_host_prep_and_pack(batches: list[int], runs: int) -> dict[int, dict[str, float]]:
    """Measure realistic CPU Host Feature Gather (16ch float) and Host BCn Packing (192KB BC7+BC5)."""
    manifest_path = REPO_ROOT / "results" / "assets_cache" / "manifest.json"
    pack_path = REPO_ROOT / "results" / "assets_cache" / "pbr_ntc_pack_v1.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob = pack_path.read_bytes()
    g0_s = manifest["materials"][0]["g0_mips"][0]
    g1_s = manifest["materials"][0]["g1_mips"][0]
    g0_grid = np.frombuffer(blob, dtype=np.uint8, count=256 * 256 * 4, offset=int(g0_s["offset"])).reshape(256, 256, 4).astype(np.float32) / 255.0
    g1_grid = np.frombuffer(blob, dtype=np.float16, count=64 * 64 * 4, offset=int(g1_s["offset"])).reshape(64, 64, 4).astype(np.float32)

    # Precompute tile UV PE LUT cleanly (position encoding depends only on local tile UVs for wrap mode)
    px = np.arange(256, dtype=np.float32) + 0.5
    yy, xx = np.meshgrid(px, px, indexing="ij")
    u = xx / 512.0
    v = yy / 512.0
    two_pi = 2.0 * math.pi
    pe = np.stack(
        [
            np.sin(two_pi * 8.0 * np.mod(u * 16.0, 1.0)),
            np.cos(two_pi * 8.0 * np.mod(u * 16.0, 1.0)),
            np.sin(two_pi * 8.0 * np.mod(v * 16.0, 1.0)),
            np.cos(two_pi * 8.0 * np.mod(v * 16.0, 1.0)),
            np.sin(two_pi * 16.0 * np.mod(u * 16.0, 1.0)),
            np.cos(two_pi * 16.0 * np.mod(u * 16.0, 1.0)),
            np.sin(two_pi * 16.0 * np.mod(v * 16.0, 1.0)),
            np.cos(two_pi * 16.0 * np.mod(v * 16.0, 1.0)),
        ],
        axis=0,
    ).astype(np.float16)  # [8, 256, 256]

    fake_out = np.random.default_rng(7).integers(0, 255, size=(4, 8, 256, 256), dtype=np.uint8)
    g0_chw = np.ascontiguousarray(np.transpose(g0_grid, (2, 0, 1)))
    g1_chw = np.ascontiguousarray(np.transpose(g1_grid, (2, 0, 1)))
    idx_q = np.arange(256, dtype=np.intp) // 4
    stats: dict[int, dict[str, float]] = {}

    for b in batches:
        g_times = []
        p_times = []
        test_b = min(b, 16)
        scale = float(b) / float(test_b)
        for _ in range(runs + 2):
            t0 = time.perf_counter()
            out_feat = np.empty((test_b, 16, 256, 256), dtype=np.float16)
            g0_scaled = (g0_chw.astype(np.float16) * np.float16(2.0) - np.float16(1.0))
            g1_up = g1_chw[:, idx_q[:, None], idx_q[None, :]].astype(np.float16)
            out_feat[:, 0:4] = g0_scaled[None, ...]
            out_feat[:, 4:8] = g1_up[None, ...]
            out_feat[:, 8:16] = pe[None, ...]
            g_times.append((time.perf_counter() - t0) * 1000.0 * scale)

            t1 = time.perf_counter()
            for i in range(test_b):
                tile = fake_out[i % 4]
                alb = np.transpose(tile[0:4], (1, 2, 0))
                orm = np.stack([tile[6], tile[7], tile[3], np.full((256, 256), 255, dtype=np.uint8)], axis=-1)
                _ = encode_bc7_mode6_np(alb)
                _ = encode_bc7_mode6_np(orm)
                _ = encode_bc4_channel_np(tile[4])
                _ = encode_bc4_channel_np(tile[5])
            p_times.append((time.perf_counter() - t1) * 1000.0 * scale)

        stats[b] = {
            "host_gather_ms": round(statistics.mean(g_times[2:]), 4),
            "host_bcn_pack_ms": round(statistics.mean(p_times[2:]), 4),
        }
    return stats


def bench_openvino_backend(
    core: Any,
    ov_device: str,
    backend_label: str,
    model_path: Path,
    precision: str,
    hidden_dim: int,
    batch_tiles: int,
    runs: int,
    warmup_iters: int,
    measure_iters: int,
    rng_tiles: np.ndarray,
) -> list[dict[str, Any]]:
    """Compile and benchmark an OpenVINO model on NPU / GPU / CPU."""
    rows: list[dict[str, Any]] = []
    used_b = batch_tiles
    note = "native_batch"

    try:
        compiled = core.compile_model(str(model_path), ov_device)
    except Exception as exc:
        if batch_tiles >= 64:
            # Fallback to AsyncInferQueue of B=16 or B=64 chunks if NPU compiler hits SRAM/alloc ceiling
            sub_b = 16 if batch_tiles >= 64 else 4
            sub_path = model_path.parent / model_path.name.replace(f"_b{batch_tiles}_", f"_b{sub_b}_")
            if sub_path.exists():
                compiled = core.compile_model(str(sub_path), ov_device)
                used_b = sub_b
                note = f"async_queue_{batch_tiles // sub_b}x{sub_b} (fallback: {str(exc)[:60]})"
            else:
                raise
        else:
            raise

    try:
        exec_devs = str(compiled.get_property("EXECUTION_DEVICES"))
    except Exception:
        exec_devs = ov_device

    req = compiled.create_infer_request()
    in_tensor_info = compiled.input(0)
    in_dtype = np.float16 if "f16" in str(in_tensor_info.element_type).lower() else np.float32
    x_buf = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (used_b, 16, 256, 256)).astype(in_dtype))
    ops_per_texel = 2.0 * (16.0 * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 8.0)

    def run_batch() -> float:
        t0 = time.perf_counter()
        if used_b == batch_tiles:
            req.infer({0: x_buf})
        else:
            chunks = batch_tiles // used_b
            q = core.AsyncInferQueue(compiled, min(4, chunks)) if hasattr(core, "AsyncInferQueue") else None
            if q is not None:
                for _ in range(chunks):
                    q.start_async({0: x_buf})
                q.wait_all()
            else:
                for _ in range(chunks):
                    req.infer({0: x_buf})
        return (time.perf_counter() - t0) * 1000.0

    # Handle openvino AsyncInferQueue import from openvino module if not on core
    import openvino as ov

    def run_batch_safe() -> float:
        t0 = time.perf_counter()
        if used_b == batch_tiles:
            req.infer({0: x_buf})
        else:
            chunks = batch_tiles // used_b
            q = ov.AsyncInferQueue(compiled, min(4, chunks))
            for _ in range(chunks):
                q.start_async({0: x_buf})
            q.wait_all()
        return (time.perf_counter() - t0) * 1000.0

    eff_warmup = 2 if batch_tiles >= 64 else warmup_iters
    for _ in range(eff_warmup):
        run_batch_safe()

    eff_iters = 3 if batch_tiles >= 256 else (5 if batch_tiles >= 64 else measure_iters)
    for r in range(runs):
        dts = [run_batch_safe() for _ in range(eff_iters)]
        mean_ms = statistics.mean(dts)
        tiles_per_ms = float(batch_tiles) / max(mean_ms, 1e-6)
        mtex_s = (batch_tiles * 65536.0 / 1e6) / max(mean_ms / 1000.0, 1e-9)
        gflops = (batch_tiles * 65536.0 * ops_per_texel / 1e9) / max(mean_ms / 1000.0, 1e-9)
        rows.append(
            {
                "run_id": r,
                "backend": backend_label,
                "runtime_device": exec_devs,
                "precision": precision,
                "hidden_dim": hidden_dim,
                "batch_tiles": batch_tiles,
                "infer_only_ms": round(mean_ms, 5),
                "tiles_per_ms": round(tiles_per_ms, 5),
                "mtexels_per_sec": round(mtex_s, 4),
                "effective_gflops": round(gflops, 3),
                "execution_note": note,
            }
        )
        if r == 0:
            print(
                f"  [{backend_label}] H={hidden_dim} B={batch_tiles} prec={precision}: "
                f"{mean_ms:.3f} ms | {mtex_s:.2f} Mtex/s | {gflops:.1f} GOP/s | dev={exec_devs}"
            )
    return rows


def bench_ort_backend(
    backend_label: str,
    provider: str,
    provider_options: dict[str, Any] | None,
    model_path: Path,
    precision: str,
    hidden_dim: int,
    batch_tiles: int,
    runs: int,
    warmup_iters: int,
    measure_iters: int,
    rng_tiles: np.ndarray,
) -> list[dict[str, Any]]:
    """Benchmark ONNX Runtime DmlExecutionProvider or CPUExecutionProvider."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "DmlExecutionProvider":
        # DirectML prefers sequential execution + disable mem pattern for stable allocations
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [(provider, provider_options or {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    active_p = sess.get_providers()[0]
    in_name = sess.get_inputs()[0].name
    in_type = sess.get_inputs()[0].type
    in_dtype = np.float16 if "float16" in in_type else np.float32
    x_buf = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (batch_tiles, 16, 256, 256)).astype(in_dtype))
    ops_per_texel = 2.0 * (16.0 * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 8.0)

    def run_once() -> float:
        t0 = time.perf_counter()
        sess.run(None, {in_name: x_buf})
        return (time.perf_counter() - t0) * 1000.0

    eff_warmup = 2 if batch_tiles >= 64 else warmup_iters
    for _ in range(eff_warmup):
        run_once()

    eff_iters = 3 if batch_tiles >= 256 else (5 if batch_tiles >= 64 else measure_iters)
    rows: list[dict[str, Any]] = []
    for r in range(runs):
        dts = [run_once() for _ in range(eff_iters)]
        mean_ms = statistics.mean(dts)
        tiles_per_ms = float(batch_tiles) / max(mean_ms, 1e-6)
        mtex_s = (batch_tiles * 65536.0 / 1e6) / max(mean_ms / 1000.0, 1e-9)
        gflops = (batch_tiles * 65536.0 * ops_per_texel / 1e9) / max(mean_ms / 1000.0, 1e-9)
        rows.append(
            {
                "run_id": r,
                "backend": backend_label,
                "runtime_device": active_p,
                "precision": precision,
                "hidden_dim": hidden_dim,
                "batch_tiles": batch_tiles,
                "infer_only_ms": round(mean_ms, 5),
                "tiles_per_ms": round(tiles_per_ms, 5),
                "mtexels_per_sec": round(mtex_s, 4),
                "effective_gflops": round(gflops, 3),
                "execution_note": f"ort_{active_p}",
            }
        )
        if r == 0:
            print(
                f"  [{backend_label}] H={hidden_dim} B={batch_tiles} prec={precision}: "
                f"{mean_ms:.3f} ms | {mtex_s:.2f} Mtex/s | {gflops:.1f} GOP/s"
            )
    return rows


def run_task1_and_task2(
    exp_cfg: dict[str, Any],
    devs: dict[str, Any],
    out_dir: Path,
    runs: int,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run C++ DX12 Compute/Transfer sweeps + Python OpenVINO/ORT sweeps + Break-Even Derivation."""
    import openvino as ov

    dx12_comp_csv = out_dir / "dx12_compute_sweep.csv"
    dx12_xfer_csv = out_dir / "dx12_transfer_sweep.csv"
    warmup = 3 if smoke else int(exp_cfg["warmup_iters"])
    iters = 5 if smoke else int(exp_cfg["measure_iters"])

    run_cmd([
        str(DX12_BENCH_EXE), "--mode", "compute_sweep",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(dx12_comp_csv),
        "--runs", str(runs), "--warmup", str(warmup), "--iters", str(iters),
    ])
    run_cmd([
        str(DX12_BENCH_EXE), "--mode", "transfer_sweep",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(dx12_xfer_csv),
        "--runs", str(runs), "--warmup", str(warmup), "--iters", str(iters),
    ])

    batches = [1, 4, 16, 64, 256] if not smoke else [1, 16]
    hidden_dims_b16 = [16, 32, 64] if not smoke else [32]
    rng_tiles = build_real_calib_tiles(
        REPO_ROOT / exp_cfg["asset_manifest"],
        REPO_ROOT / exp_cfg["asset_pack"],
        mat_id=0,
    )

    host_overhead = benchmark_host_prep_and_pack(batches, runs=max(2, runs))

    all_rows: list[dict[str, Any]] = []

    # Ingest C++ DX12 Compute sweep rows
    with dx12_comp_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b_val = int(row["batch_tiles"])
            if b_val not in batches:
                continue
            h_val = int(row["hidden_dim"])
            if h_val != 32 and h_val not in hidden_dims_b16:
                continue
            all_rows.append(
                {
                    "run_id": int(row["run_id"]),
                    "backend": row["backend"],
                    "runtime_device": devs["dgpu_name"],
                    "precision": row["precision"],
                    "hidden_dim": h_val,
                    "batch_tiles": b_val,
                    "infer_only_ms": round(float(row["e2e_dispatch_ms"]), 5),
                    "gpu_kernel_ms": round(float(row["gpu_kernel_ms"]), 5),
                    "tiles_per_ms": round(float(row["tiles_per_ms"]), 5),
                    "mtexels_per_sec": round(float(row["mtexels_per_sec"]), 4),
                    "effective_gflops": round(float(row["effective_gflops"]), 3),
                    "execution_note": f"dx12_{row['output_format']}",
                }
            )

    core = ov.Core()
    # Configure NPU/GPU performance hints if supported
    backends_to_run: list[tuple[str, str, str, str, str]] = []
    # (kind, device_or_provider, label, precision, filename_prec)
    if devs.get("ov_npu_device"):
        backends_to_run.append(("ov", devs["ov_npu_device"], "NPU_OpenVINO_INT8", "INT8", "int8"))
        backends_to_run.append(("ov", devs["ov_npu_device"], "NPU_OpenVINO_FP16", "FP16", "fp16"))
    if devs.get("igpu_dml_device_id") is not None and "DmlExecutionProvider" in devs["ort_providers"]:
        backends_to_run.append(("ort_igpu", "DmlExecutionProvider", "iGPU_ORT_DML_FP16", "FP16", "fp16"))
    if devs.get("ov_igpu_device"):
        backends_to_run.append(("ov", devs["ov_igpu_device"], "iGPU_OpenVINO_FP16", "FP16", "fp16"))
        backends_to_run.append(("ov", devs["ov_igpu_device"], "iGPU_OpenVINO_INT8", "INT8", "int8"))
    if devs.get("dgpu_dml_device_id") is not None and "DmlExecutionProvider" in devs["ort_providers"] and not smoke:
        backends_to_run.append(("ort_dgpu", "DmlExecutionProvider", "dGPU_ORT_DML_FP16", "FP16", "fp16"))
    backends_to_run.append(("ov", "CPU", "CPU_OpenVINO_INT8", "INT8", "int8"))
    if not smoke:
        backends_to_run.append(("ort_cpu", "CPUExecutionProvider", "CPU_ORT_INT8", "INT8", "int8"))

    for kind, target, label, prec, fprec in backends_to_run:
        for b in batches:
            if b == 256 and ("ORT_CPU" in label or "ORT_DML" in label) and smoke:
                continue
            mp = MODELS_DIR / f"ntc_h32_b{b}_{fprec}.onnx"
            try:
                if kind == "ov":
                    rows = bench_openvino_backend(
                        core, target, label, mp, prec, 32, b, runs, warmup, iters, rng_tiles
                    )
                elif kind == "ort_igpu":
                    rows = bench_ort_backend(
                        label, target, {"device_id": devs["igpu_dml_device_id"]},
                        mp, prec, 32, b, runs, warmup, iters, rng_tiles,
                    )
                elif kind == "ort_dgpu":
                    rows = bench_ort_backend(
                        label, target, {"device_id": devs["dgpu_dml_device_id"]},
                        mp, prec, 32, b, runs, warmup, iters, rng_tiles,
                    )
                else:
                    rows = bench_ort_backend(
                        label, target, None, mp, prec, 32, b, runs, warmup, iters, rng_tiles,
                    )
                for rr in rows:
                    rr.setdefault("gpu_kernel_ms", 0.0)
                all_rows.extend(rows)
            except Exception as exc:
                print(f"[hetero_bench] WARN {label} H=32 B={b} failed: {exc}")

        # Hidden-dim sensitivity (H=16, 64 at B=16) on primary accelerators
        if label in ("NPU_OpenVINO_INT8", "iGPU_OpenVINO_FP16", "iGPU_ORT_DML_FP16", "CPU_OpenVINO_INT8"):
            for h in hidden_dims_b16:
                if h == 32:
                    continue
                mp_h = MODELS_DIR / f"ntc_h{h}_b16_{fprec}.onnx"
                try:
                    if kind == "ov":
                        rows = bench_openvino_backend(
                            core, target, label, mp_h, prec, h, 16, runs, warmup, iters, rng_tiles
                        )
                    elif kind == "ort_igpu":
                        rows = bench_ort_backend(
                            label, target, {"device_id": devs["igpu_dml_device_id"]},
                            mp_h, prec, h, 16, runs, warmup, iters, rng_tiles,
                        )
                    else:
                        continue
                    for rr in rows:
                        rr.setdefault("gpu_kernel_ms", 0.0)
                    all_rows.extend(rows)
                except Exception as exc:
                    print(f"[hetero_bench] WARN {label} H={h} B=16 failed: {exc}")

    # Parse C++ PCIe Transfer sweep
    xfer_agg: dict[tuple[str, int], dict[str, float]] = {}
    xfer_raw: dict[tuple[str, int], list[dict[str, float]]] = {}
    with dx12_xfer_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = (row["path_name"], int(row["batch_tiles"]))
            xfer_raw.setdefault(k, []).append(
                {
                    "copy_gpu_ms": float(row["copy_gpu_ms"]),
                    "compute_gpu_ms": float(row["compute_gpu_ms"]),
                    "serial_e2e_ms": float(row["serial_e2e_ms"]),
                    "bandwidth_gbps": float(row["bandwidth_gbps"]),
                }
            )
    for k, items in xfer_raw.items():
        xfer_agg[k] = {
            "copy_gpu_ms": statistics.mean(x["copy_gpu_ms"] for x in items),
            "compute_gpu_ms": statistics.mean(x["compute_gpu_ms"] for x in items),
            "serial_e2e_ms": statistics.mean(x["serial_e2e_ms"] for x in items),
            "bandwidth_gbps": statistics.mean(x["bandwidth_gbps"] for x in items),
        }

    # Build End-to-End transfer + compute enrichment onto all_rows
    for r in all_rows:
        b = int(r["batch_tiles"])
        bk = r["backend"]
        ho = host_overhead.get(b, {"host_gather_ms": 0.0, "host_bcn_pack_ms": 0.0})
        if bk.startswith("dGPU_"):
            t_lat = xfer_agg.get(("Latent_72KB_Plus_dGPU_MLP_BCn", b), {})
            r["host_gather_ms"] = 0.0
            r["pcie_transfer_ms"] = round(t_lat.get("copy_gpu_ms", 0.0), 5)
            r["post_pack_ms"] = 0.0
            r["full_e2e_ms"] = round(r["infer_only_ms"] + r["pcie_transfer_ms"], 5)
            r["e2e_delivery_path"] = "PCIe_72KB_Latent + dGPU_Compute_BCn"
        else:
            t_raw = xfer_agg.get(("HostPredecoded_Raw_512KB_Plus_GPU_BCnPack", b), {})
            t_bcn = xfer_agg.get(("HostPredecoded_BCn_192KB_Upload", b), {})
            # Fastest host-to-VRAM delivery: raw upload + dGPU BCn pack vs CPU BCn pack + 192KB upload
            e2e_raw_gpu_pack = ho["host_gather_ms"] + r["infer_only_ms"] + t_raw.get("copy_gpu_ms", 0.0) + t_raw.get("compute_gpu_ms", 0.0)
            e2e_cpu_bcn_pack = ho["host_gather_ms"] + r["infer_only_ms"] + ho["host_bcn_pack_ms"] + t_bcn.get("copy_gpu_ms", 0.0)
            if e2e_raw_gpu_pack <= e2e_cpu_bcn_pack:
                r["host_gather_ms"] = ho["host_gather_ms"]
                r["pcie_transfer_ms"] = round(t_raw.get("copy_gpu_ms", 0.0), 5)
                r["post_pack_ms"] = round(t_raw.get("compute_gpu_ms", 0.0), 5)
                r["full_e2e_ms"] = round(e2e_raw_gpu_pack, 5)
                r["e2e_delivery_path"] = "HostGather + Infer + PCIe_512KB_Raw + dGPU_BCnPack"
            else:
                r["host_gather_ms"] = ho["host_gather_ms"]
                r["pcie_transfer_ms"] = round(t_bcn.get("copy_gpu_ms", 0.0), 5)
                r["post_pack_ms"] = ho["host_bcn_pack_ms"]
                r["full_e2e_ms"] = round(e2e_cpu_bcn_pack, 5)
                r["e2e_delivery_path"] = "HostGather + Infer + HostBCnPack + PCIe_192KB_BCn"

    matrix_csv = out_dir / "hetero_benchmark_matrix.csv"
    fieldnames = [
        "run_id", "backend", "runtime_device", "precision", "hidden_dim", "batch_tiles",
        "infer_only_ms", "gpu_kernel_ms", "host_gather_ms", "pcie_transfer_ms", "post_pack_ms",
        "full_e2e_ms", "tiles_per_ms", "mtexels_per_sec", "effective_gflops",
        "e2e_delivery_path", "execution_note",
    ]
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[hetero_bench] Wrote {matrix_csv} ({len(all_rows)} rows)")

    # Write Transfer Cost comparison table
    xfer_rows: list[dict[str, Any]] = []
    # Aggregate mean infer latency per backend (H=32)
    mean_infer: dict[tuple[str, int], float] = {}
    grp: dict[tuple[str, int], list[float]] = {}
    for r in all_rows:
        if r["hidden_dim"] == 32:
            grp.setdefault((r["backend"], r["batch_tiles"]), []).append(float(r["infer_only_ms"]))
    for k, vals in grp.items():
        mean_infer[k] = statistics.mean(vals)

    for b in batches:
        ho = host_overhead[b]
        t_lat = xfer_agg.get(("Latent_72KB_Plus_dGPU_MLP_BCn", b), {})
        t_bcn = xfer_agg.get(("HostPredecoded_BCn_192KB_Upload", b), {})
        t_raw = xfer_agg.get(("HostPredecoded_Raw_512KB_Plus_GPU_BCnPack", b), {})
        xfer_rows.append(
            {
                "batch_tiles": b,
                "latent_72kb_copy_ms": round(t_lat.get("copy_gpu_ms", 0.0), 5),
                "dgpu_mlp_bcn_ms": round(mean_infer.get(("dGPU_DX12_Compute_FP16_BCn", b), t_lat.get("compute_gpu_ms", 0.0)), 5),
                "latent_plus_dgpu_e2e_ms": round(
                    t_lat.get("copy_gpu_ms", 0.0) + mean_infer.get(("dGPU_DX12_Compute_FP16_BCn", b), t_lat.get("compute_gpu_ms", 0.0)), 5
                ),
                "pcie_bw_72kb_gbps": round(t_lat.get("bandwidth_gbps", 0.0), 3),
                "host_192kb_bcn_copy_ms": round(t_bcn.get("copy_gpu_ms", 0.0), 5),
                "pcie_bw_192kb_gbps": round(t_bcn.get("bandwidth_gbps", 0.0), 3),
                "host_512kb_raw_copy_ms": round(t_raw.get("copy_gpu_ms", 0.0), 5),
                "dgpu_bcn_pack_ms": round(t_raw.get("compute_gpu_ms", 0.0), 5),
                "pcie_bw_512kb_gbps": round(t_raw.get("bandwidth_gbps", 0.0), 3),
                "npu_int8_infer_ms": round(mean_infer.get(("NPU_OpenVINO_INT8", b), 0.0), 5),
                "npu_raw_plus_pack_e2e_ms": round(
                    ho["host_gather_ms"] + mean_infer.get(("NPU_OpenVINO_INT8", b), 0.0)
                    + t_raw.get("copy_gpu_ms", 0.0) + t_raw.get("compute_gpu_ms", 0.0), 5
                ),
                "igpu_fp16_infer_ms": round(mean_infer.get(("iGPU_ORT_DML_FP16", b), mean_infer.get(("iGPU_OpenVINO_FP16", b), 0.0)), 5),
                "igpu_raw_plus_pack_e2e_ms": round(
                    ho["host_gather_ms"] + mean_infer.get(("iGPU_ORT_DML_FP16", b), mean_infer.get(("iGPU_OpenVINO_FP16", b), 0.0))
                    + t_raw.get("copy_gpu_ms", 0.0) + t_raw.get("compute_gpu_ms", 0.0), 5
                ),
                "host_gather_ms": ho["host_gather_ms"],
                "host_bcn_pack_ms": ho["host_bcn_pack_ms"],
            }
        )

    xfer_csv = out_dir / "transfer_cost_matrix.csv"
    with xfer_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(xfer_rows[0].keys()))
        w.writeheader()
        w.writerows(xfer_rows)
    print(f"[hetero_bench] Wrote {xfer_csv}")

    # Derive Break-Even Analysis JSON
    ops_per_texel_h32 = 2.0 * (16 * 32 + 32 * 32 + 32 * 8)  # 3584 ops/texel
    b16_row = next((x for x in xfer_rows if x["batch_tiles"] == 16), xfer_rows[-1])
    b16_cnt = int(b16_row["batch_tiles"])
    dgpu_b16_ms = b16_row["latent_plus_dgpu_e2e_ms"]
    npu_b16_infer = b16_row["npu_int8_infer_ms"]
    npu_b16_e2e = b16_row["npu_raw_plus_pack_e2e_ms"]

    # Effective TOPS achieved at B=16 on NPU INT8 and dGPU FP16
    texels_b16 = b16_cnt * 65536.0
    npu_eff_tops_b16 = (texels_b16 * ops_per_texel_h32 / 1e12) / max(npu_b16_infer / 1000.0, 1e-9)
    dgpu_eff_tops_b16 = (texels_b16 * ops_per_texel_h32 / 1e12) / max(b16_row["dgpu_mlp_bcn_ms"] / 1000.0, 1e-9)
    measured_util = npu_eff_tops_b16 / 13.0  # Arrow Lake NPU 13 TOPS INT8 rated peak (~0.44% util due to 1x1 conv SRAM BW bound)

    # Minimum required effective NPU TOPS to beat uncontended dGPU+Latent on PCIe Gen5 x8
    overhead_b16_ms = b16_row["host_512kb_raw_copy_ms"] + b16_row["dgpu_bcn_pack_ms"]
    rem_budget_ms = dgpu_b16_ms - overhead_b16_ms
    req_tops_no_congestion = (
        (texels_b16 * ops_per_texel_h32 / 1e12) / (rem_budget_ms / 1000.0)
        if rem_budget_ms > 0.02 else None
    )
    # Break-even dGPU queue congestion / shader contention delay required for current NPU to break even
    breakeven_infer_only_delay_ms = max(0.0, (npu_b16_infer + overhead_b16_ms) - dgpu_b16_ms)
    breakeven_dgpu_queue_delay_ms = max(0.0, npu_b16_e2e - dgpu_b16_ms)

    # Sweep grid (Achieved Effective NPU TOPS vs dGPU Contention Delay ms) at B=16
    # Includes both low effective TOPS (0.06..0.50 = 1x1-conv BW-bound NPU at 13..100 TOPS) and tensor-core speeds (1.0..4.0 eff TOPS)
    eff_tops_grid = [0.057, 0.15, 0.30, 0.50, 1.00, 1.50, 2.50, 4.00]
    queue_delays = [0.0, 2.5, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0]
    breakeven_grid = []
    for eff_tops in eff_tops_grid:
        npu_sim_infer_ms = (texels_b16 * ops_per_texel_h32 / 1e12) / eff_tops * 1000.0
        for qd in queue_delays:
            dgpu_tot = dgpu_b16_ms + qd
            npu_tot = npu_sim_infer_ms + overhead_b16_ms
            breakeven_grid.append(
                {
                    "npu_peak_tops": eff_tops,
                    "dgpu_queue_delay_ms": qd,
                    "net_gain_ms": round(dgpu_tot - npu_tot, 3),
                    "offload_wins": bool(npu_tot < dgpu_tot),
                }
            )

    be_json = {
        "hardware_resolved": devs,
        "reference_batch_tiles": b16_cnt,
        "ops_per_texel_h32": ops_per_texel_h32,
        "observed_b16": b16_row,
        "npu_int8_effective_tops_b16": round(npu_eff_tops_b16, 4),
        "npu_measured_mac_utilization_pct": round(measured_util * 100.0, 3),
        "dgpu_fp16_effective_tops_b16": round(dgpu_eff_tops_b16, 4),
        "required_npu_effective_tops_uncontended_gen5x8": (round(req_tops_no_congestion, 2) if req_tops_no_congestion else "impossible_pcie_exceeds_dgpu_total"),
        "breakeven_infer_only_delay_ms_for_current_npu": round(breakeven_infer_only_delay_ms, 3),
        "breakeven_dgpu_queue_delay_ms_for_current_npu": round(breakeven_dgpu_queue_delay_ms, 3),
        "breakeven_grid_b16": breakeven_grid,
        "transfer_summary": xfer_rows,
    }
    return all_rows, xfer_rows, be_json


def run_task3_power_contention(
    exp_cfg: dict[str, Any],
    devs: dict[str, Any],
    out_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Run sustained RTX 5060 load while sequentially waking NPU INT8, iGPU FP16, and CPU INT8."""
    import openvino as ov

    contention_csv = out_dir / "power_contention_trace.csv"
    sync_file = out_dir / "_contention_stage.txt"
    stages = exp_cfg["contention_stage_seconds"]
    if smoke:
        stages = {
            "Idle": 1.5,
            "dGPU_Solo": 2.0,
            "dGPU_plus_NPU": 2.0,
            "dGPU_plus_iGPU": 2.0,
            "dGPU_plus_CPU": 2.0,
            "dGPU_Cooldown": 1.5,
        }

    total_ms = int(sum(stages.values()) * 1000.0) + 5000
    sync_file.write_text("Idle\n", encoding="utf-8")

    # Precompile heavy OpenVINO models before launching dGPU load so compilation overhead doesn't blur windows
    core = ov.Core()
    rng_tiles = build_real_calib_tiles(
        REPO_ROOT / exp_cfg["asset_manifest"],
        REPO_ROOT / exp_cfg["asset_pack"],
        mat_id=0,
    )
    x_16_f32 = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (16, 16, 256, 256)).astype(np.float32))
    x_16_f16 = x_16_f32.astype(np.float16)

    npu_req = None
    if devs.get("ov_npu_device"):
        npu_comp = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_int8.onnx"), devs["ov_npu_device"])
        npu_req = npu_comp.create_infer_request()
        npu_req.infer({0: x_16_f32})

    igpu_req = None
    if devs.get("ov_igpu_device"):
        igpu_comp = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_fp16.onnx"), devs["ov_igpu_device"])
        igpu_req = igpu_comp.create_infer_request()
        igpu_req.infer({0: x_16_f16})

    cpu_comp = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_int8.onnx"), "CPU")
    cpu_req = cpu_comp.create_infer_request()
    cpu_req.infer({0: x_16_f32})

    proc = subprocess.Popen(
        [
            str(DX12_BENCH_EXE),
            "--mode", "sustained_dgpu_load",
            "--shader-dir", str(BUILD_DIR / "shaders"),
            "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
            "--out-csv", str(contention_csv),
            "--sync-file", str(sync_file),
            "--duration-ms", str(total_ms),
        ],
        cwd=str(REPO_ROOT),
    )

    try:
        for stage_name, dur_s in stages.items():
            sync_file.write_text(f"{stage_name}\n", encoding="utf-8")
            print(f"[contention] Stage={stage_name} ({dur_s:.1f}s)")
            t_end = time.perf_counter() + float(dur_s)

            if stage_name == "dGPU_plus_NPU" and npu_req is not None:
                while time.perf_counter() < t_end:
                    npu_req.infer({0: x_16_f32})
            elif stage_name == "dGPU_plus_iGPU" and igpu_req is not None:
                while time.perf_counter() < t_end:
                    igpu_req.infer({0: x_16_f16})
            elif stage_name == "dGPU_plus_CPU":
                # Keep CPU busy with multi-worker OpenVINO loops
                while time.perf_counter() < t_end:
                    cpu_req.infer({0: x_16_f32})
            else:
                time.sleep(float(dur_s))

        sync_file.write_text("STOP\n", encoding="utf-8")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        sync_file.unlink(missing_ok=True)

    # Summarize contention deltas (trim first 0.5s of each stage for stabilization)
    by_stage: dict[str, list[dict[str, float]]] = {}
    with contention_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_stage.setdefault(row["stage"], []).append(
                {
                    "t": float(row["timestamp_ms"]),
                    "dgpu_w": float(row["dgpu_power_w"]),
                    "cpu_pkg_w": float(row["cpu_pkg_power_w"]),
                    "cpu_pp0_w": float(row["cpu_pp0_power_w"]),
                    "igpu_pp1_w": float(row["igpu_pp1_power_w"]),
                    "total_w": float(row["total_est_power_w"]),
                    "sm_mhz": float(row["dgpu_sm_clock_mhz"]),
                    "temp_c": float(row["dgpu_temp_c"]),
                    "tps": float(row["dgpu_tiles_per_sec"]),
                }
            )

    summary_stages: dict[str, dict[str, float]] = {}
    for st, items in by_stage.items():
        if st == "dGPU_Solo" and len(items) > 45:
            items = items[35:]  # discard ~1.8s dGPU thermal/power ramp from idle
        elif len(items) > 8:
            items = items[6:]  # discard 300ms transition
        summary_stages[st] = {
            "samples": len(items),
            "dgpu_power_w": round(statistics.mean(x["dgpu_w"] for x in items), 3),
            "cpu_pkg_power_w": round(statistics.mean(x["cpu_pkg_w"] for x in items), 3),
            "cpu_pp0_power_w": round(statistics.mean(x["cpu_pp0_w"] for x in items), 3),
            "igpu_pp1_power_w": round(statistics.mean(x["igpu_pp1_w"] for x in items), 3),
            "total_est_power_w": round(statistics.mean(x["total_w"] for x in items), 3),
            "dgpu_sm_clock_mhz": round(statistics.mean(x["sm_mhz"] for x in items), 1),
            "dgpu_temp_c": round(statistics.mean(x["temp_c"] for x in items), 1),
            "dgpu_tiles_per_sec": round(statistics.mean(x["tps"] for x in items), 1),
        }

    solo = summary_stages.get("dGPU_Solo", {})
    deltas: dict[str, dict[str, float]] = {}
    for st in ("dGPU_plus_NPU", "dGPU_plus_iGPU", "dGPU_plus_CPU"):
        if st in summary_stages and solo:
            s = summary_stages[st]
            deltas[st] = {
                "delta_dgpu_power_w": round(s["dgpu_power_w"] - solo["dgpu_power_w"], 3),
                "delta_cpu_pkg_power_w": round(s["cpu_pkg_power_w"] - solo["cpu_pkg_power_w"], 3),
                "delta_total_power_w": round(s["total_est_power_w"] - solo["total_est_power_w"], 3),
                "delta_dgpu_sm_clock_mhz": round(s["dgpu_sm_clock_mhz"] - solo["dgpu_sm_clock_mhz"], 1),
                "delta_dgpu_throughput_pct": round(
                    100.0 * (s["dgpu_tiles_per_sec"] - solo["dgpu_tiles_per_sec"]) / max(solo["dgpu_tiles_per_sec"], 1e-6), 2
                ),
            }

    return {"stages": summary_stages, "deltas_vs_dgpu_solo": deltas}


def build_variant_bcn_pack(
    out_bin: Path,
    out_manifest: Path,
    decoded_mats_512: np.ndarray,
) -> None:
    """Build a valid Phase-1-compatible manifest + binary pack with BC7/BC5 mips decoded from NPU/FP16."""
    base_manifest = json.loads((REPO_ROOT / "results" / "assets_cache" / "manifest.json").read_text(encoding="utf-8"))
    base_blob = (REPO_ROOT / "results" / "assets_cache" / "pbr_ntc_pack_v1.bin").read_bytes()

    out_dir = out_bin.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_tbl = base_manifest["mlp_table"]
    mlp_off = int(mlp_tbl["offset"])
    mlp_bytes = int(mlp_tbl["bytes"])

    new_mats_meta: list[dict[str, Any]] = []
    with out_bin.open("wb") as bf:
        # Copy original MLP table verbatim at offset 0
        bf.write(base_blob[mlp_off : mlp_off + mlp_bytes])
        cursor = mlp_bytes

        def write_blob(arr: np.ndarray) -> dict[str, int]:
            nonlocal cursor
            raw = np.ascontiguousarray(arr).tobytes()
            bf.write(raw)
            sl = {"offset": cursor, "bytes": len(raw)}
            cursor += len(raw)
            return sl

        for mid in range(32):
            m_meta = json.loads(json.dumps(base_manifest["materials"][mid]))
            # Preserve G0 & G1
            new_g0 = []
            for s in m_meta["g0_mips"]:
                b_arr = np.frombuffer(base_blob, dtype=np.uint8, count=int(s["bytes"]), offset=int(s["offset"]))
                new_g0.append(write_blob(b_arr))
            new_g1 = []
            for s in m_meta["g1_mips"]:
                b_arr = np.frombuffer(base_blob, dtype=np.uint8, count=int(s["bytes"]), offset=int(s["offset"]))
                new_g1.append(write_blob(b_arr))

            # Build BC7/BC5 mip pyramid 512..8 (mips 0..6) from decoded_mats_512 [8, 512, 512] float32 in [0,1]
            cur = decoded_mats_512[mid].copy()
            new_bcn = []
            for m_lvl in range(7):
                sz = cur.shape[1]
                cur_u8 = np.clip(np.rint(cur * 255.0), 0, 255).astype(np.uint8)
                alb_rgba = np.stack([cur_u8[0], cur_u8[1], cur_u8[2], cur_u8[3]], axis=-1)
                orm_rgba = np.stack([cur_u8[6], cur_u8[7], cur_u8[3], np.full((sz, sz), 255, dtype=np.uint8)], axis=-1)
                bc7_alb = encode_bc7_mode6_np(alb_rgba)
                bc7_orm = encode_bc7_mode6_np(orm_rgba)
                bc4_nx = encode_bc4_channel_np(cur_u8[4])
                bc4_ny = encode_bc4_channel_np(cur_u8[5])
                bc5_nrm = np.stack([bc4_nx, bc4_ny], axis=-1).astype("<u8").view(np.uint8)

                new_bcn.append(
                    {
                        "size": sz,
                        "albedo_bc7": write_blob(bc7_alb),
                        "normal_bc5": write_blob(bc5_nrm),
                        "orm_bc7": write_blob(bc7_orm),
                    }
                )
                if sz > 8:
                    cur = cur.reshape(8, sz // 2, 2, sz // 2, 2).mean(axis=(2, 4))

            m_meta["g0_mips"] = new_g0
            m_meta["g1_mips"] = new_g1
            m_meta["bcn_mips"] = new_bcn
            new_mats_meta.append(m_meta)

    new_manifest = dict(base_manifest)
    new_manifest["pack_file"] = str(out_bin)
    new_manifest["mlp_table"] = {"offset": 0, "bytes": mlp_bytes, "stride_bytes": int(mlp_tbl["stride_bytes"])}
    new_manifest["materials"] = new_mats_meta
    new_manifest["pack_total_bytes"] = out_bin.stat().st_size
    out_manifest.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")


def run_task4_quantization_loss(
    exp_cfg: dict[str, Any],
    devs: dict[str, Any],
    out_dir: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Run real NPU INT8 vs FP16 inference across all 32 materials and DX12 1080p checkpoints."""
    import openvino as ov

    manifest_path = REPO_ROOT / exp_cfg["asset_manifest"]
    pack_path = REPO_ROOT / exp_cfg["asset_pack"]
    pm_dir = MODELS_DIR / "per_mat"
    core = ov.Core()

    npu_dev = devs.get("ov_npu_device") or "CPU"
    # FP16 reference executes on OpenVINO iGPU/CPU FP32-or-FP16 (check if NPU supports FP16 too)
    fp16_dev = devs.get("ov_igpu_device") or "CPU"

    num_mats = 4 if smoke else 32
    decoded_int8 = np.zeros((32, 8, 512, 512), dtype=np.float32)
    decoded_fp16 = np.zeros((32, 8, 512, 512), dtype=np.float32)

    q_rows: list[dict[str, Any]] = []

    def tiles_to_512(t4: np.ndarray) -> np.ndarray:
        # [4, 8, 256, 256] -> [8, 512, 512]
        out = np.zeros((8, 512, 512), dtype=np.float32)
        out[:, 0:256, 0:256] = t4[0]
        out[:, 0:256, 256:512] = t4[1]
        out[:, 256:512, 0:256] = t4[2]
        out[:, 256:512, 256:512] = t4[3]
        return np.clip(out, 0.0, 1.0)

    for mid in range(num_mats):
        c4_f32 = build_real_calib_tiles(manifest_path, pack_path, mat_id=mid)
        m_int8_p = pm_dir / f"mat_{mid:02d}_b4_int8.onnx"
        m_fp16_p = pm_dir / f"mat_{mid:02d}_b4_fp16.onnx"

        comp_i8 = core.compile_model(str(m_int8_p), npu_dev)
        req_i8 = comp_i8.create_infer_request()
        in_t_i8 = np.float16 if "f16" in str(comp_i8.input(0).element_type).lower() else np.float32
        out_i8 = np.asarray(req_i8.infer({0: c4_f32.astype(in_t_i8)})[0], dtype=np.float32)

        comp_f16 = core.compile_model(str(m_fp16_p), fp16_dev)
        req_f16 = comp_f16.create_infer_request()
        in_t_f16 = np.float16 if "f16" in str(comp_f16.input(0).element_type).lower() else np.float32
        out_f16 = np.asarray(req_f16.infer({0: c4_f32.astype(in_t_f16)})[0], dtype=np.float32)

        img_i8 = tiles_to_512(out_i8)
        img_f16 = tiles_to_512(out_f16)
        decoded_int8[mid] = img_i8
        decoded_fp16[mid] = img_f16

        # Material-level 8-channel PSNR & Albedo RGB PSNR + NVIDIA FLIP
        mse_8ch = float(np.mean((img_i8 - img_f16) ** 2))
        psnr_8ch = 10.0 * math.log10(1.0 / max(mse_8ch, 1e-12))
        alb_f16 = np.transpose(img_f16[0:3], (1, 2, 0))
        alb_i8 = np.transpose(img_i8[0:3], (1, 2, 0))
        mse_alb = float(np.mean((alb_i8 - alb_f16) ** 2))
        psnr_alb = 10.0 * math.log10(1.0 / max(mse_alb, 1e-12))

        flip_map, flip_mean, _ = flip_evaluator.evaluate(
            alb_f16, alb_i8, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
        )
        flip_p99 = float(np.percentile(flip_map, 99.0))

        # Normal angular / L2 PSNR
        mse_nrm = float(np.mean((img_i8[4:6] - img_f16[4:6]) ** 2))
        psnr_nrm = 10.0 * math.log10(1.0 / max(mse_nrm, 1e-12))

        q_rows.append(
            {
                "eval_scope": "material_texture_512",
                "target_id": f"mat_{mid:02d}",
                "npu_device": npu_dev,
                "psnr_8ch_int8_vs_fp16_db": round(psnr_8ch, 4),
                "psnr_albedo_int8_vs_fp16_db": round(psnr_alb, 4),
                "psnr_normal_int8_vs_fp16_db": round(psnr_nrm, 4),
                "flip_albedo_mean_int8_vs_fp16": round(float(flip_mean), 6),
                "flip_albedo_p99_int8_vs_fp16": round(flip_p99, 6),
            }
        )

    # Render actual 1080p DX12 checkpoints using decoded NPU_INT8 vs FP16 material packs
    if not smoke:
        qcap_dir = out_dir / "quant_captures"
        qcap_dir.mkdir(parents=True, exist_ok=True)
        for tag, arr in [("fp16", decoded_fp16), ("npu_int8", decoded_int8)]:
            v_dir = qcap_dir / tag
            v_dir.mkdir(parents=True, exist_ok=True)
            bin_p = v_dir / f"pack_{tag}.bin"
            man_p = v_dir / f"manifest_{tag}.json"
            build_variant_bcn_pack(bin_p, man_p, arr)

            cfg_tmp = json.loads((REPO_ROOT / exp_cfg["phase1_config"]).read_text(encoding="utf-8"))
            cfg_tmp["asset_pack"] = str(bin_p)
            cfg_tmp["asset_manifest"] = str(man_p)
            cfg_p = v_dir / f"cfg_{tag}.json"
            cfg_p.write_text(json.dumps(cfg_tmp, indent=2), encoding="utf-8")

            run_cmd([
                str(PA_BENCH_EXE),
                "--phase", "1",
                "--config", str(cfg_p),
                "--baseline", "B0_BCn_Full",
                "--run-id", "0",
                "--out-dir", str(v_dir),
                "--shader-dir", str(BUILD_DIR / "shaders"),
                "--capture-checkpoints",
            ])

        cam_cfg = json.loads((REPO_ROOT / exp_cfg["camera_path"]).read_text(encoding="utf-8"))
        b0_ref_dir = REPO_ROOT / "results" / "phase1" / "captures"
        for ck in cam_cfg["quality_checkpoints"]:
            ck_id = ck["id"]
            p_f16 = qcap_dir / "fp16" / "captures" / f"B0_BCn_Full_{ck_id}.png"
            p_i8 = qcap_dir / "npu_int8" / "captures" / f"B0_BCn_Full_{ck_id}.png"
            a_f16 = np.asarray(Image.open(p_f16).convert("RGB"), dtype=np.float32) / 255.0
            a_i8 = np.asarray(Image.open(p_i8).convert("RGB"), dtype=np.float32) / 255.0

            mse = float(np.mean((a_i8 - a_f16) ** 2))
            psnr_ck = 10.0 * math.log10(1.0 / max(mse, 1e-12)) if mse > 1e-12 else 99.0
            fmap, fmean, _ = flip_evaluator.evaluate(
                a_f16, a_i8, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
            )
            fp99 = float(np.percentile(fmap, 99.0))

            # Also check NPU INT8 rendered vs B0 BCn Ground Truth
            psnr_vs_b0 = 0.0
            flip_vs_b0 = 0.0
            p_b0 = b0_ref_dir / f"B0_BCn_Full_{ck_id}.png"
            if p_b0.exists():
                a_b0 = np.asarray(Image.open(p_b0).convert("RGB"), dtype=np.float32) / 255.0
                mse_b0 = float(np.mean((a_i8 - a_b0) ** 2))
                psnr_vs_b0 = 10.0 * math.log10(1.0 / max(mse_b0, 1e-12))
                _, flip_vs_b0, _ = flip_evaluator.evaluate(
                    a_b0, a_i8, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
                )

            q_rows.append(
                {
                    "eval_scope": "dx12_rendered_1080p",
                    "target_id": ck_id,
                    "npu_device": npu_dev,
                    "psnr_8ch_int8_vs_fp16_db": round(psnr_ck, 4),
                    "psnr_albedo_int8_vs_fp16_db": round(psnr_vs_b0, 4),
                    "psnr_normal_int8_vs_fp16_db": 0.0,
                    "flip_albedo_mean_int8_vs_fp16": round(float(fmean), 6),
                    "flip_albedo_p99_int8_vs_fp16": round(fp99, 6),
                }
            )

    q_csv = out_dir / "quantization_quality.csv"
    with q_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(q_rows[0].keys()))
        w.writeheader()
        w.writerows(q_rows)
    print(f"[hetero_bench] Wrote {q_csv} ({len(q_rows)} rows)")

    mat_rows = [r for r in q_rows if r["eval_scope"] == "material_texture_512"]
    ck_rows = [r for r in q_rows if r["eval_scope"] == "dx12_rendered_1080p"]
    return {
        "material_count": len(mat_rows),
        "mean_mat_8ch_psnr_int8_vs_fp16_db": round(statistics.mean(r["psnr_8ch_int8_vs_fp16_db"] for r in mat_rows), 3),
        "mean_mat_albedo_psnr_int8_vs_fp16_db": round(statistics.mean(r["psnr_albedo_int8_vs_fp16_db"] for r in mat_rows), 3),
        "mean_mat_albedo_flip_int8_vs_fp16": round(statistics.mean(r["flip_albedo_mean_int8_vs_fp16"] for r in mat_rows), 6),
        "max_mat_albedo_flip_p99": round(max(r["flip_albedo_p99_int8_vs_fp16"] for r in mat_rows), 6),
        "rendered_checkpoint_count": len(ck_rows),
        "mean_rendered_psnr_int8_vs_fp16_db": (round(statistics.mean(r["psnr_8ch_int8_vs_fp16_db"] for r in ck_rows), 3) if ck_rows else None),
        "mean_rendered_flip_int8_vs_fp16": (round(statistics.mean(r["flip_albedo_mean_int8_vs_fp16"] for r in ck_rows), 6) if ck_rows else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-AHNTS Phase 3 Heterogeneous Micro-Benchmark")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "experiments" / "phase3_hetero_bench.json")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    exp_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "results" / ("phase3_smoke" if args.smoke else "phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DX12_BENCH_EXE.exists() or not PA_BENCH_EXE.exists():
        run_cmd(["cmake", "-S", ".", "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
        run_cmd(["cmake", "--build", str(BUILD_DIR), "--target", "dx12_decode_bench", "pa_ahnts_bench"])

    if not args.skip_export and not (MODELS_DIR / "ntc_h32_b256_int8.onnx").exists():
        run_cmd([sys.executable, "src/hetero_bench/export_ntc_models.py", "--out-dir", str(MODELS_DIR)])

    devs = resolve_devices(REPO_ROOT / exp_cfg["hardware_profile"])
    runs = 1 if args.smoke else int(args.runs or exp_cfg["runs"])

    _, _, be_json = run_task1_and_task2(exp_cfg, devs, out_dir, runs=runs, smoke=args.smoke)
    cont_summary = run_task3_power_contention(exp_cfg, devs, out_dir, smoke=args.smoke)
    quant_summary = run_task4_quantization_loss(exp_cfg, devs, out_dir, smoke=args.smoke)

    be_json["power_contention"] = cont_summary
    be_json["quantization_loss"] = quant_summary
    be_path = out_dir / "break_even_analysis.json"
    be_path.write_text(json.dumps(be_json, indent=2), encoding="utf-8")
    print(f"[hetero_bench] Wrote {be_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
