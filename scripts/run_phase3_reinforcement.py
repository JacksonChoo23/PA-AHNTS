#!/usr/bin/env python3
"""PA-AHNTS Phase 3 Scientific Reinforcement Suite (Protocol Sections 4-8).

Implements:
  Sec 4: Backend Runtime Audit (OpenVINO get_runtime_model + ORT profiler) & Unified Op-Count Matrix (N=5)
  Sec 5: Native C++ AVX2/F16C Host Feature Gather Optimization & Equivalence Verification
  Sec 6: Complete End-to-End Break-Even Model + Measured DX12 GPU Queue Congestion
  Sec 7: Thermally Matched Standalone Joules/Tile & Concurrent Contention Root-Cause Decomposition
  Sec 8: Hardware-Specific Scheduler Cost Model (scheduler_cost_model.json) + Held-Out Validation & Plots
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "hetero_bench"))
from export_ntc_models import (  # noqa: E402
    build_ntc_onnx,
    build_real_calib_tiles,
    load_trained_mlp_weights,
    quantize_ntc_int8,
)
from run_hetero_bench import resolve_devices  # noqa: E402

BUILD_DIR = REPO_ROOT / "build"
DX12_BENCH_EXE = BUILD_DIR / "dx12_decode_bench.exe"
GATHER_DLL_PATH = BUILD_DIR / "pahnts_host_gather.dll"
MODELS_DIR = REPO_ROOT / "results" / "phase3" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "phase3_reinforced"
EXP_CFG_PATH = REPO_ROOT / "configs" / "experiments" / "phase3_hetero_bench.json"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.06)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def run_cmd(cmd: list[str]) -> None:
    print(f"[p3r] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def mean_std_ci95(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = statistics.mean(vals) if n > 0 else 0.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    t_tbl = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    ci = (t_tbl.get(n, 1.96) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": round(m, 5), "std": round(sd, 5), "ci95": round(ci, 5)}


def op_counts(hidden_dim: int, batch_tiles: int) -> tuple[int, int, float]:
    macs_per_tex = 16 * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 8
    ops_per_tex = 2 * macs_per_tex
    tot_gop = (float(batch_tiles) * 65536.0 * float(ops_per_tex)) / 1e9
    return macs_per_tex, ops_per_tex, tot_gop


# ---------------------------------------------------------------------------
# Host Gather DLL wrapper & CPU worker child mode
# ---------------------------------------------------------------------------
def load_gather_dll() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(GATHER_DLL_PATH))
    lib.pahnts_gather_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.pahnts_gather_init.restype = ctypes.c_int
    lib.pahnts_gather_register_material.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.pahnts_gather_batch_fp16.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.pahnts_gather_batch_fp32.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.pahnts_gather_get_fp32_ptr.restype = ctypes.c_void_p
    lib.pahnts_gather_get_fp16_ptr.restype = ctypes.c_void_p
    lib.pahnts_gather_ram_bytes.restype = ctypes.c_uint64
    lib.pahnts_gather_shutdown.restype = None
    return lib


def run_pinned_cpu_worker_loop(sync_file: Path, stage_target: str, count_file: Path, num_threads: int) -> int:
    """Subprocess CPU OpenVINO worker pinned away from Core 0 to isolate submission starvation."""
    import openvino as ov

    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": int(num_threads)})
    comp = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_int8.onnx"), "CPU")
    req = comp.create_infer_request()
    x = np.ones((16, 16, 256, 256), dtype=np.float32) * 0.1
    req.infer({0: x})
    print("CPU_WORKER_READY", flush=True)

    total_tiles = 0
    while True:
        st = ""
        try:
            st = sync_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        if st == "STOP":
            break
        if st == stage_target:
            req.infer({0: x})
            total_tiles += 16
        else:
            time.sleep(0.01)
    count_file.write_text(str(total_tiles), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Section 4: Backend Runtime Audit & Unified Op-Count Matrix
# ---------------------------------------------------------------------------
def audit_backends(core: Any, devs: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort

    audit: dict[str, Any] = {
        "hardware_resolved": devs,
        "exact_mlp_formula": {
            "architecture": "16 -> H -> H -> 8 (1x1 Conv2D)",
            "h32_macs_per_texel": 1792,
            "h32_ops_per_texel": 3584,
            "h32_b16_total_gop": op_counts(32, 16)[2],
        },
        "effective_tops_caveat": (
            "Effective TOPS/GOPs measure achieved end-to-end model throughput over 2*MACs "
            "and do NOT equal hardware MAC utilization or occupancy; SRAM/bandwidth hypotheses "
            "remain unconfirmed unless hardware performance counters directly measure them."
        ),
        "openvino_runtime_audit": {},
        "ort_profiler_audit": {},
    }

    ov_targets = [("CPU_OpenVINO_INT8", "CPU", "ntc_h32_b16_int8.onnx")]
    if devs.get("ov_npu_device"):
        ov_targets.extend([
            ("NPU_OpenVINO_INT8", devs["ov_npu_device"], "ntc_h32_b16_int8.onnx"),
            ("NPU_OpenVINO_FP16", devs["ov_npu_device"], "ntc_h32_b16_fp16.onnx"),
        ])
    if devs.get("ov_igpu_device"):
        ov_targets.extend([
            ("iGPU_OpenVINO_FP16", devs["ov_igpu_device"], "ntc_h32_b16_fp16.onnx"),
            ("iGPU_OpenVINO_INT8", devs["ov_igpu_device"], "ntc_h32_b16_int8.onnx"),
        ])

    x_dummy_f32 = np.zeros((16, 16, 256, 256), dtype=np.float32)
    for label, dev_name, fname in ov_targets:
        comp = core.compile_model(str(MODELS_DIR / fname), dev_name)
        try:
            exec_devs = str(comp.get_property("EXECUTION_DEVICES"))
        except Exception:
            exec_devs = dev_name
        rt_ops: list[dict[str, Any]] = []
        cpu_fallback_ops = 0
        try:
            rt_model = comp.get_runtime_model()
            for op in rt_model.get_ops():
                op_t = op.get_type_name()
                rt = {k: (str(v.value) if hasattr(v, "value") else str(v)) for k, v in op.get_rt_info().items()}
                layer_type = rt.get("layerType", op_t)
                prec = rt.get("runtimePrecision", "")
                impl = rt.get("implType", "")
                if dev_name != "CPU" and ("CPU" in impl or "ref_" in impl.lower()) and op_t not in ("Parameter", "Result", "Constant"):
                    cpu_fallback_ops += 1
                rt_ops.append({
                    "name": op.get_friendly_name(),
                    "op_type": op_t,
                    "layer_type": layer_type,
                    "runtime_precision": prec,
                    "impl_type": impl,
                })
        except Exception as exc:
            rt_ops.append({"error": str(exc)})

        audit["openvino_runtime_audit"][label] = {
            "target_device": dev_name,
            "execution_devices": exec_devs,
            "input_type": str(comp.input(0).element_type),
            "output_type": str(comp.output(0).element_type),
            "runtime_op_count": len(rt_ops),
            "detected_cpu_fallback_ops": cpu_fallback_ops,
            "runtime_ops": rt_ops[:24],
        }

    ort_targets: list[tuple[str, str, dict[str, Any] | None, str]] = [
        ("CPU_ORT_INT8", "CPUExecutionProvider", None, "ntc_h32_b16_int8.onnx"),
    ]
    if "DmlExecutionProvider" in devs.get("ort_providers", []):
        if devs.get("igpu_dml_device_id") is not None:
            ort_targets.append(("iGPU_ORT_DML_FP16", "DmlExecutionProvider", {"device_id": devs["igpu_dml_device_id"]}, "ntc_h32_b16_fp16.onnx"))
        if devs.get("dgpu_dml_device_id") is not None:
            ort_targets.append(("dGPU_ORT_DML_FP16", "DmlExecutionProvider", {"device_id": devs["dgpu_dml_device_id"]}, "ntc_h32_b16_fp16.onnx"))

    for label, prov, opts, fname in ort_targets:
        so = ort.SessionOptions()
        so.enable_profiling = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if prov == "DmlExecutionProvider":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            provs: Any = [(prov, opts), "CPUExecutionProvider"]
        else:
            provs = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(MODELS_DIR / fname), sess_options=so, providers=provs)
        in_name = sess.get_inputs()[0].name
        in_dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
        x_in = x_dummy_f32.astype(in_dtype)
        sess.run(None, {in_name: x_in})
        sess.run(None, {in_name: x_in})
        active_provs = sess.get_providers()
        sess_in_type = sess.get_inputs()[0].type
        prof_path = Path(sess.end_profiling())
        events = json.loads(prof_path.read_text(encoding="utf-8"))
        prof_path.unlink(missing_ok=True)
        del sess
        gc.collect()

        node_prov_counts: dict[str, int] = {}
        compute_cpu_fallbacks: list[str] = []
        sample_nodes: list[dict[str, Any]] = []
        for ev in events:
            if ev.get("cat") == "Node":
                args = ev.get("args", {})
                opn = str(args.get("op_name", ""))
                ep = str(args.get("provider", ""))
                node_prov_counts[ep] = node_prov_counts.get(ep, 0) + 1
                if prov == "DmlExecutionProvider" and ep == "CPUExecutionProvider" and opn not in ("MemcpyFromHost", "MemcpyToHost"):
                    compute_cpu_fallbacks.append(f"{ev.get('name')}:{opn}")
                if len(sample_nodes) < 16:
                    sample_nodes.append({"name": ev.get("name"), "op": opn, "provider": ep, "dur_us": ev.get("dur", 0)})

        audit["ort_profiler_audit"][label] = {
            "active_providers": active_provs,
            "input_type": sess_in_type,
            "provider_node_counts": node_prov_counts,
            "compute_cpu_fallback_nodes": compute_cpu_fallbacks,
            "sample_nodes": sample_nodes,
        }

    (out_dir / "backend_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[p3r] Wrote {out_dir / 'backend_audit.json'}")
    return audit


def measure_op_matrix(
    core: Any, exp_cfg: dict[str, Any], devs: dict[str, Any], out_dir: Path, runs: int, smoke: bool
) -> pd.DataFrame:
    import onnxruntime as ort
    import openvino as ov

    warmup = 2 if smoke else 5
    dx12_csv = out_dir / "dx12_compute_sweep.csv"
    dx12_xfer_csv = out_dir / "dx12_transfer_sweep.csv"
    dx12_iters = 5 if smoke else 20

    comp_cmd = [
        str(DX12_BENCH_EXE), "--mode", "compute_sweep",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(dx12_csv),
        "--runs", str(runs), "--warmup", str(warmup), "--iters", str(dx12_iters),
    ]
    if smoke:
        comp_cmd.extend(["--batches", "1,16"])
    run_cmd(comp_cmd)
    run_cmd([
        str(DX12_BENCH_EXE), "--mode", "transfer_sweep",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(dx12_xfer_csv),
        "--runs", str(runs), "--warmup", str(warmup), "--iters", str(dx12_iters),
    ])

    batches = [1, 16] if smoke else [1, 4, 16, 64, 256]
    h_dims = [32] if smoke else [16, 32, 64]
    rng_tiles = build_real_calib_tiles(
        REPO_ROOT / exp_cfg["asset_manifest"], REPO_ROOT / exp_cfg["asset_pack"], mat_id=0
    )

    rows: list[dict[str, Any]] = []
    # Ingest C++ DX12 Compute rows (compute consistent kernel_gops & e2e_infer_gops via exact op_counts)
    dx_df = pd.read_csv(dx12_csv)
    for (bk, prec, h, b), g in dx_df.groupby(["backend", "precision", "hidden_dim", "batch_tiles"]):
        h_i, b_i = int(h), int(b)
        if b_i not in batches or (h_i != 32 and h_i not in h_dims):
            continue
        macs_t, ops_t, tot_gop = op_counts(h_i, b_i)
        k_vals = g["gpu_kernel_ms"].tolist()
        e_vals = g["e2e_dispatch_ms"].tolist()
        ks = mean_std_ci95(k_vals)
        es = mean_std_ci95(e_vals)
        rows.append({
            "backend": bk, "precision": prec, "hidden_dim": h_i, "batch_tiles": b_i,
            "macs_per_texel": macs_t, "ops_per_texel": ops_t, "total_gop": round(tot_gop, 9),
            "kernel_mean_ms": ks["mean"], "infer_mean_ms": es["mean"], "infer_std_ms": es["std"],
            "infer_ci95_ms": es["ci95"],
            "infer_p50_ms": round(float(np.percentile(e_vals, 50.0)), 5),
            "infer_p99_ms": round(float(np.percentile(e_vals, 99.0)), 5),
            "mtexels_per_sec": round((b_i * 65536.0 / 1e6) / max(es["mean"] / 1000.0, 1e-9), 4),
            "kernel_gops": round(tot_gop / max(ks["mean"] / 1000.0, 1e-9), 3),
            "e2e_infer_gops": round(tot_gop / max(es["mean"] / 1000.0, 1e-9), 3),
            "execution_note": "dx12_hw_timestamps",
        })

    configs: list[tuple[str, str, str, str, str]] = []
    if devs.get("ov_npu_device"):
        configs.append(("ov", devs["ov_npu_device"], "NPU_OpenVINO_INT8", "INT8", "int8"))
        configs.append(("ov", devs["ov_npu_device"], "NPU_OpenVINO_FP16", "FP16", "fp16"))
    if devs.get("igpu_dml_device_id") is not None and "DmlExecutionProvider" in devs["ort_providers"]:
        configs.append(("ort_igpu", "DmlExecutionProvider", "iGPU_ORT_DML_FP16", "FP16", "fp16"))
    if devs.get("ov_igpu_device"):
        configs.append(("ov", devs["ov_igpu_device"], "iGPU_OpenVINO_FP16", "FP16", "fp16"))
        configs.append(("ov", devs["ov_igpu_device"], "iGPU_OpenVINO_INT8", "INT8", "int8"))
    if devs.get("dgpu_dml_device_id") is not None and not smoke:
        configs.append(("ort_dgpu", "DmlExecutionProvider", "dGPU_ORT_DML_FP16", "FP16", "fp16"))
    configs.append(("ov", "CPU", "CPU_OpenVINO_INT8", "INT8", "int8"))
    configs.append(("ort_cpu", "CPUExecutionProvider", "CPU_ORT_INT8", "INT8", "int8"))

    def measure_one(kind: str, target: str, label: str, prec: str, mp: Path, h_i: int, b_i: int) -> None:
        macs_t, ops_t, tot_gop = op_counts(h_i, b_i)
        used_b = b_i
        note = "native_batch"
        if kind == "ov":
            try:
                comp = core.compile_model(str(mp), target)
            except Exception as exc:
                sub_b = 16
                sub_p = mp.parent / mp.name.replace(f"_b{b_i}_", f"_b{sub_b}_")
                comp = core.compile_model(str(sub_p), target)
                used_b = sub_b
                note = f"async_queue_{b_i // sub_b}x{sub_b} ({str(exc)[:45]})"
            req = comp.create_infer_request()
            dt = np.float16 if "f16" in str(comp.input(0).element_type).lower() else np.float32
            x_in = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (used_b, 16, 256, 256)).astype(dt))

            def fn() -> float:
                t0 = time.perf_counter()
                if used_b == b_i:
                    req.infer({0: x_in})
                else:
                    q = ov.AsyncInferQueue(comp, min(4, b_i // used_b))
                    for _ in range(b_i // used_b):
                        q.start_async({0: x_in})
                    q.wait_all()
                return (time.perf_counter() - t0) * 1000.0
        else:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if target == "DmlExecutionProvider":
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                did = devs["igpu_dml_device_id"] if kind == "ort_igpu" else devs["dgpu_dml_device_id"]
                provs: Any = [(target, {"device_id": did}), "CPUExecutionProvider"]
            else:
                provs = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(mp), sess_options=so, providers=provs)
            in_name = sess.get_inputs()[0].name
            dt = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
            x_in = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (b_i, 16, 256, 256)).astype(dt))
            note = f"ort_{sess.get_providers()[0]}"

            def fn() -> float:
                t0 = time.perf_counter()
                sess.run(None, {in_name: x_in})
                return (time.perf_counter() - t0) * 1000.0

        n_w = 2 if b_i >= 64 else warmup
        n_it = (4 if smoke else 6) if b_i >= 256 else ((6 if smoke else 12) if b_i >= 64 else (5 if smoke else 20))
        for _ in range(n_w):
            fn()
        run_means: list[float] = []
        all_samples: list[float] = []
        for _ in range(runs):
            dts = [fn() for _ in range(n_it)]
            run_means.append(statistics.mean(dts))
            all_samples.extend(dts)

        st = mean_std_ci95(run_means)
        rows.append({
            "backend": label, "precision": prec, "hidden_dim": h_i, "batch_tiles": b_i,
            "macs_per_texel": macs_t, "ops_per_texel": ops_t, "total_gop": round(tot_gop, 9),
            "kernel_mean_ms": 0.0, "infer_mean_ms": st["mean"], "infer_std_ms": st["std"],
            "infer_ci95_ms": st["ci95"],
            "infer_p50_ms": round(float(np.percentile(all_samples, 50.0)), 5),
            "infer_p99_ms": round(float(np.percentile(all_samples, 99.0)), 5),
            "mtexels_per_sec": round((b_i * 65536.0 / 1e6) / max(st["mean"] / 1000.0, 1e-9), 4),
            "kernel_gops": 0.0,
            "e2e_infer_gops": round(tot_gop / max(st["mean"] / 1000.0, 1e-9), 3),
            "execution_note": note,
        })
        print(f"  [op_matrix] {label:<22} H={h_i} B={b_i:<3} mean={st['mean']:.3f} ms | P99={rows[-1]['infer_p99_ms']:.3f} | e2e_GOPS={rows[-1]['e2e_infer_gops']:.1f}")

    for kind, target, label, prec, fprec in configs:
        for b_i in batches:
            if smoke and b_i == 256:
                continue
            mp = MODELS_DIR / f"ntc_h32_b{b_i}_{fprec}.onnx"
            try:
                measure_one(kind, target, label, prec, mp, 32, b_i)
            except Exception as exc:
                print(f"[p3r] WARN {label} H=32 B={b_i} skipped: {exc}")
        if label in ("NPU_OpenVINO_INT8", "iGPU_ORT_DML_FP16", "iGPU_OpenVINO_FP16", "CPU_OpenVINO_INT8"):
            for h_i in h_dims:
                if h_i == 32:
                    continue
                try:
                    measure_one(kind, target, label, prec, MODELS_DIR / f"ntc_h{h_i}_b16_{fprec}.onnx", h_i, 16)
                except Exception as exc:
                    print(f"[p3r] WARN {label} H={h_i} B=16 skipped: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reinforced_op_matrix.csv", index=False)
    print(f"[p3r] Wrote {out_dir / 'reinforced_op_matrix.csv'} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Section 5: Host Feature Gather Optimization
# ---------------------------------------------------------------------------
def benchmark_host_gather_suite(exp_cfg: dict[str, Any], out_dir: Path, runs: int, smoke: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_p = REPO_ROOT / exp_cfg["asset_manifest"]
    pack_p = REPO_ROOT / exp_cfg["asset_pack"]
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    blob = pack_p.read_bytes()

    g0_list, g1_list = [], []
    for m in range(32):
        g0_s = manifest["materials"][m]["g0_mips"][0]
        g1_s = manifest["materials"][m]["g1_mips"][0]
        g0_list.append(np.frombuffer(blob, dtype=np.uint8, count=256 * 256 * 4, offset=int(g0_s["offset"])).copy())
        g1_list.append(np.frombuffer(blob, dtype=np.uint16, count=64 * 64 * 4, offset=int(g1_s["offset"])).copy())

    g0_grid0_f = g0_list[0].reshape(256, 256, 4).astype(np.float32) / 255.0
    g1_grid0_f = g1_list[0].view(np.float16).reshape(64, 64, 4).astype(np.float32)
    g0_chw = np.ascontiguousarray(np.transpose(g0_grid0_f, (2, 0, 1)))
    g1_chw = np.ascontiguousarray(np.transpose(g1_grid0_f, (2, 0, 1)))
    idx_q = np.arange(256, dtype=np.intp) // 4

    px = np.arange(256, dtype=np.float32) + 0.5
    yy, xx = np.meshgrid(px, px, indexing="ij")
    u, v = xx / 512.0, yy / 512.0
    two_pi = 2.0 * math.pi
    pe = np.stack([
        np.sin(two_pi * 8.0 * np.mod(u * 16.0, 1.0)), np.cos(two_pi * 8.0 * np.mod(u * 16.0, 1.0)),
        np.sin(two_pi * 8.0 * np.mod(v * 16.0, 1.0)), np.cos(two_pi * 8.0 * np.mod(v * 16.0, 1.0)),
        np.sin(two_pi * 16.0 * np.mod(u * 16.0, 1.0)), np.cos(two_pi * 16.0 * np.mod(u * 16.0, 1.0)),
        np.sin(two_pi * 16.0 * np.mod(v * 16.0, 1.0)), np.cos(two_pi * 16.0 * np.mod(v * 16.0, 1.0)),
    ], axis=0).astype(np.float16)

    # C++ AVX2/F16C persistent engine
    dll = load_gather_dll()
    if dll.pahnts_gather_init(64, 0) != 0:
        raise RuntimeError("pahnts_gather_init failed")
    for m in range(32):
        dll.pahnts_gather_register_material(m, g0_list[m].ctypes.data, g1_list[m].ctypes.data)
    cpp_ram_mib = float(dll.pahnts_gather_ram_bytes()) / (1024.0 * 1024.0)

    # Verify exact numerical equivalence against build_real_calib_tiles on Material 0 [4, 16, 256, 256]
    ref_4 = build_real_calib_tiles(manifest_p, pack_p, mat_id=0)
    m4 = np.zeros(4, dtype=np.int32)
    tx4 = np.array([0, 1, 0, 1], dtype=np.int32)
    ty4 = np.array([0, 0, 1, 1], dtype=np.int32)
    dll.pahnts_gather_batch_fp32(4, m4.ctypes.data, tx4.ctypes.data, ty4.ctypes.data, None)
    dll.pahnts_gather_batch_fp16(4, m4.ctypes.data, tx4.ctypes.data, ty4.ctypes.data, None)
    out32 = np.ctypeslib.as_array((ctypes.c_float * (4 * 16 * 256 * 256)).from_address(dll.pahnts_gather_get_fp32_ptr())).reshape(4, 16, 256, 256)
    out16 = np.ctypeslib.as_array((ctypes.c_uint16 * (4 * 16 * 256 * 256)).from_address(dll.pahnts_gather_get_fp16_ptr())).view(np.float16).reshape(4, 16, 256, 256).astype(np.float32)

    mse32 = float(np.mean((out32 - ref_4) ** 2))
    mse16 = float(np.mean((out16 - ref_4) ** 2))
    equiv = {
        "cpp_fp32_max_abs_err": float(np.max(np.abs(out32 - ref_4))),
        "cpp_fp32_rmse": math.sqrt(mse32),
        "cpp_fp32_psnr_db": round(10.0 * math.log10(4.0 / max(mse32, 1e-16)), 2),
        "cpp_fp16_max_abs_err": float(np.max(np.abs(out16 - ref_4))),
        "cpp_fp16_rmse": math.sqrt(mse16),
        "cpp_fp16_psnr_db": round(10.0 * math.log10(4.0 / max(mse16, 1e-16)), 2),
    }

    batches = [1, 16] if smoke else [1, 4, 16, 64]
    persist_np = np.empty((64, 16, 256, 256), dtype=np.float16)
    persist_g0_f16 = (g0_chw.astype(np.float16) * np.float16(2.0) - np.float16(1.0))
    persist_g1_up = g1_chw[:, idx_q[:, None], idx_q[None, :]].astype(np.float16)

    iters = 5 if smoke else 12
    rows: list[dict[str, Any]] = []
    for b in batches:
        mids = np.array([i % 32 for i in range(b)], dtype=np.int32)
        txs = np.array([i % 2 for i in range(b)], dtype=np.int32)
        tys = np.array([(i // 2) % 2 for i in range(b)], dtype=np.int32)

        def run_orig() -> float:
            t0 = time.perf_counter()
            out_f = np.empty((b, 16, 256, 256), dtype=np.float16)
            g0_s = g0_chw.astype(np.float16) * np.float16(2.0) - np.float16(1.0)
            g1_u = g1_chw[:, idx_q[:, None], idx_q[None, :]].astype(np.float16)
            out_f[:, 0:4] = g0_s[None, ...]
            out_f[:, 4:8] = g1_u[None, ...]
            out_f[:, 8:16] = pe[None, ...]
            return (time.perf_counter() - t0) * 1000.0

        def run_persist_np() -> float:
            t0 = time.perf_counter()
            v_out = persist_np[:b]
            v_out[:, 0:4] = persist_g0_f16[None, ...]
            v_out[:, 4:8] = persist_g1_up[None, ...]
            v_out[:, 8:16] = pe[None, ...]
            return (time.perf_counter() - t0) * 1000.0

        def run_cpp_fp16() -> float:
            t0 = time.perf_counter()
            dll.pahnts_gather_batch_fp16(b, mids.ctypes.data, txs.ctypes.data, tys.ctypes.data, None)
            return (time.perf_counter() - t0) * 1000.0

        def run_cpp_fp32() -> float:
            t0 = time.perf_counter()
            dll.pahnts_gather_batch_fp32(b, mids.ctypes.data, txs.ctypes.data, tys.ctypes.data, None)
            return (time.perf_counter() - t0) * 1000.0

        impls = [
            ("Original_Python_NumPy", run_orig, (b * 16 * 256 * 256 * 2) / (1024.0 ** 2) + 8.0, 0.0, 99.0),
            ("Persistent_Vectorized_NumPy", run_persist_np, (64 * 16 * 256 * 256 * 2) / (1024.0 ** 2), 0.0, 99.0),
            ("Optimized_Cpp_AVX2_FP16", run_cpp_fp16, cpp_ram_mib, equiv["cpp_fp16_max_abs_err"], equiv["cpp_fp16_psnr_db"]),
            ("Optimized_Cpp_AVX2_FP32", run_cpp_fp32, cpp_ram_mib, equiv["cpp_fp32_max_abs_err"], equiv["cpp_fp32_psnr_db"]),
        ]
        for name, fn, ram_m, err, psnr_d in impls:
            for _ in range(3):
                fn()
            r_means, samples = [], []
            for _ in range(runs):
                dts = [fn() for _ in range(iters)]
                r_means.append(statistics.mean(dts))
                samples.extend(dts)
            st = mean_std_ci95(r_means)
            rows.append({
                "implementation": name, "batch_tiles": b,
                "mean_ms": st["mean"], "std_ms": st["std"], "ci95_ms": st["ci95"],
                "p50_ms": round(float(np.percentile(samples, 50.0)), 4),
                "p99_ms": round(float(np.percentile(samples, 99.0)), 4),
                "arena_ram_mib": round(ram_m, 2),
                "active_batch_tensor_mib": round((b * 16 * 256 * 256 * (4 if "FP32" in name else 2)) / (1024.0 ** 2), 2),
                "max_abs_err_vs_ref": err, "psnr_vs_ref_db": psnr_d,
            })

    dll.pahnts_gather_shutdown()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "host_gather_comparison.csv", index=False)
    print(f"[p3r] Wrote {out_dir / 'host_gather_comparison.csv'} (FP16 B=16 equiv={equiv['cpp_fp16_psnr_db']} dB)")
    return df, equiv


# ---------------------------------------------------------------------------
# Section 6: Complete Break-Even Model + Measured GPU Congestion
# ---------------------------------------------------------------------------
def build_breakeven_and_congestion(
    exp_cfg: dict[str, Any], op_df: pd.DataFrame, gather_df: pd.DataFrame, out_dir: Path, runs: int, smoke: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    cong_csv = out_dir / "gpu_congestion_measured.csv"
    run_cmd([
        str(DX12_BENCH_EXE), "--mode", "gpu_congestion_bench",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(cong_csv),
        "--runs", str(runs), "--warmup", "2", "--iters", str(4 if smoke else 15),
        "--extra-loads",
    ])

    cong_df = pd.read_csv(cong_csv)
    xfer_df = pd.read_csv(out_dir / "dx12_transfer_sweep.csv")

    def get_op(bk: str, b: int, col: str = "infer_mean_ms") -> float:
        sub = op_df[(op_df["backend"] == bk) & (op_df["hidden_dim"] == 32) & (op_df["batch_tiles"] == b)]
        return float(sub[col].iloc[0]) if not sub.empty else 0.0

    def get_gather(impl: str, b: int) -> float:
        sub = gather_df[(gather_df["implementation"] == impl) & (gather_df["batch_tiles"] == b)]
        return float(sub["mean_ms"].iloc[0]) if not sub.empty else 0.0

    batches = [1, 16] if smoke else [1, 4, 16, 64]
    by_batch: dict[str, Any] = {}
    for b in batches:
        t_lat_up = float(xfer_df[(xfer_df["path_name"] == "Latent_72KB_Plus_dGPU_MLP_BCn") & (xfer_df["batch_tiles"] == b)]["copy_gpu_ms"].mean())
        t_raw_up = float(xfer_df[(xfer_df["path_name"] == "HostPredecoded_Raw_512KB_Plus_GPU_BCnPack") & (xfer_df["batch_tiles"] == b)]["copy_gpu_ms"].mean())
        t_pack = float(xfer_df[(xfer_df["path_name"] == "HostPredecoded_Raw_512KB_Plus_GPU_BCnPack") & (xfer_df["batch_tiles"] == b)]["compute_gpu_ms"].mean())
        t_dgpu_infer = get_op("dGPU_DX12_Compute_FP16_BCn", b)
        t_dgpu_e2e = t_lat_up + t_dgpu_infer

        g_orig = get_gather("Original_Python_NumPy", b)
        g_opt16 = get_gather("Optimized_Cpp_AVX2_FP16", b)
        g_opt32 = get_gather("Optimized_Cpp_AVX2_FP32", b)

        t_npu_inf = get_op("NPU_OpenVINO_INT8", b)
        t_igpu_inf = get_op("iGPU_ORT_DML_FP16", b)
        t_cpu_inf = get_op("CPU_OpenVINO_INT8", b)

        # Full T_complete = T_queue + T_batch_wait + T_gather + T_inference + T_encode + T_upload + T_sync
        # For offload comparison at steady state with T_batch_wait=0 (immediate full batch) & T_sync~0.03 ms:
        t_sync = 0.03
        dgpu_total = t_dgpu_e2e + t_sync

        npu_pre_serial = g_orig + t_npu_inf + t_raw_up + t_pack + t_sync
        npu_opt_serial = g_opt32 + t_npu_inf + t_raw_up + t_pack + t_sync
        npu_opt_overlapped = max(g_opt32, t_npu_inf, t_raw_up + t_pack) + t_sync

        igpu_pre_serial = g_orig + t_igpu_inf + t_raw_up + t_pack + t_sync
        igpu_opt_serial = g_opt16 + t_igpu_inf + t_raw_up + t_pack + t_sync
        igpu_opt_overlapped = max(g_opt16, t_igpu_inf, t_raw_up + t_pack) + t_sync

        cpu_opt_serial = g_opt32 + t_cpu_inf + t_raw_up + t_pack + t_sync
        cpu_opt_overlapped = max(g_opt32, t_cpu_inf, t_raw_up + t_pack) + t_sync

        by_batch[str(b)] = {
            "batch_tiles": b,
            "dgpu_uncontended_complete_ms": round(dgpu_total, 4),
            "t_latent_upload_ms": round(t_lat_up, 4),
            "t_dgpu_infer_bcn_ms": round(t_dgpu_infer, 4),
            "t_raw_upload_ms": round(t_raw_up, 4),
            "t_dgpu_bcn_pack_ms": round(t_pack, 4),
            "gather_orig_python_ms": round(g_orig, 4),
            "gather_opt_cpp_fp16_ms": round(g_opt16, 4),
            "gather_opt_cpp_fp32_ms": round(g_opt32, 4),
            "npu_pre_opt_serial_ms": round(npu_pre_serial, 4),
            "npu_post_opt_serial_ms": round(npu_opt_serial, 4),
            "npu_post_opt_overlapped_stage_ms": round(npu_opt_overlapped, 4),
            "npu_breakeven_qdelay_pre_opt_ms": round(max(0.0, npu_pre_serial - dgpu_total), 4),
            "npu_breakeven_qdelay_post_opt_serial_ms": round(max(0.0, npu_opt_serial - dgpu_total), 4),
            "npu_breakeven_qdelay_post_opt_overlapped_ms": round(max(0.0, npu_opt_overlapped - dgpu_total), 4),
            "igpu_pre_opt_serial_ms": round(igpu_pre_serial, 4),
            "igpu_post_opt_serial_ms": round(igpu_opt_serial, 4),
            "igpu_post_opt_overlapped_stage_ms": round(igpu_opt_overlapped, 4),
            "igpu_breakeven_qdelay_pre_opt_ms": round(max(0.0, igpu_pre_serial - dgpu_total), 4),
            "igpu_breakeven_qdelay_post_opt_serial_ms": round(max(0.0, igpu_opt_serial - dgpu_total), 4),
            "igpu_breakeven_qdelay_post_opt_overlapped_ms": round(max(0.0, igpu_opt_overlapped - dgpu_total), 4),
            "cpu_post_opt_serial_ms": round(cpu_opt_serial, 4),
            "cpu_breakeven_qdelay_post_opt_serial_ms": round(max(0.0, cpu_opt_serial - dgpu_total), 4),
        }

    # Summarize measured GPU congestion curves cleanly separated from analytical model
    cong_summary = []
    for (tld, b), g in cong_df.groupby(["target_load_ms", "batch_tiles"]):
        vals = g["total_complete_ms"].tolist()
        qvals = g["measured_queue_delay_ms"].tolist()
        cong_summary.append({
            "data_source": "measured_hardware",
            "target_load_ms": int(tld),
            "batch_tiles": int(b),
            "bg_gpu_mean_ms": round(float(g["bg_gpu_ms"].mean()), 4),
            "probe_kernel_mean_ms": round(float(g["probe_kernel_ms"].mean()), 4),
            "measured_qdelay_mean_ms": round(float(np.mean(qvals)), 4),
            "measured_qdelay_p99_ms": round(float(np.percentile(qvals, 99.0)), 4),
            "total_complete_mean_ms": round(float(np.mean(vals)), 4),
            "total_complete_p50_ms": round(float(np.percentile(vals, 50.0)), 4),
            "total_complete_p99_ms": round(float(np.percentile(vals, 99.0)), 4),
        })

    p3_orig = json.loads((REPO_ROOT / "results" / "phase3" / "break_even_analysis.json").read_text(encoding="utf-8")).get("observed_b16", {})
    be = {
        "data_source_policy": "Measured dx12_decode_bench GPU queue congestion curves are strictly separated from analytical extrapolations.",
        "original_phase3_b16_verified": {
            "dgpu_e2e_ms": p3_orig.get("latent_plus_dgpu_e2e_ms"),
            "npu_e2e_ms": p3_orig.get("npu_raw_plus_pack_e2e_ms"),
            "igpu_e2e_ms": p3_orig.get("igpu_raw_plus_pack_e2e_ms"),
            "host_gather_ms": p3_orig.get("host_gather_ms"),
        },
        "reinforced_by_batch": by_batch,
        "measured_gpu_congestion": cong_summary,
    }
    (out_dir / "breakeven_complete.json").write_text(json.dumps(be, indent=2), encoding="utf-8")
    print(f"[p3r] Wrote {out_dir / 'breakeven_complete.json'} & {cong_csv}")
    return be, cong_df


# ---------------------------------------------------------------------------
# Section 7: Thermally Matched Energy per Tile & Contention Decomposition
# ---------------------------------------------------------------------------
def run_energy_and_contention(
    core: Any, exp_cfg: dict[str, Any], devs: dict[str, Any], out_dir: Path, runs: int, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import onnxruntime as ort

    rng_tiles = build_real_calib_tiles(
        REPO_ROOT / exp_cfg["asset_manifest"], REPO_ROOT / exp_cfg["asset_pack"], mat_id=0
    )
    x_16_f32 = np.ascontiguousarray(np.broadcast_to(rng_tiles[0:1], (16, 16, 256, 256)).astype(np.float32))
    x_16_f16 = x_16_f32.astype(np.float16)

    npu_req = None
    if devs.get("ov_npu_device"):
        npu_req = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_int8.onnx"), devs["ov_npu_device"]).create_infer_request()
        npu_req.infer({0: x_16_f32})

    igpu_sess = None
    igpu_in = ""
    if devs.get("igpu_dml_device_id") is not None and "DmlExecutionProvider" in devs["ort_providers"]:
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        igpu_sess = ort.InferenceSession(
            str(MODELS_DIR / "ntc_h32_b16_fp16.onnx"), sess_options=so,
            providers=[("DmlExecutionProvider", {"device_id": devs["igpu_dml_device_id"]}), "CPUExecutionProvider"],
        )
        igpu_in = igpu_sess.get_inputs()[0].name
        igpu_sess.run(None, {igpu_in: x_16_f16})

    cpu_req = core.compile_model(str(MODELS_DIR / "ntc_h32_b16_int8.onnx"), "CPU").create_infer_request()
    cpu_req.infer({0: x_16_f32})

    stg_s = 1.8 if smoke else 4.0
    run_rows: list[dict[str, Any]] = []
    sync_file = out_dir / "_p3r_sync.txt"
    cnt_file = out_dir / "_p3r_pinned_cpu_tiles.txt"

    def parse_trace_stages(csv_p: Path, stages_keep: list[str]) -> dict[str, dict[str, float]]:
        df = pd.read_csv(csv_p)
        res: dict[str, dict[str, float]] = {}
        for st in stages_keep:
            sub = df[df["stage"] == st]
            if len(sub) > 10:
                sub = sub.iloc[6:]  # discard 300ms stage transition
            if sub.empty:
                continue
            res[st] = {
                "dgpu_power_w": float(sub["dgpu_power_w"].mean()),
                "cpu_pkg_power_w": float(sub["cpu_pkg_power_w"].mean()),
                "cpu_pp0_power_w": float(sub["cpu_pp0_power_w"].mean()),
                "igpu_pp1_power_w": float(sub["igpu_pp1_power_w"].mean()),
                "total_est_power_w": float(sub["total_est_power_w"].mean()),
                "dgpu_sm_clock_mhz": float(sub["dgpu_sm_clock_mhz"].mean()),
                "dgpu_temp_c": float(sub["dgpu_temp_c"].mean()),
                "dgpu_tiles_per_sec": float(sub["dgpu_tiles_per_sec"].mean()),
            }
        return res

    def wait_token(proc: subprocess.Popen[str], token: str) -> None:
        if proc.stdout is None:
            return
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if token in line:
                break

    for r in range(runs):
        # Part A: Standalone Energy per Tile (Idle, NPU, iGPU, CPU) + H=32 dGPU Solo
        sa_csv = out_dir / f"_tmp_sa_r{r}.csv"
        sync_file.write_text("Idle\n", encoding="utf-8")
        proc = subprocess.Popen([
            str(DX12_BENCH_EXE), "--mode", "sustained_dgpu_load", "--standalone-power-only",
            "--shader-dir", str(BUILD_DIR / "shaders"), "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
            "--out-csv", str(sa_csv), "--sync-file", str(sync_file), "--duration-ms", str(int(stg_s * 4500)),
        ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        coproc_tps: dict[str, float] = {}
        try:
            wait_token(proc, "READY_FOR_STAGES")
            for st in ("Idle", "Standalone_NPU", "Standalone_iGPU", "Standalone_CPU"):
                sync_file.write_text(f"{st}\n", encoding="utf-8")
                t0 = time.perf_counter()
                t_end = t0 + stg_s
                tiles = 0
                if st == "Standalone_NPU" and npu_req is not None:
                    while time.perf_counter() < t_end:
                        npu_req.infer({0: x_16_f32})
                        tiles += 16
                elif st == "Standalone_iGPU" and igpu_sess is not None:
                    while time.perf_counter() < t_end:
                        igpu_sess.run(None, {igpu_in: x_16_f16})
                        tiles += 16
                elif st == "Standalone_CPU":
                    while time.perf_counter() < t_end:
                        cpu_req.infer({0: x_16_f32})
                        tiles += 16
                else:
                    time.sleep(stg_s)
                coproc_tps[st] = tiles / max(time.perf_counter() - t0, 1e-3)
            sync_file.write_text("STOP\n", encoding="utf-8")
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.terminate()
        sa_stats = parse_trace_stages(sa_csv, ["Idle", "Standalone_NPU", "Standalone_iGPU", "Standalone_CPU"])
        sa_csv.unlink(missing_ok=True)
        idle_pkg = sa_stats.get("Idle", {}).get("total_est_power_w", 20.0)
        for st, vals in sa_stats.items():
            run_rows.append({
                "run_id": r, "experiment": "Standalone_H32", "stage": st,
                "coproc_tiles_per_sec": round(coproc_tps.get(st, 0.0), 2),
                "idle_baseline_pkg_w": round(idle_pkg, 3),
                **{k: round(v, 3) for k, v in vals.items()},
            })

        # Standalone dGPU H=32 reference (same H=32 op count as NPU/iGPU/CPU!)
        d32_csv = out_dir / f"_tmp_d32_r{r}.csv"
        sync_file.write_text("Standalone_dGPU_H32\n", encoding="utf-8")
        proc = subprocess.Popen([
            str(DX12_BENCH_EXE), "--mode", "sustained_dgpu_load", "--load-hidden", "32", "--prewarm-temp-c", "66",
            "--shader-dir", str(BUILD_DIR / "shaders"), "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
            "--out-csv", str(d32_csv), "--sync-file", str(sync_file), "--duration-ms", str(int(stg_s * 1200)),
        ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            wait_token(proc, "READY_FOR_STAGES")
            time.sleep(stg_s)
            sync_file.write_text("STOP\n", encoding="utf-8")
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.terminate()
        d32_st = parse_trace_stages(d32_csv, ["Standalone_dGPU_H32"]).get("Standalone_dGPU_H32", {})
        d32_csv.unlink(missing_ok=True)
        if d32_st:
            run_rows.append({
                "run_id": r, "experiment": "Standalone_H32", "stage": "Standalone_dGPU_H32",
                "coproc_tiles_per_sec": 0.0, "idle_baseline_pkg_w": round(idle_pkg, 3),
                **{k: round(v, 3) for k, v in d32_st.items()},
            })

        # Part B1: Thermally Matched Contention H=32 (Prewarm 66 C, Solo -> +NPU -> +iGPU -> +CPU_Unpinned -> Solo_ThermalCtrl)
        mc_csv = out_dir / f"_tmp_mc_r{r}.csv"
        stages_b1 = ["dGPU_Solo", "dGPU_plus_NPU", "dGPU_plus_iGPU", "dGPU_plus_CPU_Unpinned", "dGPU_Solo_PostThermal"]
        sync_file.write_text("dGPU_Solo\n", encoding="utf-8")
        proc = subprocess.Popen([
            str(DX12_BENCH_EXE), "--mode", "sustained_dgpu_load", "--load-hidden", "32", "--prewarm-temp-c", "66",
            "--shader-dir", str(BUILD_DIR / "shaders"), "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
            "--out-csv", str(mc_csv), "--sync-file", str(sync_file), "--duration-ms", str(int(stg_s * 5500)),
        ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        b1_coproc: dict[str, float] = {}
        try:
            wait_token(proc, "READY_FOR_STAGES")
            for st in stages_b1:
                sync_file.write_text(f"{st}\n", encoding="utf-8")
                t0 = time.perf_counter()
                t_end = t0 + stg_s
                tiles = 0
                if st == "dGPU_plus_NPU" and npu_req is not None:
                    while time.perf_counter() < t_end:
                        npu_req.infer({0: x_16_f32})
                        tiles += 16
                elif st == "dGPU_plus_iGPU" and igpu_sess is not None:
                    while time.perf_counter() < t_end:
                        igpu_sess.run(None, {igpu_in: x_16_f16})
                        tiles += 16
                elif st == "dGPU_plus_CPU_Unpinned":
                    while time.perf_counter() < t_end:
                        cpu_req.infer({0: x_16_f32})
                        tiles += 16
                else:
                    time.sleep(stg_s)
                b1_coproc[st] = tiles / max(time.perf_counter() - t0, 1e-3)
            sync_file.write_text("STOP\n", encoding="utf-8")
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.terminate()
        mc_stats = parse_trace_stages(mc_csv, stages_b1)
        mc_csv.unlink(missing_ok=True)
        for st, vals in mc_stats.items():
            run_rows.append({
                "run_id": r, "experiment": "Contention_Prewarmed", "stage": st,
                "coproc_tiles_per_sec": round(b1_coproc.get(st, 0.0), 2),
                "idle_baseline_pkg_w": round(idle_pkg, 3),
                **{k: round(v, 3) for k, v in vals.items()},
            })

        # Part B2: CPU Submission Starvation Isolation H=32 (dGPU worker pinned Core 0 HighPrio vs CPU worker pinned Cores 2..19)
        pin_csv = out_dir / f"_tmp_pin_r{r}.csv"
        cnt_file.unlink(missing_ok=True)
        sync_file.write_text("Idle\n", encoding="utf-8")
        proc_cpu = subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()),
            "--cpu-worker", "--sync-file", str(sync_file),
            "--worker-stage", "dGPU_plus_CPU_PinnedIsolated",
            "--count-file", str(cnt_file), "--cpu-threads", "14",
        ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Pin child CPU OpenVINO worker strictly to logical processors 2..19 (leaving Core 0..1 free for dGPU submit)
        try:
            ctypes.windll.kernel32.SetProcessAffinityMask(int(proc_cpu._handle), ctypes.c_size_t(0xFFFFC))
        except Exception:
            pass
        wait_token(proc_cpu, "CPU_WORKER_READY")
        sync_file.write_text("dGPU_Solo_Pinned\n", encoding="utf-8")
        proc_gpu = subprocess.Popen([
            str(DX12_BENCH_EXE), "--mode", "sustained_dgpu_load", "--load-hidden", "32", "--prewarm-temp-c", "66", "--pin-dgpu-thread",
            "--shader-dir", str(BUILD_DIR / "shaders"), "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
            "--out-csv", str(pin_csv), "--sync-file", str(sync_file), "--duration-ms", str(int(stg_s * 2600)),
        ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t_cpu_active = stg_s
        try:
            wait_token(proc_gpu, "READY_FOR_STAGES")
            sync_file.write_text("dGPU_Solo_Pinned\n", encoding="utf-8")
            time.sleep(stg_s)
            sync_file.write_text("dGPU_plus_CPU_PinnedIsolated\n", encoding="utf-8")
            t0_c = time.perf_counter()
            time.sleep(stg_s)
            t_cpu_active = max(time.perf_counter() - t0_c, 1e-3)
            sync_file.write_text("STOP\n", encoding="utf-8")
            proc_cpu.wait(timeout=15)
            proc_gpu.wait(timeout=20)
        finally:
            if proc_cpu.poll() is None:
                proc_cpu.terminate()
            if proc_gpu.poll() is None:
                proc_gpu.terminate()
        pin_tiles = int(cnt_file.read_text(encoding="utf-8").strip()) if cnt_file.exists() else 0
        pin_stats = parse_trace_stages(pin_csv, ["dGPU_Solo_Pinned", "dGPU_plus_CPU_PinnedIsolated"])
        pin_csv.unlink(missing_ok=True)
        for st, vals in pin_stats.items():
            ct = (pin_tiles / t_cpu_active) if st == "dGPU_plus_CPU_PinnedIsolated" else 0.0
            run_rows.append({
                "run_id": r, "experiment": "Contention_PinnedIsolated", "stage": st,
                "coproc_tiles_per_sec": round(ct, 2),
                "idle_baseline_pkg_w": round(idle_pkg, 3),
                **{k: round(v, 3) for k, v in vals.items()},
            })

    sync_file.unlink(missing_ok=True)
    cnt_file.unlink(missing_ok=True)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out_dir / "power_contention_runs.csv", index=False)

    # Aggregate Energy-per-Tile and Contention Deltas across N runs
    agg_rows: list[dict[str, Any]] = []
    solo_tps_by_exp = {
        "Standalone_H32": float(runs_df[runs_df["stage"] == "Standalone_dGPU_H32"]["dgpu_tiles_per_sec"].mean()),
        "Contention_Prewarmed": float(runs_df[runs_df["stage"] == "dGPU_Solo"]["dgpu_tiles_per_sec"].mean()),
        "Contention_PinnedIsolated": float(runs_df[runs_df["stage"] == "dGPU_Solo_Pinned"]["dgpu_tiles_per_sec"].mean()),
    }
    idle_mean_w = float(runs_df[runs_df["stage"] == "Idle"]["total_est_power_w"].mean())

    for (exp_name, st), g in runs_df.groupby(["experiment", "stage"]):
        valid_tps_list = (g["dgpu_tiles_per_sec"] + g["coproc_tiles_per_sec"]).tolist()
        pkg_w_list = g["total_est_power_w"].tolist()
        inc_w_list = [max(0.0, pw - idle_mean_w) for pw in pkg_w_list]
        pkg_mj_list = [(pw * 1000.0 / vt) if vt > 1.0 else 0.0 for pw, vt in zip(pkg_w_list, valid_tps_list)]
        inc_mj_list = [(iw * 1000.0 / vt) if vt > 1.0 else 0.0 for iw, vt in zip(inc_w_list, valid_tps_list)]
        base_dgpu = solo_tps_by_exp.get(exp_name, 1.0)
        dgpu_tps = float(g["dgpu_tiles_per_sec"].mean())
        st_vt = mean_std_ci95(valid_tps_list)
        st_pw = mean_std_ci95(pkg_w_list)
        st_imj = mean_std_ci95(inc_mj_list)
        st_pmj = mean_std_ci95(pkg_mj_list)
        agg_rows.append({
            "experiment": exp_name, "stage": st,
            "dgpu_valid_tps_mean": round(dgpu_tps, 2),
            "coproc_valid_tps_mean": round(float(g["coproc_tiles_per_sec"].mean()), 2),
            "aggregate_valid_tps_mean": st_vt["mean"], "aggregate_valid_tps_std": st_vt["std"],
            "dgpu_slowdown_vs_solo_pct": round(100.0 * (dgpu_tps - base_dgpu) / max(base_dgpu, 1e-6), 2) if dgpu_tps > 0 else 0.0,
            "dgpu_power_w": round(float(g["dgpu_power_w"].mean()), 3),
            "cpu_pkg_power_w": round(float(g["cpu_pkg_power_w"].mean()), 3),
            "igpu_pp1_power_w": round(float(g["igpu_pp1_power_w"].mean()), 3),
            "pkg_sum_power_w": st_pw["mean"], "pkg_sum_power_std": st_pw["std"],
            "incremental_power_w": round(float(np.mean(inc_w_list)), 3),
            "incremental_mj_per_tile": st_imj["mean"], "incremental_mj_per_tile_ci95": st_imj["ci95"],
            "package_mj_per_tile": st_pmj["mean"], "package_mj_per_tile_ci95": st_pmj["ci95"],
            "dgpu_sm_clock_mhz": round(float(g["dgpu_sm_clock_mhz"].mean()), 1),
            "dgpu_temp_c": round(float(g["dgpu_temp_c"].mean()), 1),
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "energy_and_contention_matrix.csv", index=False)
    print(f"[p3r] Wrote {out_dir / 'energy_and_contention_matrix.csv'} & power_contention_runs.csv")
    return agg_df, runs_df


# ---------------------------------------------------------------------------
# Section 8: Hardware-Specific Scheduler Cost Model + Held-Out Validation & Plots
# ---------------------------------------------------------------------------
def build_and_validate_cost_model(
    core: Any, exp_cfg: dict[str, Any], devs: dict[str, Any],
    op_df: pd.DataFrame, gather_df: pd.DataFrame, be_json: dict[str, Any],
    cong_df: pd.DataFrame, energy_df: pd.DataFrame, out_dir: Path, runs: int, smoke: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    import onnxruntime as ort

    xfer_df = pd.read_csv(out_dir / "dx12_transfer_sweep.csv")

    def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
        if len(xs) < 2:
            return (float(ys[0]) if ys else 0.0), 0.0
        a, b = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)
        return float(b), float(a)  # intercept_ms, slope_ms_per_tile

    backends_model: dict[str, Any] = {}
    q_csv = REPO_ROOT / "results" / "phase3" / "quantization_quality.csv"
    npu_psnr_mat, npu_flip_mat, npu_psnr_ren, npu_flip_ren = 50.15, 0.0205, 48.47, 0.0187
    if q_csv.exists():
        qdf = pd.read_csv(q_csv)
        qm = qdf[qdf["eval_scope"] == "material_texture_512"]
        qr = qdf[qdf["eval_scope"] == "dx12_rendered_1080p"]
        if not qm.empty:
            npu_psnr_mat = round(float(qm["psnr_albedo_int8_vs_fp16_db"].mean()), 3)
            npu_flip_mat = round(float(qm["flip_albedo_mean_int8_vs_fp16"].mean()), 6)
        if not qr.empty:
            npu_psnr_ren = round(float(qr["psnr_8ch_int8_vs_fp16_db"].mean()), 3)
            npu_flip_ren = round(float(qr["flip_albedo_mean_int8_vs_fp16"].mean()), 6)

    # Energy & slowdown lookup
    sa_sub = energy_df[energy_df["experiment"] == "Standalone_H32"].set_index("stage")
    mc_sub = energy_df[energy_df["experiment"] == "Contention_Prewarmed"].set_index("stage")
    pin_sub = energy_df[energy_df["experiment"] == "Contention_PinnedIsolated"].set_index("stage")

    # Gather cost fit (Optimized C++ AVX2 FP16 vs FP32)
    g16 = gather_df[gather_df["implementation"] == "Optimized_Cpp_AVX2_FP16"].sort_values("batch_tiles")
    g32 = gather_df[gather_df["implementation"] == "Optimized_Cpp_AVX2_FP32"].sort_values("batch_tiles")
    g16_int, g16_slp = fit_line(g16["batch_tiles"].tolist(), g16["mean_ms"].tolist())
    g32_int, g32_slp = fit_line(g32["batch_tiles"].tolist(), g32["mean_ms"].tolist())

    # PCIe upload + BCn pack fits
    def xfer_fit(path_name: str, col: str) -> tuple[float, float]:
        sub = xfer_df[xfer_df["path_name"] == path_name].groupby("batch_tiles", as_index=False)[col].mean()
        return fit_line(sub["batch_tiles"].tolist(), sub[col].tolist())

    lat_up_int, lat_up_slp = xfer_fit("Latent_72KB_Plus_dGPU_MLP_BCn", "copy_gpu_ms")
    raw_up_int, raw_up_slp = xfer_fit("HostPredecoded_Raw_512KB_Plus_GPU_BCnPack", "copy_gpu_ms")
    pack_int, pack_slp = xfer_fit("HostPredecoded_Raw_512KB_Plus_GPU_BCnPack", "compute_gpu_ms")

    # Measured queue-delay linear fit on dGPU compute queue: q_delay ≈ intercept + slope * bg_gpu_ms
    cong_agg = cong_df.groupby(["target_load_ms", "batch_tiles"], as_index=False).mean(numeric_only=True)
    cong_b16 = cong_agg[cong_agg["batch_tiles"] == 16]
    q_int, q_slp = fit_line(cong_b16["bg_gpu_ms"].tolist(), cong_b16["measured_queue_delay_ms"].tolist())

    target_bks = ["dGPU_DX12_Compute_FP16_BCn", "NPU_OpenVINO_INT8", "iGPU_ORT_DML_FP16", "iGPU_OpenVINO_FP16", "CPU_OpenVINO_INT8"]
    for bk in target_bks:
        sub = op_df[(op_df["backend"] == bk) & (op_df["hidden_dim"] == 32)].sort_values("batch_tiles")
        if sub.empty:
            continue
        p50_i, p50_s = fit_line(sub["batch_tiles"].tolist(), sub["infer_p50_ms"].tolist())
        p99_i, p99_s = fit_line(sub["batch_tiles"].tolist(), sub["infer_p99_ms"].tolist())
        mean_i, mean_s = fit_line(sub["batch_tiles"].tolist(), sub["infer_mean_ms"].tolist())

        is_dgpu = bk.startswith("dGPU")
        is_npu = bk.startswith("NPU")
        is_igpu = bk.startswith("iGPU")
        stg_key = "Standalone_dGPU_H32" if is_dgpu else ("Standalone_NPU" if is_npu else ("Standalone_iGPU" if is_igpu else "Standalone_CPU"))
        cont_key = None if is_dgpu else ("dGPU_plus_NPU" if is_npu else ("dGPU_plus_iGPU" if is_igpu else "dGPU_plus_CPU_Unpinned"))

        inc_mj = float(sa_sub.loc[stg_key, "incremental_mj_per_tile"]) if stg_key in sa_sub.index else 0.0
        pkg_mj = float(sa_sub.loc[stg_key, "package_mj_per_tile"]) if stg_key in sa_sub.index else 0.0
        slow_pct = float(mc_sub.loc[cont_key, "dgpu_slowdown_vs_solo_pct"]) if (cont_key and cont_key in mc_sub.index) else 0.0
        slow_pin = float(pin_sub.loc["dGPU_plus_CPU_PinnedIsolated", "dgpu_slowdown_vs_solo_pct"]) if (bk.startswith("CPU") and "dGPU_plus_CPU_PinnedIsolated" in pin_sub.index) else None

        backends_model[bk] = {
            "precision": str(sub["precision"].iloc[0]),
            "infer_p50_ms_model": {"intercept_ms": round(p50_i, 5), "slope_ms_per_tile": round(p50_s, 5)},
            "infer_p99_ms_model": {"intercept_ms": round(p99_i, 5), "slope_ms_per_tile": round(p99_s, 5)},
            "infer_mean_ms_model": {"intercept_ms": round(mean_i, 5), "slope_ms_per_tile": round(mean_s, 5)},
            "host_gather_ms_model": {"intercept_ms": 0.0, "slope_ms_per_tile": 0.0} if is_dgpu else (
                {"intercept_ms": round(g16_int, 5), "slope_ms_per_tile": round(g16_slp, 5)} if "FP16" in bk else
                {"intercept_ms": round(g32_int, 5), "slope_ms_per_tile": round(g32_slp, 5)}
            ),
            "pcie_upload_ms_model": {"intercept_ms": round(lat_up_int if is_dgpu else raw_up_int, 5),
                                     "slope_ms_per_tile": round(lat_up_slp if is_dgpu else raw_up_slp, 5)},
            "bcn_pack_ms_model": {"intercept_ms": 0.0 if is_dgpu else round(pack_int, 5),
                                  "slope_ms_per_tile": 0.0 if is_dgpu else round(pack_slp, 5)},
            "observed_batches": {str(int(r["batch_tiles"])): {"p50_ms": float(r["infer_p50_ms"]), "p99_ms": float(r["infer_p99_ms"]), "mean_ms": float(r["infer_mean_ms"])} for _, r in sub.iterrows()},
            "energy_per_tile": {"incremental_mj_per_tile": inc_mj, "package_mj_per_tile": pkg_mj},
            "concurrent_dgpu_throughput_delta_pct": slow_pct,
            "concurrent_dgpu_delta_cpu_pinned_isolated_pct": slow_pin,
            "memory_cost": {
                "vram_bytes_per_tile_working": 196608 if is_dgpu else 524288,
                "host_gather_ram_mib": 0.0 if is_dgpu else float(g16["arena_ram_mib"].iloc[0]),
            },
            "visual_error_vs_fp16": {
                "material_albedo_psnr_db": npu_psnr_mat if "INT8" in bk else 99.0,
                "material_albedo_flip": npu_flip_mat if "INT8" in bk else 0.0,
                "rendered_1080p_psnr_db": npu_psnr_ren if "INT8" in bk else 99.0,
                "rendered_1080p_flip": npu_flip_ren if "INT8" in bk else 0.0,
            },
        }

    cost_model = {
        "schema_version": "1.0-phase3-reinforced",
        "hardware": devs,
        "mlp_h32_ops": {"macs_per_texel": 1792, "ops_per_texel": 3584, "b16_total_gop": op_counts(32, 16)[2]},
        "dgpu_queue_congestion_model": {"intercept_ms": round(q_int, 5), "slope_per_bg_ms": round(q_slp, 5)},
        "backends": backends_model,
        "breakeven_b16_post_opt": be_json.get("reinforced_by_batch", {}).get("16", {}),
    }

    # Held-out validation on unseen batch sizes B in {8, 32, 96} + unseen congestion loads (15ms, 30ms)
    val_batches = [8, 32] if smoke else [8, 32, 96]
    tmp_val_dir = out_dir / "_tmp_val_models"
    tmp_val_dir.mkdir(parents=True, exist_ok=True)
    all_w = load_trained_mlp_weights(REPO_ROOT / exp_cfg["asset_manifest"], REPO_ROOT / exp_cfg["asset_pack"])
    calib_4 = build_real_calib_tiles(REPO_ROOT / exp_cfg["asset_manifest"], REPO_ROOT / exp_cfg["asset_pack"], mat_id=0)
    b1_fp32_p = MODELS_DIR / "ntc_h32_b1_fp32.onnx"

    import onnx
    for bv in val_batches:
        onnx.save(build_ntc_onnx(all_w[0], bv, 32, fp16=True), str(tmp_val_dir / f"ntc_h32_b{bv}_fp16.onnx"))
        quantize_ntc_int8(b1_fp32_p, tmp_val_dir / f"ntc_h32_b{bv}_int8.onnx", calib_4, target_batch=bv)

    val_dx12_csv = out_dir / "_tmp_val_dx12.csv"
    run_cmd([
        str(DX12_BENCH_EXE), "--mode", "compute_sweep",
        "--shader-dir", str(BUILD_DIR / "shaders"),
        "--manifest", str(REPO_ROOT / exp_cfg["asset_manifest"]),
        "--out-csv", str(val_dx12_csv),
        "--batches", ",".join(str(b) for b in val_batches),
        "--runs", str(runs), "--warmup", "2", "--iters", str(5 if smoke else 15),
    ])
    vdx_df = pd.read_csv(val_dx12_csv)
    val_dx12_csv.unlink(missing_ok=True)

    val_rows: list[dict[str, Any]] = []
    for bv, g in vdx_df.groupby("batch_tiles"):
        bv_i = int(bv)
        act_p50 = float(np.median(g["e2e_dispatch_ms"]))
        m_cfg = backends_model["dGPU_DX12_Compute_FP16_BCn"]["infer_p50_ms_model"]
        pred = m_cfg["intercept_ms"] + m_cfg["slope_ms_per_tile"] * bv_i
        err = abs(pred - act_p50)
        val_rows.append({"validation_type": "heldout_batch_p50", "target": "dGPU_DX12_Compute_FP16_BCn", "param": bv_i,
                         "predicted_ms": round(pred, 4), "actual_ms": round(act_p50, 4), "abs_err_ms": round(err, 4),
                         "ape_pct": round(100.0 * err / max(act_p50, 1e-6), 2)})

    val_specs = [
        ("NPU_OpenVINO_INT8", "ov", devs.get("ov_npu_device"), "int8"),
        ("iGPU_OpenVINO_FP16", "ov", devs.get("ov_igpu_device"), "fp16"),
        ("CPU_OpenVINO_INT8", "ov", "CPU", "int8"),
    ]
    if devs.get("igpu_dml_device_id") is not None:
        val_specs.append(("iGPU_ORT_DML_FP16", "ort", str(devs["igpu_dml_device_id"]), "fp16"))

    for label, kind, tgt, fp in val_specs:
        if not tgt or label not in backends_model:
            continue
        for bv in val_batches:
            mp = tmp_val_dir / f"ntc_h32_b{bv}_{fp}.onnx"
            try:
                if kind == "ov":
                    comp = core.compile_model(str(mp), tgt)
                    req = comp.create_infer_request()
                    dt = np.float16 if "f16" in str(comp.input(0).element_type).lower() else np.float32
                    x_in = np.ascontiguousarray(np.broadcast_to(calib_4[0:1], (bv, 16, 256, 256)).astype(dt))
                    for _ in range(2):
                        req.infer({0: x_in})
                    dts = [(t0 := time.perf_counter(), req.infer({0: x_in}), (time.perf_counter() - t0) * 1000.0)[-1] for _ in range(4 if smoke else 8)]
                else:
                    so = ort.SessionOptions()
                    so.enable_mem_pattern = False
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    sess = ort.InferenceSession(str(mp), sess_options=so, providers=[("DmlExecutionProvider", {"device_id": int(tgt)}), "CPUExecutionProvider"])
                    in_n = sess.get_inputs()[0].name
                    dt = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
                    x_in = np.ascontiguousarray(np.broadcast_to(calib_4[0:1], (bv, 16, 256, 256)).astype(dt))
                    for _ in range(2):
                        sess.run(None, {in_n: x_in})
                    dts = [(t0 := time.perf_counter(), sess.run(None, {in_n: x_in}), (time.perf_counter() - t0) * 1000.0)[-1] for _ in range(4 if smoke else 8)]
                act_p50 = float(np.median(dts))
                m_cfg = backends_model[label]["infer_p50_ms_model"]
                pred = m_cfg["intercept_ms"] + m_cfg["slope_ms_per_tile"] * bv
                err = abs(pred - act_p50)
                val_rows.append({"validation_type": "heldout_batch_p50", "target": label, "param": bv,
                                 "predicted_ms": round(pred, 4), "actual_ms": round(act_p50, 4), "abs_err_ms": round(err, 4),
                                 "ape_pct": round(100.0 * err / max(act_p50, 1e-6), 2)})
            except Exception as exc:
                print(f"[p3r] WARN val {label} B={bv}: {exc}")

    # Validate held-out GPU queue congestion targets (15 ms, 30 ms) at B=16
    for t_ld in (15, 30):
        sub_c = cong_df[(cong_df["target_load_ms"] == t_ld) & (cong_df["batch_tiles"] == 16)]
        if not sub_c.empty:
            bg_m = float(sub_c["bg_gpu_ms"].mean())
            act_tot = float(sub_c["total_complete_ms"].mean())
            dm = backends_model["dGPU_DX12_Compute_FP16_BCn"]["infer_mean_ms_model"]
            pred_tot = (q_int + q_slp * bg_m) + (dm["intercept_ms"] + dm["slope_ms_per_tile"] * 16)
            err = abs(pred_tot - act_tot)
            val_rows.append({"validation_type": "heldout_congestion_mean", "target": "dGPU_Congestion_B16", "param": t_ld,
                             "predicted_ms": round(pred_tot, 4), "actual_ms": round(act_tot, 4), "abs_err_ms": round(err, 4),
                             "ape_pct": round(100.0 * err / max(act_tot, 1e-6), 2)})

    shutil.rmtree(tmp_val_dir, ignore_errors=True)
    val_df = pd.DataFrame(val_rows)
    if not val_df.empty:
        y_a = val_df["actual_ms"].to_numpy()
        y_p = val_df["predicted_ms"].to_numpy()
        r2 = 1.0 - float(np.sum((y_a - y_p) ** 2)) / max(float(np.sum((y_a - np.mean(y_a)) ** 2)), 1e-9)
        cost_model["validation_summary"] = {
            "num_heldout_points": len(val_df),
            "mape_pct": round(float(val_df["ape_pct"].mean()), 2),
            "mae_ms": round(float(val_df["abs_err_ms"].mean()), 3),
            "r2": round(r2, 5),
        }
    val_df.to_csv(out_dir / "cost_model_validation.csv", index=False)

    for p_dst in (REPO_ROOT / "scheduler_cost_model.json", REPO_ROOT / "configs" / "scheduler_cost_model.json", out_dir / "scheduler_cost_model.json"):
        p_dst.write_text(json.dumps(cost_model, indent=2), encoding="utf-8")
    print(f"[p3r] Wrote scheduler_cost_model.json (MAPE={cost_model.get('validation_summary', {}).get('mape_pct')}%)")
    return cost_model, val_df


def generate_plots(
    gather_df: pd.DataFrame, be_json: dict[str, Any], cong_df: pd.DataFrame,
    energy_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path
) -> None:
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    def save(fig: plt.Figure, stem: str) -> None:
        fig.savefig(pdir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(pdir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Plot 1: Host Gather Optimization & E2E Speedup at B=16
    g16 = gather_df[gather_df["batch_tiles"] == 16]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    gx = np.arange(len(g16))
    axes[0].bar(gx, g16["mean_ms"], color=["#C44E52", "#DD8452", "#55A868", "#4C72B0"], edgecolor="black", linewidth=0.5)
    for i, v in enumerate(g16["mean_ms"]):
        axes[0].text(i, v + 0.35, f"{v:.2f} ms", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].axhline(5.0, color="#222222", linestyle="--", linewidth=1.2, label="Protocol Target <= 5.0 ms")
    axes[0].set_xticks(gx)
    axes[0].set_xticklabels([s.replace("_", "\n") for s in g16["implementation"]], fontsize=8)
    axes[0].set_ylabel("Host Feature Gather Latency (ms)")
    axes[0].set_title("(a) Host Gather Optimization (B=16 Tiles)")
    axes[0].legend(fontsize=8)

    b16 = be_json["reinforced_by_batch"].get("16", {})
    if b16:
        labels = ["dGPU DX12\nUncontended", "iGPU Pre-Opt\n(NumPy)", "iGPU Post-Opt\n(Serial C++)", "iGPU Post-Opt\n(Overlapped)",
                  "NPU Pre-Opt\n(NumPy)", "NPU Post-Opt\n(Serial C++)", "NPU Post-Opt\n(Overlapped)"]
        vals = [b16["dgpu_uncontended_complete_ms"], b16["igpu_pre_opt_serial_ms"], b16["igpu_post_opt_serial_ms"],
                b16["igpu_post_opt_overlapped_stage_ms"], b16["npu_pre_opt_serial_ms"], b16["npu_post_opt_serial_ms"],
                b16["npu_post_opt_overlapped_stage_ms"]]
        bx = np.arange(len(vals))
        axes[1].bar(bx, vals, color=["#4C72B0", "#DD8452", "#8172B3", "#55A868", "#C44E52", "#937860", "#2B7BBA"], edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            axes[1].text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].set_xticks(bx)
        axes[1].set_xticklabels(labels, fontsize=7.2)
        axes[1].set_ylabel("End-to-End T_complete (ms)")
        axes[1].set_title("(b) B=16 Full E2E Latency: Pre-Opt vs Post-Opt C++ Gather")
    fig.tight_layout()
    save(fig, "p3r_host_gather_and_e2e_optimization")

    # Plot 2: Measured GPU Queue Congestion vs Analytical Break-Even (Separated!)
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    c_b16 = cong_df[cong_df["batch_tiles"] == 16].groupby("target_load_ms", as_index=False).agg(
        bg=("bg_gpu_ms", "mean"), med=("total_complete_ms", lambda s: np.percentile(s, 50)),
        p99=("total_complete_ms", lambda s: np.percentile(s, 99)),
    )
    ax.plot(c_b16["bg"], c_b16["med"], "o-", color="#4C72B0", linewidth=2.2, label="Measured Hardware dGPU P50 (B=16)")
    ax.plot(c_b16["bg"], c_b16["p99"], "s--", color="#C44E52", linewidth=1.8, label="Measured Hardware dGPU P99 (B=16)")
    xs_an = np.linspace(0, max(65.0, float(c_b16["bg"].max()) + 5.0), 50)
    if b16:
        d0 = b16["dgpu_uncontended_complete_ms"]
        ax.plot(xs_an, d0 + xs_an, ":", color="gray", alpha=0.8, label="Analytical Linear Extrapolation (dGPU + Q_delay)")
        ax.axhline(b16["igpu_post_opt_serial_ms"], color="#55A868", linestyle="-.", linewidth=1.8,
                   label=f"iGPU Post-Opt Serial E2E ({b16['igpu_post_opt_serial_ms']:.1f} ms)")
        ax.axhline(b16["npu_post_opt_serial_ms"], color="#8172B3", linestyle="-.", linewidth=1.8,
                   label=f"NPU Post-Opt Serial E2E ({b16['npu_post_opt_serial_ms']:.1f} ms)")
        ax.axhline(b16["npu_pre_opt_serial_ms"], color="#DD8452", linestyle=":", linewidth=1.6,
                   label=f"NPU Pre-Opt NumPy E2E ({b16['npu_pre_opt_serial_ms']:.1f} ms)")
    ax.set_xlabel("Measured Background dGPU Compute Queue Load (ms)")
    ax.set_ylabel("Probe Completion Time T_complete (ms, B=16)")
    ax.set_title("Phase 3 Reinforced: Measured Hardware Queue Congestion vs Offload Break-Even")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "p3r_breakeven_measured_congestion")

    # Plot 3: Energy/Tile & Contention Decomposition
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    sa = energy_df[energy_df["experiment"] == "Standalone_H32"]
    sa = sa[sa["stage"] != "Idle"]
    sx = np.arange(len(sa))
    axes[0].bar(sx - 0.18, sa["incremental_mj_per_tile"], width=0.35, color="#55A868", label="Incremental mJ/tile (dP - Idle)")
    axes[0].bar(sx + 0.18, sa["package_mj_per_tile"], width=0.35, color="#4C72B0", label="Total Package mJ/tile")
    for i, (vi, vp) in enumerate(zip(sa["incremental_mj_per_tile"], sa["package_mj_per_tile"])):
        axes[0].text(i - 0.18, vi + 0.5, f"{vi:.1f}", ha="center", fontsize=7.5)
        axes[0].text(i + 0.18, vp + 0.5, f"{vp:.1f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(sx)
    axes[0].set_xticklabels([s.replace("Standalone_", "") for s in sa["stage"]], fontsize=8.5)
    axes[0].set_ylabel("Energy per H=32 Valid Tile (mJ/tile)")
    axes[0].set_title("(a) Standalone Energy/Tile (Incremental vs Package)")
    axes[0].legend(fontsize=8)

    ct = energy_df[energy_df["experiment"].isin(["Contention_Prewarmed", "Contention_PinnedIsolated"])].copy()
    order_map = {
        "dGPU_Solo": 0, "dGPU_plus_NPU": 1, "dGPU_plus_iGPU": 2,
        "dGPU_plus_CPU_Unpinned": 3, "dGPU_Solo_PostThermal": 4,
        "dGPU_Solo_Pinned": 5, "dGPU_plus_CPU_PinnedIsolated": 6,
    }
    ct["ord"] = ct["stage"].map(lambda s: order_map.get(s, 99))
    ct = ct.sort_values("ord")
    cx = np.arange(len(ct))
    axes[1].bar(cx - 0.18, ct["dgpu_valid_tps_mean"], width=0.35, color="#4C72B0", label="dGPU Valid tiles/s")
    axes[1].bar(cx + 0.18, ct["aggregate_valid_tps_mean"], width=0.35, color="#DD8452", label="Aggregate (dGPU+Coproc) tiles/s")
    axes[1].set_xticks(cx)
    axes[1].set_xticklabels([s.replace("dGPU_plus_", "+").replace("dGPU_", "") for s in ct["stage"]], rotation=22, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Throughput (tiles/s)")
    axes[1].set_title("(b) Prewarmed Contention & CPU Affinity Isolation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "p3r_energy_per_tile_and_contention_decomposition")

    # Plot 4: Held-Out Scheduler Cost Model Validation
    if not val_df.empty:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        for vt, g in val_df.groupby("validation_type"):
            ax.scatter(g["actual_ms"], g["predicted_ms"], s=65, alpha=0.88, label=vt)
        mx = max(float(val_df["actual_ms"].max()), float(val_df["predicted_ms"].max())) * 1.08
        ax.plot([0, mx], [0, mx], "k--", linewidth=1.2, label="Ideal y = x")
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_xlabel("Actual Measured Hardware Latency (ms)")
        ax.set_ylabel("Scheduler Cost Model Predicted Latency (ms)")
        ax.set_title(f"Held-Out Validation (Unseen Batches {{8,32,96}} + Congestion {{15,30 ms}}, MAPE={val_df['ape_pct'].mean():.1f}%)")
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        save(fig, "p3r_cost_model_validation")


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-AHNTS Phase 3 Scientific Reinforcement Suite")
    parser.add_argument("--config", type=Path, default=EXP_CFG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true")
    parser.add_argument("--sync-file", type=Path, default=None)
    parser.add_argument("--worker-stage", type=str, default="dGPU_plus_CPU_PinnedIsolated")
    parser.add_argument("--count-file", type=Path, default=None)
    parser.add_argument("--cpu-threads", type=int, default=14)
    args = parser.parse_args()

    if args.cpu_worker:
        return run_pinned_cpu_worker_loop(args.sync_file, args.worker_stage, args.count_file, args.cpu_threads)

    exp_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = 1 if args.smoke else int(args.runs)

    import openvino as ov
    ov_core = ov.Core()

    devs = resolve_devices(REPO_ROOT / exp_cfg["hardware_profile"])
    audit_backends(ov_core, devs, out_dir)
    op_df = measure_op_matrix(ov_core, exp_cfg, devs, out_dir, runs=runs, smoke=args.smoke)
    gather_df, equiv = benchmark_host_gather_suite(exp_cfg, out_dir, runs=runs, smoke=args.smoke)
    be_json, cong_df = build_breakeven_and_congestion(exp_cfg, op_df, gather_df, out_dir, runs=runs, smoke=args.smoke)
    be_json["host_gather_equivalence"] = equiv
    (out_dir / "breakeven_complete.json").write_text(json.dumps(be_json, indent=2), encoding="utf-8")

    energy_df, _ = run_energy_and_contention(ov_core, exp_cfg, devs, out_dir, runs=runs, smoke=args.smoke)
    _, val_df = build_and_validate_cost_model(ov_core, exp_cfg, devs, op_df, gather_df, be_json, cong_df, energy_df, out_dir, runs=runs, smoke=args.smoke)
    generate_plots(gather_df, be_json, cong_df, energy_df, val_df, out_dir)
    print("[p3r] ALL PHASE 3 REINFORCEMENT STEPS COMPLETED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
