#!/usr/bin/env python3
"""Automated benchmark runner & quality evaluator for PA-AHNTS."""

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
from typing import Any

import flip_evaluator
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
BENCH_EXE = BUILD_DIR / "pa_ahnts_bench.exe"


def run_cmd(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print(f"[run_phase] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd))


def ensure_phase1_assets_and_binary(force_rebuild: bool = False) -> None:
    manifest = REPO_ROOT / "results" / "assets_cache" / "manifest.json"
    if not manifest.exists():
        run_cmd([sys.executable, "tools/asset_gen/generate_pbr_assets.py"])

    if force_rebuild or not BENCH_EXE.exists():
        run_cmd(["cmake", "-S", ".", "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
        run_cmd(["cmake", "--build", str(BUILD_DIR), "--target", "pa_ahnts_bench"])


def compute_psnr_db(ref_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    mse = float(np.mean((ref_rgb - test_rgb) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def evaluate_quality(
    exp_cfg: dict[str, Any], config_names: list[str], out_dir: Path, ref_cap_dir: Path | None = None
) -> list[dict[str, Any]]:
    cam_cfg = json.loads((REPO_ROOT / exp_cfg["camera_path"]).read_text(encoding="utf-8"))
    checkpoints = [ck["id"] for ck in cam_cfg["quality_checkpoints"]]
    ref_cfg = exp_cfg["reference_config"]
    cap_dir = out_dir / "captures"
    ref_dir = ref_cap_dir or cap_dir

    rows: list[dict[str, Any]] = []
    for b in config_names:
        for ck_id in checkpoints:
            ref_path = ref_dir / f"{ref_cfg}_{ck_id}.png"
            test_path = cap_dir / f"{b}_{ck_id}.png"
            if not ref_path.exists() or not test_path.exists():
                raise RuntimeError(f"Missing checkpoint image: {ref_path} or {test_path}")

            ref_arr = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
            test_arr = np.asarray(Image.open(test_path).convert("RGB"), dtype=np.float32) / 255.0

            psnr_db = compute_psnr_db(ref_arr, test_arr)
            if b == ref_cfg and ref_dir == cap_dir:
                flip_mean = 0.0
                flip_p99 = 0.0
            else:
                flip_map, flip_mean, _ = flip_evaluator.evaluate(
                    ref_arr, test_arr, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
                )
                flip_p99 = float(np.percentile(flip_map, 99.0))

            rows.append(
                {
                    "config_name": b,
                    "camera_checkpoint_id": ck_id,
                    "psnr_db": round(psnr_db, 4),
                    "flip_mean": round(float(flip_mean), 6),
                    "flip_p99": round(flip_p99, 6),
                    "reference_config": ref_cfg,
                }
            )

    q_csv = out_dir / "quality_metrics.csv"
    with q_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config_name", "camera_checkpoint_id", "psnr_db", "flip_mean", "flip_p99", "reference_config"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_phase] Wrote {q_csv} ({len(rows)} rows)")
    return rows


def mean_std_ci95(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = statistics.mean(vals) if n > 0 else 0.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    t_table = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    t_val = t_table.get(n, 1.96)
    ci95 = (t_val * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": round(m, 4), "std": round(sd, 4), "ci95": round(ci95, 4)}


def summarize_phase(
    phase: int,
    config_names: list[str],
    runs_per_config: int,
    out_dir: Path,
    quality_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    f_csv = out_dir / "frame_metrics.csv"
    m_csv = out_dir / "memory_power.csv"
    p_csv = out_dir / "pcie_transfer_trace.csv"

    frame_by_run: dict[tuple[str, int], list[dict[str, float]]] = {}
    with f_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["config_name"], int(row["run_id"]))
            frame_by_run.setdefault(key, []).append(
                {
                    "cpu_ms": float(row["cpu_frame_ms"]),
                    "gpu_ms": float(row["gpu_frame_ms"]),
                    "present_ms": float(row["present_ms"]),
                    "pcie_mb": float(row["pcie_upload_mb"]),
                    "tiles_dgpu": float(row["tiles_decoded_dgpu"]),
                }
            )

    mem_by_run: dict[tuple[str, int], list[dict[str, float]]] = {}
    with m_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["config_name"], int(row["run_id"]))
            mem_by_run.setdefault(key, []).append(
                {
                    "local_mb": float(row["local_vram_mb"]),
                    "nonlocal_mb": float(row["nonlocal_vram_mb"]),
                    "sys_ram_mb": float(row["sys_ram_latent_mb"]),
                    "dgpu_w": float(row["dgpu_power_w"]),
                    "cpu_w": float(row["cpu_pkg_power_w"]),
                    "total_w": float(row["total_est_power_w"]),
                    "clk_mhz": float(row["dgpu_clock_mhz"]),
                    "temp_c": float(row["dgpu_temp_c"]),
                }
            )

    pcie_by_cfg: dict[str, list[dict[str, float]]] = {}
    if p_csv.exists():
        with p_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pcie_by_cfg.setdefault(row["config_name"], []).append(
                    {
                        "lat_us": float(row["copy_gpu_latency_us"]),
                        "bw_gbps": float(row["bandwidth_gbps"]),
                        "upload_kb": float(row["upload_kb"]),
                    }
                )

    p1_b0_vram, p1_b3_vram = None, None
    p1_sum_path = REPO_ROOT / "results" / "phase1" / "phase1_summary.json"
    if phase == 2 and p1_sum_path.exists():
        p1_sum = json.loads(p1_sum_path.read_text(encoding="utf-8"))
        p1_b0_vram = p1_sum["configs"]["B0_BCn_Full"]["peak_local_vram_mb"]["mean"]
        p1_b3_vram = p1_sum["configs"]["B3_NTC_OnFeedback"]["peak_local_vram_mb"]["mean"]

    summary: dict[str, Any] = {"phase": phase, "runs_per_config": runs_per_config, "configs": {}}
    for b in config_names:
        run_ids = sorted({rid for (cfg, rid) in frame_by_run.keys() if cfg == b})
        fps_runs, cpu_ms_runs, gpu_ms_runs, p99_cpu_runs, p99_gpu_runs = [], [], [], [], []
        pcie_tot_runs, decoded_tot_runs = [], []
        peak_local_runs, peak_nonlocal_runs, peak_ram_runs = [], [], []
        dgpu_w_runs, cpu_w_runs, total_w_runs = [], [], []

        for rid in run_ids:
            fr = frame_by_run[(b, rid)]
            valid_gpu = [x["gpu_ms"] for x in fr if x["gpu_ms"] > 0.0]
            cpu_list = [x["cpu_ms"] for x in fr]
            mean_cpu = statistics.mean(cpu_list)
            mean_gpu = statistics.mean(valid_gpu) if valid_gpu else 0.0
            fps_runs.append(1000.0 / max(mean_cpu, 1e-4))
            cpu_ms_runs.append(mean_cpu)
            gpu_ms_runs.append(mean_gpu)
            p99_cpu_runs.append(float(np.percentile(cpu_list, 99.0)))
            p99_gpu_runs.append(float(np.percentile(valid_gpu, 99.0)) if valid_gpu else 0.0)
            pcie_tot_runs.append(sum(x["pcie_mb"] for x in fr))
            decoded_tot_runs.append(sum(x["tiles_dgpu"] for x in fr))

            mr = mem_by_run.get((b, rid), [])
            if mr:
                peak_local_runs.append(max(x["local_mb"] for x in mr))
                peak_nonlocal_runs.append(max(x["nonlocal_mb"] for x in mr))
                peak_ram_runs.append(max(x["sys_ram_mb"] for x in mr))
                dgpu_w_runs.append(statistics.mean(x["dgpu_w"] for x in mr))
                cpu_w_runs.append(statistics.mean(x["cpu_w"] for x in mr))
                total_w_runs.append(statistics.mean(x["total_w"] for x in mr))

        q_sub = [q for q in quality_rows if q["config_name"] == b]
        entry: dict[str, Any] = {
            "avg_fps": mean_std_ci95(fps_runs),
            "avg_cpu_frame_ms": mean_std_ci95(cpu_ms_runs),
            "p99_cpu_frame_ms": mean_std_ci95(p99_cpu_runs),
            "avg_gpu_frame_ms": mean_std_ci95(gpu_ms_runs),
            "p99_gpu_frame_ms": mean_std_ci95(p99_gpu_runs),
            "peak_local_vram_mb": mean_std_ci95(peak_local_runs),
            "peak_nonlocal_vram_mb": mean_std_ci95(peak_nonlocal_runs),
            "peak_sys_ram_latent_mb": mean_std_ci95(peak_ram_runs),
            "total_pcie_upload_mb": mean_std_ci95(pcie_tot_runs),
            "total_tiles_decoded_dgpu": mean_std_ci95(decoded_tot_runs),
            "avg_dgpu_power_w": mean_std_ci95(dgpu_w_runs),
            "avg_cpu_pkg_power_w": mean_std_ci95(cpu_w_runs),
            "avg_total_power_w": mean_std_ci95(total_w_runs),
            "mean_psnr_db": round(statistics.mean(q["psnr_db"] for q in q_sub), 4) if q_sub else None,
            "mean_flip": round(statistics.mean(q["flip_mean"] for q in q_sub), 6) if q_sub else None,
            "max_flip_p99": round(max(q["flip_p99"] for q in q_sub), 6) if q_sub else None,
        }
        if p1_b0_vram is not None and peak_local_runs:
            loc_mean = entry["peak_local_vram_mb"]["mean"]
            entry["vram_savings_vs_b0_mb"] = round(p1_b0_vram - loc_mean, 2)
            entry["vram_savings_vs_b3_mb"] = round(p1_b3_vram - loc_mean, 2)
        pt = pcie_by_cfg.get(b, [])
        if pt:
            lats = [x["lat_us"] for x in pt if x["lat_us"] > 0.0]
            bws = [x["bw_gbps"] for x in pt if x["bw_gbps"] > 0.0]
            entry["pcie_copy_latency_us"] = {
                "p50": round(float(np.percentile(lats, 50.0)), 2) if lats else 0.0,
                "p95": round(float(np.percentile(lats, 95.0)), 2) if lats else 0.0,
                "p99": round(float(np.percentile(lats, 99.0)), 2) if lats else 0.0,
                "mean_bw_gbps": round(statistics.mean(bws), 3) if bws else 0.0,
            }
        summary["configs"][b] = entry

    sum_path = out_dir / f"phase{phase}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[run_phase] Summary written to {sum_path}")
    print(
        f"{'Config':<22} {'FPS':>7} {'CPUms':>7} {'GPUms':>7} {'P99ms':>7} "
        f"{'LocMB':>8} {'RamL2':>8} {'PCIeMB':>7} {'DecT':>6} {'TotW':>6} {'PSNR':>6} {'FLIP':>7}"
    )
    for b in config_names:
        c = summary["configs"][b]
        print(
            f"{b:<22} "
            f"{c['avg_fps']['mean']:7.1f} "
            f"{c['avg_cpu_frame_ms']['mean']:7.2f} "
            f"{c['avg_gpu_frame_ms']['mean']:7.2f} "
            f"{c['p99_cpu_frame_ms']['mean']:7.2f} "
            f"{c['peak_local_vram_mb']['mean']:8.1f} "
            f"{c['peak_sys_ram_latent_mb']['mean']:8.1f} "
            f"{c['total_pcie_upload_mb']['mean']:7.1f} "
            f"{c['total_tiles_decoded_dgpu']['mean']:6.0f} "
            f"{c['avg_total_power_w']['mean']:6.2f} "
            f"{(c['mean_psnr_db'] or 0.0):6.2f} "
            f"{(c['mean_flip'] or 0.0):7.4f}"
        )
    return summary


def run_phase1(args: argparse.Namespace) -> int:
    cfg_path = REPO_ROOT / "configs" / "experiments" / "phase1_baselines.json"
    exp_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "results" / ("phase1_smoke" if args.smoke else "phase1")

    ensure_phase1_assets_and_binary(force_rebuild=args.rebuild)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = 1 if args.smoke else int(args.runs or exp_cfg["runs"])
    warmup = 10 if args.smoke else int(exp_cfg["warmup_frames"])
    frames = 30 if args.smoke else int(exp_cfg["measured_frames"])

    for b in exp_cfg["baselines"]:
        for r in range(runs):
            run_cmd([
                str(BENCH_EXE), "--phase", "1", "--config", str(cfg_path),
                "--baseline", b, "--run-id", str(r),
                "--warmup", str(warmup), "--frames", str(frames),
                "--out-dir", str(out_dir), "--shader-dir", str(BUILD_DIR / "shaders"),
            ])

    if not args.smoke:
        for b in exp_cfg["baselines"]:
            run_cmd([
                str(BENCH_EXE), "--phase", "1", "--config", str(cfg_path),
                "--baseline", b, "--run-id", "0",
                "--out-dir", str(out_dir), "--shader-dir", str(BUILD_DIR / "shaders"),
                "--capture-checkpoints",
            ])
        q_rows = evaluate_quality(exp_cfg, exp_cfg["baselines"], out_dir)
        summarize_phase(1, exp_cfg["baselines"], runs, out_dir, q_rows)
    else:
        print("[run_phase] Phase 1 smoke test passed!")
    return 0


def run_phase2(args: argparse.Namespace) -> int:
    cfg_path = REPO_ROOT / "configs" / "experiments" / "phase2_ram_tier.json"
    exp_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "results" / ("phase2_smoke" if args.smoke else "phase2")

    ensure_phase1_assets_and_binary(force_rebuild=args.rebuild)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = exp_cfg["configs"]
    if args.smoke:
        cfgs = [cfgs[0], cfgs[2]]  # R1 (2.2GB) and R3 (4.0GB)

    runs = 1 if args.smoke else int(args.runs or exp_cfg["runs"])
    warmup = 10 if args.smoke else int(exp_cfg["warmup_frames"])
    frames = 30 if args.smoke else int(exp_cfg["measured_frames"])

    for c in cfgs:
        for r in range(runs):
            run_cmd([
                str(BENCH_EXE), "--phase", "2", "--config", str(cfg_path),
                "--config-name", c["name"], "--run-id", str(r),
                "--l1-mib", str(c["l1_latent_pool_mib"]),
                "--ram-l2-mib", str(c["ram_l2_pool_mib"]),
                "--warmup", str(warmup), "--frames", str(frames),
                "--out-dir", str(out_dir), "--shader-dir", str(BUILD_DIR / "shaders"),
            ])

    if not args.smoke:
        ref_cap_dir = REPO_ROOT / exp_cfg["reference_capture_dir"]
        for c in cfgs:
            run_cmd([
                str(BENCH_EXE), "--phase", "2", "--config", str(cfg_path),
                "--config-name", c["name"], "--run-id", "0",
                "--l1-mib", str(c["l1_latent_pool_mib"]),
                "--ram-l2-mib", str(c["ram_l2_pool_mib"]),
                "--out-dir", str(out_dir), "--shader-dir", str(BUILD_DIR / "shaders"),
                "--capture-checkpoints",
            ])
        names = [c["name"] for c in cfgs]
        q_rows = evaluate_quality(exp_cfg, names, out_dir, ref_cap_dir=ref_cap_dir)
        summarize_phase(2, names, runs, out_dir, q_rows)
    else:
        print("[run_phase] Phase 2 smoke test passed!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-AHNTS Phase Runner")
    parser.add_argument("--phase", type=int, default=1, help="Phase index (1..6)")
    parser.add_argument("--runs", type=int, default=None, help="Repeated runs per config")
    parser.add_argument("--smoke", action="store_true", help="Run fast 30-frame smoke test")
    parser.add_argument("--rebuild", action="store_true", help="Force CMake rebuild")
    args = parser.parse_args()

    if args.phase == 1:
        return run_phase1(args)
    if args.phase == 2:
        return run_phase2(args)
    raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
