#!/usr/bin/env python3
"""PA-AHNTS Phase 2 Scientific Reinforcement Suite.

Executes controlled 2x2 factorial ablation + optimized Async SFS-BCn (N=5 runs x 1000 frames),
live-streamed visual quality validation (without artificial sync waits), resource-level memory
accounting across historical (B0..B3, R1..R5) and reinforced runs, and publication plots.
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
from typing import Any

import flip_evaluator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
BENCH_EXE = BUILD_DIR / "pa_ahnts_bench.exe"
PHASE1_CFG = REPO_ROOT / "configs" / "experiments" / "phase1_baselines.json"
PHASE2_CFG = REPO_ROOT / "configs" / "experiments" / "phase2_ram_tier.json"
OUT_DIR = REPO_ROOT / "results" / "phase2_reinforced"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.08)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

CONFIG_SPECS: list[dict[str, Any]] = [
    {"name": "C0_VRAM_Sync",              "l1_mib": 0,   "l2_mib": 0,   "group": "2x2_Sync_VRAM"},
    {"name": "C1_RAM_R1_Sync",            "l1_mib": 256, "l2_mib": 512, "group": "2x2_Sync_RAM"},
    {"name": "C1_RAM_R2_Sync",            "l1_mib": 512, "l2_mib": 512, "group": "2x2_Sync_RAM"},
    {"name": "C2_VRAM_Async",             "l1_mib": 0,   "l2_mib": 0,   "group": "2x2_Async_VRAM"},
    {"name": "C3_RAM_R0_128MB_Async",     "l1_mib": 128, "l2_mib": 512, "group": "Eviction_Stress"},
    {"name": "C3_RAM_R1_Async",           "l1_mib": 256, "l2_mib": 512, "group": "2x2_Async_RAM"},
    {"name": "C3_RAM_R2_Async",           "l1_mib": 512, "l2_mib": 512, "group": "2x2_Async_RAM"},
    {"name": "B1_SFS_BCn_Async_RamCache", "l1_mib": 0,   "l2_mib": 768, "group": "Async_SFS_BCn"},
]


def run_cmd(cmd: list[str]) -> None:
    print(f"[p2r] $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def mean_std_ci95(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = statistics.mean(vals) if n > 0 else 0.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    t_table = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    ci95 = (t_table.get(n, 1.96) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": round(m, 4), "std": round(sd, 4), "ci95": round(ci95, 4)}


def compute_psnr_db(ref_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    mse = float(np.mean((ref_rgb - test_rgb) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def evaluate_streamed_quality(cfg_names: list[str], out_dir: Path, keep_captures: bool = False) -> list[dict[str, Any]]:
    cam_cfg = json.loads((REPO_ROOT / "configs" / "camera_paths" / "hallway_flythrough_v1.json").read_text(encoding="utf-8"))
    checkpoints = [ck["id"] for ck in cam_cfg["quality_checkpoints"]]
    ref_dir = REPO_ROOT / "results" / "phase1" / "captures"
    cap_dir = out_dir / "captures"
    rows: list[dict[str, Any]] = []

    for c in cfg_names:
        for ck_id in checkpoints:
            ref_p = ref_dir / f"B0_BCn_Full_{ck_id}.png"
            test_p = cap_dir / f"{c}_{ck_id}.png"
            ref_arr = np.asarray(Image.open(ref_p).convert("RGB"), dtype=np.float32) / 255.0
            test_arr = np.asarray(Image.open(test_p).convert("RGB"), dtype=np.float32) / 255.0
            psnr_db = compute_psnr_db(ref_arr, test_arr)
            flip_map, flip_mean, _ = flip_evaluator.evaluate(
                ref_arr, test_arr, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
            )
            flip_p99 = float(np.percentile(flip_map, 99.0))
            rows.append(
                {
                    "config_name": c,
                    "camera_checkpoint_id": ck_id,
                    "capture_mode": "streamed_no_forced_sync",
                    "psnr_db": round(psnr_db, 4),
                    "flip_mean": round(float(flip_mean), 6),
                    "flip_p99": round(flip_p99, 6),
                    "reference_config": "B0_BCn_Full",
                }
            )

    q_csv = out_dir / "quality_metrics.csv"
    with q_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[p2r] Wrote {q_csv} ({len(rows)} rows)")

    for c in cfg_names:
        c_rows = [r for r in rows if r["config_name"] == c]
        m_psnr = statistics.mean(r["psnr_db"] for r in c_rows)
        m_flip = statistics.mean(r["flip_mean"] for r in c_rows)
        print(f"[p2r][quality] {c:<26} mean_psnr={m_psnr:.4f} dB | mean_flip={m_flip:.6f}")

    if "C2_VRAM_Async" in cfg_names and "C3_RAM_R2_Async" in cfg_names:
        print("[p2r][parity] C2_VRAM_Async vs C3_RAM_R2_Async per-checkpoint:")
        for ck_id in checkpoints:
            r_c2 = next(r for r in rows if r["config_name"] == "C2_VRAM_Async" and r["camera_checkpoint_id"] == ck_id)
            r_c3 = next(r for r in rows if r["config_name"] == "C3_RAM_R2_Async" and r["camera_checkpoint_id"] == ck_id)
            print(
                f"  {ck_id}: C2={r_c2['psnr_db']:.4f} dB (FLIP={r_c2['flip_mean']:.6f}) | "
                f"C3={r_c3['psnr_db']:.4f} dB (FLIP={r_c3['flip_mean']:.6f}) | "
                f"diff={r_c3['psnr_db'] - r_c2['psnr_db']:+.4f} dB"
            )

    if not keep_captures and cap_dir.exists():
        keep_set = {"C2_VRAM_Async_ckpt_00.png", "C3_RAM_R2_Async_ckpt_00.png"}
        for p in cap_dir.glob("*.png"):
            if p.name in keep_set:
                with Image.open(p) as im:
                    im.save(p, format="PNG", optimize=True, compress_level=9)
            else:
                p.unlink(missing_ok=True)
        print(f"[p2r] Cleaned bulky capture PNGs (kept compressed sample pair for visual audit)")

    return rows


def build_historical_memory_comparison(out_dir: Path, rmb_df: pd.DataFrame) -> pd.DataFrame:
    """Unify B0..B3 (Phase 1), R1..R5 (Phase 2), and C0..B1_RamCache (Phase 2 Reinforced) in MiB/GiB."""
    rows: list[dict[str, Any]] = []
    p1_sum = REPO_ROOT / "results" / "phase1" / "phase1_summary.json"
    p2_sum = REPO_ROOT / "results" / "phase2" / "phase2_summary.json"

    if p1_sum.exists():
        p1 = json.loads(p1_sum.read_text(encoding="utf-8"))
        for b, d in p1["configs"].items():
            loc = d["peak_local_vram_mb"]["mean"]
            nloc = d["peak_nonlocal_vram_mb"]["mean"]
            rows.append(
                {
                    "cohort": "Phase1_Baseline",
                    "config_name": b,
                    "dxgi_peak_local_vram_mib": round(loc, 2),
                    "dxgi_peak_local_vram_gib": round(loc / 1024.0, 4),
                    "dxgi_peak_nonlocal_mib": round(nloc, 2),
                    "ram_working_set_populated_mib": 0.0,
                    "bcn_cache_mib": 1536.0 if b in ("B1_SFS_BCn", "B3_NTC_OnFeedback") else (8192.0 if b == "B0_BCn_Full" else 0.0),
                    "latent_vram_mib": 3072.0 if "NTC" in b else 0.0,
                    "driver_plus_runtime_mib": round(loc - (1536.0 if b in ("B1_SFS_BCn", "B3_NTC_OnFeedback") else 0.0) - (3072.0 if "NTC" in b else (8192.0 if b == "B0_BCn_Full" else 0.0)), 2),
                }
            )
    if p2_sum.exists():
        p2 = json.loads(p2_sum.read_text(encoding="utf-8"))
        for cfg in json.loads(PHASE2_CFG.read_text(encoding="utf-8"))["configs"]:
            c = cfg["name"]
            if c not in p2["configs"]:
                continue
            d = p2["configs"][c]
            loc = d["peak_local_vram_mb"]["mean"]
            nloc = d["peak_nonlocal_vram_mb"]["mean"]
            l1 = float(cfg["l1_latent_pool_mib"])
            ram_pop = float(d["peak_sys_ram_latent_mb"]["mean"])
            rows.append(
                {
                    "cohort": "Phase2_Original",
                    "config_name": c,
                    "dxgi_peak_local_vram_mib": round(loc, 2),
                    "dxgi_peak_local_vram_gib": round(loc / 1024.0, 4),
                    "dxgi_peak_nonlocal_mib": round(nloc, 2),
                    "ram_working_set_populated_mib": round(ram_pop, 2),
                    "bcn_cache_mib": 1536.0,
                    "latent_vram_mib": l1,
                    "driver_plus_runtime_mib": round(loc - 1536.0 - l1, 2),
                }
            )

    grp = rmb_df.groupby("config_name", as_index=False).mean(numeric_only=True)
    for _, r in grp.iterrows():
        loc = float(r["dxgi_peak_local_vram_mib"])
        rows.append(
            {
                "cohort": "Phase2_Reinforced",
                "config_name": r["config_name"],
                "dxgi_peak_local_vram_mib": round(loc, 2),
                "dxgi_peak_local_vram_gib": round(loc / 1024.0, 4),
                "dxgi_peak_nonlocal_mib": round(float(r["dxgi_peak_nonlocal_mib"]), 2),
                "ram_working_set_populated_mib": round(float(r["ram_working_set_populated_mib"]), 2),
                "bcn_cache_mib": round(float(r["bcn_cache_heap_mib"]), 2),
                "latent_vram_mib": round(float(r["l1_latent_heap_mib"] + r["committed_latent_textures_mib"]), 2),
                "driver_plus_runtime_mib": round(loc - float(r["bcn_cache_heap_mib"]) - float(r["l1_latent_heap_mib"] + r["committed_latent_textures_mib"]), 2),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "historical_vs_reinforced_memory_table.csv", index=False)
    return df


def summarize_and_plot(out_dir: Path, cfg_names: list[str], runs: int) -> dict[str, Any]:
    ext_df = pd.read_csv(out_dir / "frame_extended_metrics.csv")
    mem_df = pd.read_csv(out_dir / "memory_power.csv")
    pcie_df = pd.read_csv(out_dir / "pcie_transfer_trace.csv")
    rmb_df = pd.read_csv(out_dir / "resource_memory_breakdown.csv")
    qual_df = pd.read_csv(out_dir / "quality_metrics.csv")

    build_historical_memory_comparison(out_dir, rmb_df)

    summary_cfgs: dict[str, Any] = {}
    for c in cfg_names:
        sub_e = ext_df[ext_df["config_name"] == c]
        sub_m = mem_df[mem_df["config_name"] == c]
        sub_p = pcie_df[pcie_df["config_name"] == c]
        sub_r = rmb_df[rmb_df["config_name"] == c]
        sub_q = qual_df[qual_df["config_name"] == c]

        run_ids = sorted(sub_e["run_id"].unique().tolist())
        pres_fps_r, low1_fps_r, p99_disp_r, cpu_work_r, gpu_ms_r = [], [], [], [], []
        pkg_w_r, dgpu_w_r, cpu_w_r, jpf_r, loc_mib_r = [], [], [], [], []
        bcn_hit_r, l1_hit_r, l2_hit_r, miss_r, popin_r, l1_ev_r = [], [], [], [], [], []

        for rid in run_ids:
            fe = sub_e[sub_e["run_id"] == rid]
            disp = fe["display_frame_ms"].to_numpy()
            mean_disp = float(np.mean(disp))
            worst_1pct = np.sort(disp)[max(0, int(0.99 * len(disp))):]
            low1_ms = float(np.mean(worst_1pct))
            p99_ms = float(np.percentile(disp, 99.0))
            pres_fps_r.append(1000.0 / max(mean_disp, 1e-6))
            low1_fps_r.append(1000.0 / max(low1_ms, 1e-6))
            p99_disp_r.append(p99_ms)
            cpu_work_r.append(float(fe["cpu_work_ms"].mean()))
            vg = fe[fe["gpu_frame_ms"] > 0.0]["gpu_frame_ms"]
            gpu_ms_r.append(float(vg.mean()) if not vg.empty else 0.0)

            bcn_hit_r.append(float(fe["bcn_hit_rate"].mean()) * 100.0)
            l1_hit_r.append(float(fe["l1_hit_rate"].mean()) * 100.0)
            l2_hit_r.append(float(fe["l2_hit_rate"].mean()) * 100.0)
            miss_r.append(float(fe["deadline_miss_rate"].mean()) * 100.0)
            popin_r.append(float(fe["popin_transitions"].sum()))
            l1_ev_r.append(float(fe["l1_evictions"].sum()))

            fm = sub_m[sub_m["run_id"] == rid]
            if not fm.empty:
                dgw = float(fm["dgpu_power_w"].mean())
                cpw = float(fm["cpu_pkg_power_w"].mean())
                totw = float(fm["total_est_power_w"].mean())
                dgpu_w_r.append(dgw)
                cpu_w_r.append(cpw)
                pkg_w_r.append(totw)
                jpf_r.append(totw * (mean_disp / 1000.0))

            fr = sub_r[sub_r["run_id"] == rid]
            if not fr.empty:
                loc_mib_r.append(float(fr["dxgi_peak_local_vram_mib"].iloc[0]))

        summary_cfgs[c] = {
            "present_fps": mean_std_ci95(pres_fps_r),
            "low_1pct_fps": mean_std_ci95(low1_fps_r),
            "p99_display_ms": mean_std_ci95(p99_disp_r),
            "cpu_work_ms": mean_std_ci95(cpu_work_r),
            "gpu_frame_ms": mean_std_ci95(gpu_ms_r),
            "peak_local_vram_mib": mean_std_ci95(loc_mib_r),
            "ram_arena_committed_mib": round(float(sub_r["ram_arena_committed_mib"].mean()), 2),
            "ram_populated_mib": round(float(sub_r["ram_working_set_populated_mib"].mean()), 2),
            "l1_populated_peak_mib": round(float(sub_r["vram_l1_populated_peak_mib"].mean()), 2),
            "pkg_sum_power_w": mean_std_ci95(pkg_w_r),
            "dgpu_power_w": mean_std_ci95(dgpu_w_r),
            "cpu_pkg_power_w": mean_std_ci95(cpu_w_r),
            "joules_per_frame": mean_std_ci95(jpf_r),
            "bcn_hit_rate_pct": mean_std_ci95(bcn_hit_r),
            "l1_hit_rate_pct": mean_std_ci95(l1_hit_r),
            "l2_hit_rate_pct": mean_std_ci95(l2_hit_r),
            "deadline_miss_rate_pct": mean_std_ci95(miss_r),
            "total_popin_transitions": mean_std_ci95(popin_r),
            "total_l1_evictions": mean_std_ci95(l1_ev_r),
            "total_pcie_upload_mib": round(float(sub_p["upload_kb"].sum() / (max(1, len(run_ids)) * 1024.0)), 2) if not sub_p.empty else 0.0,
            "pcie_p50_us": round(float(np.percentile(sub_p["copy_gpu_latency_us"], 50.0)), 2) if not sub_p.empty else 0.0,
            "pcie_p99_us": round(float(np.percentile(sub_p["copy_gpu_latency_us"], 99.0)), 2) if not sub_p.empty else 0.0,
            "pcie_mean_bw_gbps": round(float(sub_p["bandwidth_gbps"].mean()), 3) if not sub_p.empty else 0.0,
            "mean_psnr_db": round(float(sub_q["psnr_db"].mean()), 4),
            "mean_flip": round(float(sub_q["flip_mean"].mean()), 6),
            "max_flip_p99": round(float(sub_q["flip_p99"].max()), 6),
        }

    # Isolate 2x2 factorial effects: Tiering (RAM R1/R2 vs VRAM) and Scheduling (Async vs Sync)
    def val(cfg: str, metric: str) -> float:
        return float(summary_cfgs[cfg][metric]["mean"]) if cfg in summary_cfgs else 0.0

    factorial = {}
    if all(k in summary_cfgs for k in ("C0_VRAM_Sync", "C1_RAM_R2_Sync", "C2_VRAM_Async", "C3_RAM_R2_Async")):
        for metric in ("present_fps", "low_1pct_fps", "p99_display_ms", "peak_local_vram_mib", "joules_per_frame"):
            v00 = val("C0_VRAM_Sync", metric)
            v10 = val("C1_RAM_R2_Sync", metric)
            v01 = val("C2_VRAM_Async", metric)
            v11 = val("C3_RAM_R2_Async", metric)
            factorial[metric] = {
                "C0_VRAM_Sync": v00,
                "C1_RAM_R2_Sync": v10,
                "C2_VRAM_Async": v01,
                "C3_RAM_R2_Async": v11,
                "ram_tier_effect_under_sync": round(v10 - v00, 4),
                "ram_tier_effect_under_async": round(v11 - v01, 4),
                "async_effect_under_vram": round(v01 - v00, 4),
                "async_effect_under_ram": round(v11 - v10, 4),
            }

    copy_freq = int(rmb_df["copy_queue_freq_hz"].iloc[0]) if not rmb_df.empty else 0
    summary = {
        "runs_per_config": runs,
        "copy_queue_timestamp_freq_hz": copy_freq,
        "configs": summary_cfgs,
        "factorial_2x2_R2_512MiB": factorial,
    }
    (out_dir / "phase2_reinforced_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Generate 4 Publication Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig: plt.Figure, stem: str) -> None:
        fig.savefig(plots_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Plot 1: Memory Accounting Waterfall
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    rmb_m = rmb_df.groupby("config_name", as_index=False).mean(numeric_only=True)
    rmb_m["order"] = rmb_m["config_name"].map({n: i for i, n in enumerate(cfg_names)})
    rmb_m = rmb_m.sort_values("order")
    x = np.arange(len(rmb_m))
    bcn_h = rmb_m["bcn_cache_heap_mib"].to_numpy() / 1024.0
    lat_c = rmb_m["committed_latent_textures_mib"].to_numpy() / 1024.0
    l1_h = rmb_m["l1_latent_heap_mib"].to_numpy() / 1024.0
    tails = (rmb_m["bcn_tail_heap_mib"] + rmb_m["latent_tail_heap_mib"]).to_numpy() / 1024.0
    rest = (rmb_m["dxgi_peak_local_vram_mib"].to_numpy() / 1024.0) - (bcn_h + lat_c + l1_h + tails)
    ax.bar(x, bcn_h, label="BCn Tile Cache Heap (1.50 GiB)", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x, lat_c, bottom=bcn_h, label="Committed G0/G1 Latents (3.00 GiB)", color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.bar(x, l1_h, bottom=bcn_h + lat_c, label="VRAM L1 Latent Pool (0.125-0.50 GiB)", color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(x, tails, bottom=bcn_h + lat_c + l1_h, label="Mip-Tail Heaps", color="#8172B3", edgecolor="black", linewidth=0.5)
    ax.bar(x, rest, bottom=bcn_h + lat_c + l1_h + tails, label="Staging + RT/FB + Driver Overhead", color="#DD8452", edgecolor="black", linewidth=0.5)
    tot_g = rmb_m["dxgi_peak_local_vram_mib"].to_numpy() / 1024.0
    for i, v in enumerate(tot_g):
        ax.text(i, v + 0.08, f"{v:.2f} GiB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in rmb_m["config_name"]], fontsize=8)
    ax.set_ylabel("Peak DXGI Local VRAM (GiB)")
    ax.set_title("Phase 2 Reinforced: Resource-Level VRAM Accounting (Zero Double-Counting)")
    ax.set_ylim(0, max(tot_g) * 1.15)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "p2r_memory_accounting_waterfall")

    # Plot 2: Controlled 2x2 Ablation (Sync vs Async x VRAM vs RAM)
    c_2x2 = [k for k in ["C0_VRAM_Sync", "C1_RAM_R1_Sync", "C1_RAM_R2_Sync", "C2_VRAM_Async", "C3_RAM_R1_Async", "C3_RAM_R2_Async"] if k in summary_cfgs]
    if c_2x2:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
        cx = np.arange(len(c_2x2))
        cols = ["#8C8C8C" if "Sync" in n else "#2B7BBA" for n in c_2x2]
        fps_v = [summary_cfgs[n]["present_fps"]["mean"] for n in c_2x2]
        low1_v = [summary_cfgs[n]["low_1pct_fps"]["mean"] for n in c_2x2]
        p99_v = [summary_cfgs[n]["p99_display_ms"]["mean"] for n in c_2x2]
        vram_v = [summary_cfgs[n]["peak_local_vram_mib"]["mean"] / 1024.0 for n in c_2x2]

        axes[0].bar(cx - 0.18, fps_v, width=0.35, color="#4C72B0", label="Present FPS")
        axes[0].bar(cx + 0.18, low1_v, width=0.35, color="#55A868", label="1% Low FPS")
        axes[0].set_xticks(cx)
        axes[0].set_xticklabels([n.replace("_", "\n") for n in c_2x2], fontsize=7.5)
        axes[0].set_ylabel("Frames / sec")
        axes[0].set_title("(a) Present FPS & 1% Low FPS")
        axes[0].legend(fontsize=8)

        axes[1].bar(cx, p99_v, color=cols, edgecolor="black", linewidth=0.5)
        for i, v in enumerate(p99_v):
            axes[1].text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].set_xticks(cx)
        axes[1].set_xticklabels([n.replace("_", "\n") for n in c_2x2], fontsize=7.5)
        axes[1].set_ylabel("P99 Displayed Frame Time (ms)")
        axes[1].set_title("(b) P99 Display Latency (Sync vs Async Isolation)")

        axes[2].bar(cx, vram_v, color=cols, edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vram_v):
            axes[2].text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        axes[2].set_xticks(cx)
        axes[2].set_xticklabels([n.replace("_", "\n") for n in c_2x2], fontsize=7.5)
        axes[2].set_ylabel("Peak Local VRAM (GiB)")
        axes[2].set_title("(c) Local VRAM Savings (VRAM vs RAM Tier)")
        fig.suptitle("Phase 2 Reinforced: Controlled 2x2 Factorial Ablation (N=5)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, "p2r_2x2_factorial_ablation")

    # Plot 3: Visual Quality (Streamed PSNR & FLIP) & Deadline Miss / Pop-In
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    psnrs = [summary_cfgs[n]["mean_psnr_db"] for n in cfg_names]
    flips = [summary_cfgs[n]["mean_flip"] for n in cfg_names]
    miss_rates = [summary_cfgs[n]["deadline_miss_rate_pct"]["mean"] for n in cfg_names]
    popins = [summary_cfgs[n]["total_popin_transitions"]["mean"] for n in cfg_names]
    qx = np.arange(len(cfg_names))

    ax2 = axes[0].twinx()
    axes[0].bar(qx - 0.18, psnrs, width=0.34, color="#4C72B0", label="Streamed PSNR (dB)")
    ax2.bar(qx + 0.18, flips, width=0.34, color="#DD8452", label="Streamed FLIP Mean")
    axes[0].set_xticks(qx)
    axes[0].set_xticklabels([n.replace("_", "\n") for n in cfg_names], fontsize=7.2)
    axes[0].set_ylabel("PSNR vs B0 (dB, higher=better)")
    ax2.set_ylabel("NVIDIA FLIP (lower=better)", color="#DD8452")
    axes[0].set_title("(a) Live-Streamed Checkpoint Quality (Fixed G1 + Halo)")

    ax2b = axes[1].twinx()
    axes[1].bar(qx - 0.18, miss_rates, width=0.34, color="#C44E52", label="Deadline Miss Rate (%)")
    ax2b.bar(qx + 0.18, popins, width=0.34, color="#8172B3", label="Pop-In Transitions (1000f)")
    axes[1].set_xticks(qx)
    axes[1].set_xticklabels([n.replace("_", "\n") for n in cfg_names], fontsize=7.2)
    axes[1].set_ylabel("Tile Deadline Miss Rate (%)", color="#C44E52")
    ax2b.set_ylabel("Visible Pop-In Transitions", color="#8172B3")
    axes[1].set_title("(b) Tile Deadline Miss Rate & Pop-In Frequency")
    fig.tight_layout()
    save_fig(fig, "p2r_quality_and_popin_validation")

    # Plot 4: Package Power & Joules/Frame
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    p_tot = [summary_cfgs[n]["pkg_sum_power_w"]["mean"] for n in cfg_names]
    jpf = [summary_cfgs[n]["joules_per_frame"]["mean"] for n in cfg_names]
    axes[0].bar(qx, p_tot, color="#64B5CD", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(p_tot):
        axes[0].text(i, v + 0.8, f"{v:.1f}W", ha="center", fontsize=8)
    axes[0].set_xticks(qx)
    axes[0].set_xticklabels([n.replace("_", "\n") for n in cfg_names], fontsize=7.2)
    axes[0].set_ylabel("Package-Sum Power (dGPU + CPU PKG, W)")
    axes[0].set_title("(a) Average Package Power")

    axes[1].bar(qx, jpf, color="#55A868", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(jpf):
        axes[1].text(i, v + 0.002, f"{v:.3f}J", ha="center", fontsize=8)
    axes[1].set_xticks(qx)
    axes[1].set_xticklabels([n.replace("_", "\n") for n in cfg_names], fontsize=7.2)
    axes[1].set_ylabel("Package Energy per Displayed Frame (J)")
    axes[1].set_title("(b) Energy per Frame (Package-Sum)")
    fig.tight_layout()
    save_fig(fig, "p2r_fps_1pctlow_energy")

    print("\n[p2r] Phase 2 Reinforced Summary:")
    print(f"{'Config':<26} {'FPS':>7} {'1%Low':>7} {'P99ms':>7} {'VRAM_GiB':>8} {'RAM_Pop':>8} {'PSNR':>6} {'FLIP':>7} {'PkgW':>6} {'J/frm':>7}")
    for n in cfg_names:
        c = summary_cfgs[n]
        print(
            f"{n:<26} {c['present_fps']['mean']:7.1f} {c['low_1pct_fps']['mean']:7.1f} "
            f"{c['p99_display_ms']['mean']:7.2f} {c['peak_local_vram_mib']['mean']/1024.0:8.3f} "
            f"{c['ram_populated_mib']:8.1f} {c['mean_psnr_db']:6.2f} {c['mean_flip']:7.4f} "
            f"{c['pkg_sum_power_w']['mean']:6.2f} {c['joules_per_frame']['mean']:7.4f}"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="PA-AHNTS Phase 2 Reinforced Suite")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-exec", action="store_true")
    parser.add_argument("--keep-captures", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BENCH_EXE.exists():
        run_cmd(["cmake", "-S", ".", "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
        run_cmd(["cmake", "--build", str(BUILD_DIR), "--target", "pa_ahnts_bench"])

    exp_cfg = json.loads(PHASE2_CFG.read_text(encoding="utf-8"))
    runs = 1 if args.smoke else int(args.runs)
    warmup = 15 if args.smoke else int(exp_cfg["warmup_frames"])
    frames = 40 if args.smoke else int(exp_cfg["measured_frames"])
    specs = [CONFIG_SPECS[3], CONFIG_SPECS[6]] if args.smoke else CONFIG_SPECS

    if not args.skip_exec:
        for csv_name in ("frame_metrics.csv", "memory_power.csv", "pcie_transfer_trace.csv",
                         "frame_extended_metrics.csv", "resource_memory_breakdown.csv"):
            (OUT_DIR / csv_name).unlink(missing_ok=True)

        for sp in specs:
            for r in range(runs):
                run_cmd([
                    str(BENCH_EXE), "--phase", "21", "--config", str(PHASE2_CFG),
                    "--config-name", sp["name"], "--run-id", str(r),
                    "--l1-mib", str(sp["l1_mib"]), "--ram-l2-mib", str(sp["l2_mib"]),
                    "--warmup", str(warmup), "--frames", str(frames),
                    "--out-dir", str(OUT_DIR), "--shader-dir", str(BUILD_DIR / "shaders"),
                ])

            run_cmd([
                str(BENCH_EXE), "--phase", "21", "--config", str(PHASE2_CFG),
                "--config-name", sp["name"], "--run-id", "0",
                "--l1-mib", str(sp["l1_mib"]), "--ram-l2-mib", str(sp["l2_mib"]),
                "--out-dir", str(OUT_DIR), "--shader-dir", str(BUILD_DIR / "shaders"),
                "--capture-checkpoints",
            ])

    names = [sp["name"] for sp in specs]
    evaluate_streamed_quality(names, OUT_DIR, keep_captures=args.keep_captures)
    summarize_and_plot(OUT_DIR, names, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
