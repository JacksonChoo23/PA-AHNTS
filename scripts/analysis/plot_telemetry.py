#!/usr/bin/env python3
"""Publication-grade CSV visualization suite for PA-AHNTS Phase 1 & Phase 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {png_path} and {pdf_path}")
    return png_path


def plot_phase1(p1_dir: Path) -> list[Path]:
    summary_path = p1_dir / "phase1_summary.json"
    frame_csv = p1_dir / "frame_metrics.csv"
    qual_csv = p1_dir / "quality_metrics.csv"
    if not summary_path.exists():
        print(f"[plot] Skipping Phase 1 (missing {summary_path})")
        return []

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    f_df = pd.read_csv(frame_csv)
    q_df = pd.read_csv(qual_csv)
    plots_dir = p1_dir / "plots"

    cfgs = list(summary["configs"].keys())
    short = ["B0: BCn-Full", "B1: SFS-BCn", "B2: NTC-Sample", "B3: NTC-Feedback"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    # 1. 4-Panel Overview Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: VRAM footprint
    ax = axes[0, 0]
    loc = [summary["configs"][c]["peak_local_vram_mb"]["mean"] / 1024.0 for c in cfgs]
    nloc = [summary["configs"][c]["peak_nonlocal_vram_mb"]["mean"] / 1024.0 for c in cfgs]
    x = np.arange(len(cfgs))
    ax.bar(x, loc, color=colors, label="Local VRAM (GiB)", edgecolor="black", linewidth=0.6)
    ax.bar(x, nloc, bottom=loc, color="#DD8452", label="Non-Local Shared (GiB)", edgecolor="black", linewidth=0.6)
    ax.axhline(7.0, color="red", linestyle="--", linewidth=1.1, label="DXGI Local Budget (~7.0 GiB)")
    for i, v in enumerate(loc):
        ax.text(i, v + nloc[i] + 0.15, f"{v:.2f} GiB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=12)
    ax.set_ylabel("Peak Memory (GiB)")
    ax.set_title("(a) Peak Process VRAM Footprint")
    ax.set_ylim(0, 9.6)
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Avg GPU/CPU & P99 CPU Frame Time
    ax = axes[0, 1]
    w = 0.27
    gpu_ms = [summary["configs"][c]["avg_gpu_frame_ms"]["mean"] for c in cfgs]
    cpu_ms = [summary["configs"][c]["avg_cpu_frame_ms"]["mean"] for c in cfgs]
    p99_ms = [summary["configs"][c]["p99_cpu_frame_ms"]["mean"] for c in cfgs]
    ax.bar(x - w, gpu_ms, width=w, color="#4C72B0", label="Avg GPU Frame (ms)")
    ax.bar(x, cpu_ms, width=w, color="#55A868", label="Avg CPU Frame (ms)")
    ax.bar(x + w, p99_ms, width=w, color="#C44E52", label="P99 CPU Frame (ms)")
    for i in range(len(cfgs)):
        ax.text(x[i] - w, gpu_ms[i] + 0.25, f"{gpu_ms[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w, p99_ms[i] + 0.25, f"{p99_ms[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=12)
    ax.set_ylabel("Frame Time (ms)")
    ax.set_title("(b) GPU / CPU Latency & P99 Tail Stutter")
    ax.set_ylim(0, max(p99_ms) * 1.18)
    ax.legend(fontsize=8)

    # Panel C: Power Breakdown (dGPU + CPU/SoC Package)
    ax = axes[1, 0]
    dgpu_w = [summary["configs"][c]["avg_dgpu_power_w"]["mean"] for c in cfgs]
    cpu_w = [summary["configs"][c]["avg_cpu_pkg_power_w"]["mean"] for c in cfgs]
    tot_err = [summary["configs"][c]["avg_total_power_w"]["ci95"] for c in cfgs]
    ax.bar(x, dgpu_w, color="#DD8452", label="dGPU Power (NVML, W)", edgecolor="black", linewidth=0.6)
    ax.bar(x, cpu_w, bottom=dgpu_w, yerr=tot_err, capsize=4, color="#64B5CD",
           label="CPU/SoC Pkg (RAPL, W)", edgecolor="black", linewidth=0.6)
    for i in range(len(cfgs)):
        tot = dgpu_w[i] + cpu_w[i]
        ax.text(i, tot + tot_err[i] + 2.0, f"{tot:.1f} W", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=12)
    ax.set_ylabel("Average Power (W)")
    ax.set_title("(c) System Package Power Breakdown (N=5, 95% CI)")
    ax.set_ylim(0, max(d + c for d, c in zip(dgpu_w, cpu_w)) * 1.22)
    ax.legend(fontsize=8)

    # Panel D: Visual Quality (PSNR & NVIDIA FLIP)
    ax = axes[1, 1]
    q_sub = q_df[q_df["config_name"] != "B0_BCn_Full"]
    q_names = ["B1_SFS_BCn", "B2_NTC_OnSample", "B3_NTC_OnFeedback"]
    q_short = ["B1: SFS-BCn", "B2: NTC-Sample", "B3: NTC-Feedback"]
    qx = np.arange(len(q_names))
    psnrs = [q_sub[q_sub["config_name"] == q]["psnr_db"].mean() for q in q_names]
    flips = [q_sub[q_sub["config_name"] == q]["flip_mean"].mean() for q in q_names]

    ax2 = ax.twinx()
    b1 = ax.bar(qx - 0.18, psnrs, width=0.34, color="#4C72B0", label="Mean PSNR (dB)")
    b2 = ax2.bar(qx + 0.18, flips, width=0.34, color="#C44E52", label="Mean FLIP (lower better)")
    for i, (p, fl) in enumerate(zip(psnrs, flips)):
        ax.text(qx[i] - 0.18, p + 1.2, f"{p:.1f}", ha="center", fontsize=8)
        ax2.text(qx[i] + 0.18, fl + 0.002, f"{fl:.4f}", ha="center", fontsize=8)
    ax.set_xticks(qx)
    ax.set_xticklabels(q_short)
    ax.set_ylabel("PSNR vs B0 (dB)", color="#4C72B0")
    ax2.set_ylabel("NVIDIA FLIP Error", color="#C44E52")
    ax.set_ylim(0, 82)
    ax2.set_ylim(0, 0.08)
    ax.set_title("(d) Visual Quality Across 5 Checkpoints")
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

    fig.suptitle("PA-AHNTS Phase 1: Baseline Reproduction on RTX 5060 8GB Laptop (1920x1080, N=5)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p_overview = save_both(fig, plots_dir, "phase1_baselines_overview")

    # 2. Flythrough Time-Series (run_id = 0)
    f0 = f_df[f_df["run_id"] == 0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for c, lbl, col in zip(cfgs, short, colors):
        sub = f0[f0["config_name"] == c].sort_values("frame_idx")
        axes[0].plot(sub["frame_idx"], sub["gpu_frame_ms"], label=lbl, color=col, linewidth=1.0, alpha=0.9)
        axes[1].plot(sub["frame_idx"], sub["cpu_frame_ms"].rolling(5, min_periods=1).mean(),
                     label=lbl, color=col, linewidth=1.0, alpha=0.9)

    b1_sub = f0[f0["config_name"] == "B1_SFS_BCn"].sort_values("frame_idx")
    b3_sub = f0[f0["config_name"] == "B3_NTC_OnFeedback"].sort_values("frame_idx")
    axes[2].plot(b1_sub["frame_idx"], b1_sub["pcie_upload_mb"], color="#55A868",
                 label="B1: PCIe BCn Upload (MB/frame)", linewidth=1.0)
    ax_t = axes[2].twinx()
    ax_t.plot(b3_sub["frame_idx"], b3_sub["tiles_decoded_dgpu"], color="#8172B3", alpha=0.75,
              label="B3: Async Compute Decoded Tiles/frame", linewidth=1.0)

    axes[0].set_ylabel("GPU Frame (ms)")
    axes[0].set_title("(a) Per-Frame GPU Execution Time (NTC-on-Sample 8.7x Shader Overhead)")
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")

    axes[1].set_ylabel("CPU Frame (ms, 5-f MA)")
    axes[1].set_title("(b) CPU Frame Time Across Deterministic 16-Bay Gallery Traversal")
    axes[1].set_ylim(0, 12)

    axes[2].set_ylabel("PCIe Upload (MB)", color="#55A868")
    ax_t.set_ylabel("Decoded Tiles", color="#8172B3")
    axes[2].set_xlabel("Measured Frame Index (0..999)")
    axes[2].set_title("(c) Streaming Activity: B1 PCIe BCn vs B3 Async Compute NTC Decode")

    fig.tight_layout()
    p_ts = save_both(fig, plots_dir, "phase1_flythrough_timeseries")
    return [p_overview, p_ts]


def plot_phase2(p1_dir: Path, p2_dir: Path) -> list[Path]:
    p2_sum_path = p2_dir / "phase2_summary.json"
    p1_sum_path = p1_dir / "phase1_summary.json"
    pcie_csv = p2_dir / "pcie_transfer_trace.csv"
    if not p2_sum_path.exists():
        print(f"[plot] Skipping Phase 2 (missing {p2_sum_path})")
        return []

    p2 = json.loads(p2_sum_path.read_text(encoding="utf-8"))
    p1 = json.loads(p1_sum_path.read_text(encoding="utf-8")) if p1_sum_path.exists() else None
    pcie_df = pd.read_csv(pcie_csv) if pcie_csv.exists() else pd.DataFrame()
    plots_dir = p2_dir / "plots"

    cfgs = list(p2["configs"].keys())
    labels = [c.replace("_L1_", "\nL1=").replace("Cap", "Cap ") for c in cfgs]
    x = np.arange(len(cfgs))

    # 1. VRAM Savings Curve across caps
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    loc_gb = [p2["configs"][c]["peak_local_vram_mb"]["mean"] / 1024.0 for c in cfgs]
    ram_gb = [p2["configs"][c]["peak_sys_ram_latent_mb"]["mean"] / 1024.0 for c in cfgs]

    ax.plot(x, loc_gb, marker="o", linewidth=2.4, markersize=8, color="#2B7BBA", label="PA-AHNTS RAM-Tiered Local VRAM (GiB)")
    ax.plot(x, ram_gb, marker="s", linewidth=2.0, linestyle="--", markersize=7, color="#E67E22", label="System RAM L2 Latent Pool (GiB)")

    if p1:
        b0_gb = p1["configs"]["B0_BCn_Full"]["peak_local_vram_mb"]["mean"] / 1024.0
        b3_gb = p1["configs"]["B3_NTC_OnFeedback"]["peak_local_vram_mb"]["mean"] / 1024.0
        ax.axhline(b0_gb, color="#C0392B", linestyle="-.", linewidth=1.6, label=f"B0: BCn-Full ({b0_gb:.2f} GiB)")
        ax.axhline(b3_gb, color="#8E44AD", linestyle=":", linewidth=1.8, label=f"B3: NTC-on-Feedback ({b3_gb:.2f} GiB)")
        ax.axhline(b3_gb - 2.0, color="#27AE60", linestyle="--", linewidth=1.3, label=f"Target >=2.0 GiB Savings vs B3 ({b3_gb - 2.0:.2f} GiB)")
        ax.fill_between(x, 0, b3_gb - 2.0, color="#27AE60", alpha=0.07)

    for i, (lv, rv) in enumerate(zip(loc_gb, ram_gb)):
        ax.text(i, lv + 0.18, f"{lv:.2f} GiB", ha="center", fontsize=9, fontweight="bold", color="#2B7BBA")
        ax.text(i, rv - 0.35, f"{rv:.2f} GiB", ha="center", fontsize=8.5, color="#D35400")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory Footprint (GiB)")
    ax.set_title("Phase 2 Gate 2: Local VRAM Savings & RAM L2 Residency Across VRAM Caps (N=5)")
    ax.set_ylim(0, 9.0)
    ax.legend(fontsize=8.5, loc="center right")
    p_vram = save_both(fig, plots_dir, "phase2_vram_savings_curve")

    # 2. PCIe Latency Histogram + CDF + Bandwidth vs Payload
    p_pcie = None
    if not pcie_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
        valid = pcie_df[pcie_df["copy_gpu_latency_us"] > 0.0]

        ax = axes[0]
        for c in cfgs:
            sub = valid[valid["config_name"] == c]["copy_gpu_latency_us"]
            if len(sub) > 5:
                sns.kdeplot(sub, ax=ax, label=c.split("_")[0] + " (" + c.split("_")[1] + ")", linewidth=1.6, clip=(0, None))
        ax.set_xlabel("Copy Queue Hardware GPU DMA Latency (us)")
        ax.set_ylabel("Density")
        ax.set_title("(a) Async Copy Queue PCIe Latency Distribution")
        ax.legend(fontsize=8)

        ax = axes[1]
        for c in cfgs:
            sub = valid[valid["config_name"] == c]
            if not sub.empty:
                ax.scatter(sub["upload_kb"] / 1024.0, sub["copy_gpu_latency_us"], s=14, alpha=0.45,
                           label=c.split("_")[0] + f" (mean {sub['bandwidth_gbps'].mean():.1f} GB/s)")
        ax.set_xlabel("Batch Upload Size (MiB)")
        ax.set_ylabel("Copy Queue GPU Latency (us)")
        ax.set_title("(b) PCIe Gen5 x8 DMA Latency vs Batch Size")
        ax.legend(fontsize=8)

        fig.suptitle("PA-AHNTS Phase 2: Hardware 1GHz Copy Queue Telemetry (L2 RAM -> L1 VRAM)",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout()
        p_pcie = save_both(fig, plots_dir, "phase2_pcie_latency_histogram")

    # 3. P99 Frame Time & Total PCIe Traffic Comparison (Phase 2 vs Phase 1 B1/B3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    comp_names = []
    comp_avg_cpu = []
    comp_p99_cpu = []
    comp_pcie_mb = []
    comp_colors = []

    if p1:
        for b, lbl, col in [("B1_SFS_BCn", "P1: B1\nSFS-BCn", "#55A868"),
                            ("B3_NTC_OnFeedback", "P1: B3\nNTC-FB", "#8172B3")]:
            comp_names.append(lbl)
            comp_avg_cpu.append(p1["configs"][b]["avg_cpu_frame_ms"]["mean"])
            comp_p99_cpu.append(p1["configs"][b]["p99_cpu_frame_ms"]["mean"])
            comp_pcie_mb.append(p1["configs"][b]["total_pcie_upload_mb"]["mean"])
            comp_colors.append(col)

    for idx, c in enumerate(cfgs):
        comp_names.append(f"P2: {c.split('_')[0]}\n({c.split('_')[-1]})")
        comp_avg_cpu.append(p2["configs"][c]["avg_cpu_frame_ms"]["mean"])
        comp_p99_cpu.append(p2["configs"][c]["p99_cpu_frame_ms"]["mean"])
        comp_pcie_mb.append(p2["configs"][c]["total_pcie_upload_mb"]["mean"])
        comp_colors.append("#2B7BBA")

    cx = np.arange(len(comp_names))
    w = 0.36
    axes[0].bar(cx - w / 2, comp_avg_cpu, width=w, color="#4C72B0", label="Avg Frame Time (ms)")
    axes[0].bar(cx + w / 2, comp_p99_cpu, width=w, color="#C44E52", label="P99 Frame Time (ms)")
    for i, (a, p) in enumerate(zip(comp_avg_cpu, comp_p99_cpu)):
        axes[0].text(cx[i] - w / 2, a + 0.25, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(cx[i] + w / 2, p + 0.25, f"{p:.2f}", ha="center", fontsize=8, fontweight="bold")
    axes[0].set_xticks(cx)
    axes[0].set_xticklabels(comp_names, fontsize=8.5)
    axes[0].set_ylabel("Frame Time (ms)")
    axes[0].set_title("(a) Avg & P99 Jitter: Non-Blocking Batched vs Phase 1 Sync")
    axes[0].set_ylim(0, max(comp_p99_cpu) * 1.22)
    axes[0].legend(fontsize=8.5)

    axes[1].bar(cx, comp_pcie_mb, color=comp_colors, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(comp_pcie_mb):
        axes[1].text(i, v + 8.0, f"{v:.1f} MB", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(cx)
    axes[1].set_xticklabels(comp_names, fontsize=8.5)
    axes[1].set_ylabel("Total PCIe Upload over 1000 Frames (MB)")
    axes[1].set_title("(b) Total PCIe Upload Traffic: Latent Streaming vs Raw BCn")
    axes[1].set_ylim(0, max(comp_pcie_mb) * 1.2)

    fig.tight_layout()
    p_p99 = save_both(fig, plots_dir, "phase2_p99_and_traffic_comparison")

    out = [p_vram, p_p99]
    if p_pcie is not None:
        out.append(p_pcie)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 1 & Phase 2 plots from raw CSVs")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Optional directory to copy PNGs")
    args = parser.parse_args()

    p1_dir = REPO_ROOT / "results" / "phase1"
    p2_dir = REPO_ROOT / "results" / "phase2"

    generated = plot_phase1(p1_dir) + plot_phase2(p1_dir, p2_dir)
    if args.artifact_dir and generated:
        ad = Path(args.artifact_dir)
        ad.mkdir(parents=True, exist_ok=True)
        for p in generated:
            shutil.copy2(p, ad / p.name)
            print(f"[plot] Copied {p.name} -> {ad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
