# PA-AHNTS

**Power-Aware Adaptive Heterogeneous Neural Texture Streaming for VRAM-Constrained Gaming Laptops**

PA-AHNTS is a reproducible DirectX 12 prototype and automated benchmark suite designed for **8 GB VRAM gaming laptops** under heavy 4K/8K PBR texture workloads. It evaluates neural texture compression (NTC), Sampler Feedback Streaming (SFS), RAM-tiered latent pools, heterogeneous decode micro-benchmarks (dGPU / iGPU / NPU / CPU), and power-aware adaptive scheduling using live hardware telemetry (DXGI, D3D12 timestamps, NVML, and Windows PDH RAPL).

## Research Targets

Compared against optimized SFS and NTC-on-Feedback baselines at matched visual quality and residency policy:

| Metric | Target | Measurement Basis |
| :--- | :--- | :--- |
| **VRAM Savings** | >= 2 GB | Peak process Local VRAM vs. Full BCn baseline |
| **Avg FPS Loss** | <= 5% | Deterministic flythrough vs. baseline |
| **P99 Frame Time** | <= +5% | 99th-percentile frame time (ms) |
| **Extra System Power** | ~0 W | Total package delta (CPU + iGPU + NPU + dGPU) |
| **Visual Quality** | ~BCn baseline | Offline PSNR (dB) & NVIDIA FLIP (mean / P99) |

## Repository Layout

```text
PA-AHNTS/
├── configs/
│   ├── hardware_profile.json     # Auto-generated hardware & capability profile (Phase 0)
│   ├── camera_paths/             # Deterministic camera flythrough paths
│   └── experiments/              # Phase & ablation JSON configurations
├── third_party/                  # External SDKs (Agility SDK, DXC, etc.; gitignored)
├── tools/
│   ├── env_probe/                # Phase 0 hardware/driver/NPU/DXGI/DirectML probe
│   └── asset_gen/                # 4K/8K PBR stress asset & NTC latent generator
├── src/
│   ├── core/                     # DX12 device, context, scene geometry, JSON helpers
│   ├── telemetry/                # Unified logger (VRAM, FrameTime, NVML/RAPL Power, PCIe)
│   ├── baselines/                # Phase 1: B0 BCn-Full, B1 SFS-BCn, B2 NTC-on-Sample, B3 NTC-on-Feedback
│   ├── ram_tier/                 # Phase 2: RAM-tiered latent pool, LRU, async Copy Queue
│   ├── hetero_bench/             # Phase 3: CPU / NPU / iGPU / dGPU decode micro-benchmarks
│   └── scheduler/                # Phase 4: Power-aware adaptive heterogeneous scheduler
├── scripts/
│   ├── setup_phase1_deps.py      # Fetch Agility SDK & DXC into third_party/
│   ├── run_phase.py              # Automated benchmark runner & quality evaluator
│   └── analysis/                 # Telemetry plotting & statistical analysis
├── results/                      # Raw CSVs, captures, plots (gitignored)
├── CMakeLists.txt                # Windows x64 build configuration
└── EXPERIMENT_LOG.md             # Chronological experiment & engineering log
```

## Prerequisites

- **OS:** Windows 11 x64
- **Toolchain:** CMake >= 3.24, Ninja (or Visual Studio 2022), C++17 compiler
- **Python:** Python 3.12+ with `numpy`, `Pillow`, `flip-evaluator`, and PyTorch (for asset/latent generation)
- **SDKs:** DirectX 12 Agility SDK and DXC (installed to `third_party/` via `scripts/setup_phase1_deps.py`)

## Quick Start

1. **Run the Phase 0 hardware & environment probe:**
   ```powershell
   python tools/env_probe/probe_hardware.py
   ```
   This generates `configs/hardware_profile.json` from live DXGI, D3D12, DirectML, NVML, and Windows PDH queries.

2. **Install Phase 1 SDK dependencies (`third_party/agility_sdk`, `third_party/dxc`):**
   ```powershell
   python scripts/setup_phase1_deps.py
   ```

3. **Generate PBR stress assets & NTC latents:**
   ```powershell
   python tools/asset_gen/generate_pbr_assets.py
   ```

4. **Build `pa_ahnts_bench`:**
   ```powershell
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target pa_ahnts_bench
   ```

5. **Run benchmarks:**
   ```powershell
   python scripts/run_phase.py
   ```

## Current Status

See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for full hardware probe details and phase-by-phase results:

- **Phase 0 (Gate 0 — Hardware & Environment Probe):** Passed on `Intel Core Ultra 7 255HX` + `NVIDIA GeForce RTX 5060 Laptop GPU` (8 GB VRAM) + `Intel AI Boost` NPU.
- **Phase 1 (Gate 1 — Baseline Reproduction):** Passed (`N = 5` runs, 120 warm-up + 1,000 measured frames per baseline across `B0_BCn_Full`, `B1_SFS_BCn`, `B2_NTC_OnSample`, and `B3_NTC_OnFeedback`).
- **Phases 2–6:** RAM-tiered latent pool (`src/ram_tier/`), heterogeneous decode micro-benchmarks, power-aware adaptive scheduling, cross-platform extension, and final ablation artifacts.
