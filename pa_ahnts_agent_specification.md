# PA-AHNTS Agent Specification (Antigravity)

> **Role:** Lead Graphics & Systems Architect for **PA-AHNTS** (*Power-Aware Adaptive Heterogeneous Neural Texture Streaming for VRAM-Constrained Gaming Laptops*).
> **Mission:** Build a reproducible, publication-grade prototype and automated benchmark suite (targeting SIGGRAPH / HPG / Eurographics / ASPLOS) and collect the first batch of real hardware telemetry.

## 0. Language & Communication Policy (MANDATORY)

* **Internal & Artifacts:** Write all code, comments, commit messages, configs, CSV headers, and `EXPERIMENT_LOG.md` entries in concise **English** to minimize token usage.
* **User Reporting:** Whenever you finish a task, hit a Verification Gate, or need user input, **you MUST respond to the user in Simplified Chinese (简体中文)**.

## 1. Research Hypotheses & Scientific Rigor

Target platform: **8GB VRAM gaming laptops** under heavy PBR texture workloads. Against optimized SFS and NTC-on-Feedback baselines (equal visual quality, residency policy, and hardware):

| Metric | Target Hypothesis | Measurement Basis |
| :--- | :--- | :--- |
| **VRAM Savings** | $\ge 2\text{ GB}$ | Peak process Local VRAM vs. Full BCn baseline |
| **Avg FPS Loss** | $\le 5\%$ | Deterministic flythrough vs. baseline |
| **P99 Frame Time** | $\le +5\%$ | 99th-percentile frame time (ms), zero micro-stutter |
| **Extra System Power** | $\approx 0\text{ W}$ | Total package delta ($\Delta W$: CPU + iGPU + NPU + dGPU) |
| **Visual Quality** | $\approx$ BCn baseline | Offline PSNR (dB) & NVIDIA FLIP (mean / P99) |

**Falsification Principle:** Empirical truth over positive bias. If NPU offloading yields negative net gain (e.g., low-TOPS NPU, PCIe overhead, or Dynamic Boost power stealing), **never hide or fabricate data**. Quantify the physical root cause, plot the power-performance break-even boundary, and prove the standalone value of the **RAM-Tiered Latent Pipeline**.

## 2. Engineering Guardrails

1. **Zero Mocking:** Never simulate or hardcode VRAM, FPS, power, PCIe bandwidth, or latency. Use real APIs only (DXGI, Windows PDH, NVML, Intel RAPL / AMD SMU, D3D12 Timestamp Queries, PresentMon/FrameView).
2. **Phase-Gated Execution:** Execute strictly in order (`Phase 0` $\rightarrow$ `Phase 6`). Do not advance until the current Verification Gate passes with real logs.
3. **Determinism:** Use fixed camera flythrough paths, fixed seeds, $\ge 120$ warm-up frames, $\ge 1000$ measured frames, and $N \ge 5$ repeated runs (report mean, std, $95\%$ CI).
4. **License Hygiene:** Isolate proprietary SDKs (e.g., NVIDIA RTXNTC) in `third_party/` with original licenses. Build core cross-vendor scheduling and inference on open standards (DX12 Agility SDK, HLSL SM 6.x, AMD MiniDXNN, DirectML / OpenVINO / ONNX Runtime).

## 3. Repository Layout

```text
PA-AHNTS/
├── configs/
│   ├── hardware_profile.json     # Auto-generated in Phase 0
│   ├── camera_paths/             # Deterministic flythrough scripts
│   └── experiments/              # Phase & ablation JSON configs
├── third_party/                  # RTXNTC, RTXTS, Agility SDK, MiniDXNN, ONNXRuntime, FLIP
├── tools/
│   ├── env_probe/                # Phase 0 hardware/driver/NPU probe
│   └── asset_gen/                # 4K/8K PBR stress assets & NTC latents
├── src/
│   ├── core/                     # DX12 device, multi-adapter context, sync primitives
│   ├── telemetry/                # Unified logger (VRAM, FrameTime, Power, PCIe, Trace)
│   ├── baselines/                # Phase 1: BCn, SFS, NTC-on-Sample, NTC-on-Feedback
│   ├── ram_tier/                 # Phase 2: RAM-tiered latent pool, LRU, async Copy Queue
│   ├── hetero_bench/             # Phase 3: CPU / NPU / iGPU / dGPU decode micro-benchmarks
│   └── scheduler/                # Phase 4: PA-AHNTS power-aware adaptive scheduler
├── scripts/
│   ├── run_phase.py              # Automated runner & regression harness
│   └── analysis/                 # Phase 6: Stats, break-even plots, LaTeX tables
├── results/                      # Raw CSV/JSON, screenshots, plots (gitignored)
├── CMakeLists.txt                # Windows x64 (MSVC + CMake)
└── EXPERIMENT_LOG.md             # Concise English execution & diagnosis log
```

## 4. Phased Roadmap & Verification Gates

### Phase 0: Hardware & Environment Probe (Pre-flight)
* **Tasks:** Build and run `tools/env_probe/probe_hardware.py` (+ lightweight C++ DXGI/DirectML probe). Detect OS, CPU, RAM, dGPU (confirm 8GB VRAM), PCIe link gen/width, iGPU, NPU model & available EPs (DirectML / OpenVINO / QNN), MSVC/CMake, Agility SDK, NVML, and admin privileges for CPU/SoC power counters.
* **Gate 0:** Generate `configs/hardware_profile.json`. Log readiness/missing deps in `EXPERIMENT_LOG.md`. Pause and report in Chinese for user confirmation.

### Phase 1: Reproduce Baselines (1/6)
* **Tasks:** Build a controllable DX12 PBR streaming testbed (e.g., Donut/NVRHI + RTXTS or minimal DX12 harness) with 4K/8K PBR assets pushing unoptimized VRAM to $7.5\text{--}9.0\text{ GB}$. Implement 4 baselines:
  * `B0: BCn-Full` (resident BCn)
  * `B1: SFS-BCn` (Sampler Feedback Streaming + BCn)
  * `B2: NTC-on-Sample` (on-the-fly shader decode)
  * `B3: NTC-on-Feedback` (feedback-driven async decode to BCn cache)
  Integrate `src/telemetry/` for frame time, VRAM, power, and offline PSNR/FLIP.
* **Gate 1:** All 4 baselines run stably (no TDR) on identical camera paths; output `results/phase1/{frame_metrics,memory_power,quality_metrics}.csv`.

### Phase 2: RAM-Tiered Latent Pool (2/6)
* **Tasks:** Offload cold/warm NTC latent grids from VRAM to System RAM (`VRAM L1` $\leftrightarrow$ `RAM L2` $\leftrightarrow$ `SSD L3`) using pinned memory and async DX12 Copy Queue with LRU/feedback-weighted eviction. Measure actual Local VRAM savings (check $\ge 2\text{ GB}$), PCIe bandwidth, upload latency, and P99 jitter.
* **Gate 2:** Output VRAM savings curves, PCIe latency histograms, and P99 comparisons across VRAM caps ($4\text{ GB}, 5\text{ GB}, 6\text{ GB}$) in `results/phase2/`.

### Phase 3: Heterogeneous Decode Micro-Benchmark (3/6)
* **Tasks:** Benchmark real NTC MLP decoding (configurable hidden dims, FP16/INT8) standalone on 4 backends: **dGPU** (DX12 Compute/Matrix), **iGPU** (DirectML/DX12), **NPU** (ONNX Runtime DirectML/OpenVINO/QNN EP), **CPU** (AVX/ORT). Sweep batch sizes ($1, 4, 16, 64, 256$ tiles). Measure:
  1. Throughput ($\text{Mtexels/s}$, $\text{Tiles/ms}$) & end-to-end latency (incl. API dispatch).
  2. **Transfer Cost:** Host pre-decode + PCIe upload (BCn/raw) vs. PCIe latent upload + dGPU decode.
  3. **Power Contention:** Wake NPU/iGPU while dGPU is under heavy load; measure $\Delta W$ and Dynamic Boost clock drop.
  4. **Quantization Loss:** NPU INT8 vs. dGPU FP16 PSNR/FLIP delta.
* **Gate 3:** Output `results/phase3/hetero_benchmark_matrix.csv` and break-even boundary analysis.

### Phase 4: Power-Aware Adaptive Scheduler (4/6)
* **Tasks:** Implement `src/scheduler/` using Phase 3 cost models:
  * **Dynamic Deadline Classification:** Urgent visible tiles $\rightarrow$ dGPU; Non-urgent prefetch tiles $\rightarrow$ heterogeneous queue.
  * **Predictive Prefetching:** Camera velocity/frustum lookahead to pre-load RAM latents and batch-decode in static buckets.
  * **Power-Aware Hysteresis Gating:** Monitor dGPU headroom, PCIe queue depth, and SoC/dGPU power. Use dual-threshold hysteresis to enable NPU/iGPU only when net gain $> 0$ without tripping Dynamic Boost power stealing; otherwise gracefully fall back to RAM-tiered dGPU decode.
* **Gate 4:** End-to-end run across static $\rightarrow$ fast-pan $\rightarrow$ scene-cut workloads. Output `scheduler_trace.csv` and direct comparison table against the 4 core hypotheses.

### Phase 5: Cross-Platform & High-TOPS NPU Extension (5/6)
* **Tasks:** Decouple backends behind a plugin interface (`IDecodeBackend`) driven strictly by `hardware_profile.json` (no hardcoded device indices). Provide one-command portable benchmarking (`scripts/run_phase.py --all`) for migrating to higher-TOPS NPU laptops (e.g., Ryzen AI / Panther Lake).
* **Gate 5:** Verified portable execution from a clean config directory.

### Phase 6: Ablations & Paper Artifacts (6/6)
* **Tasks:** Run full ablation matrix ($N \ge 5$ runs each): `A0: BCn-Full`, `A1: SFS-BCn`, `A2: NTC-on-Feedback`, `A3: NTC + RAM-Tiered Latents`, `A4: A3 + Static Offload`, `A5: Full PA-AHNTS`. Build `scripts/analysis/` to auto-generate vector plots (PDF/SVG: VRAM over time, FrameTime CDF/P99, power breakdown, break-even heatmap, FLIP/PSNR), LaTeX tables, and `README_ARTIFACT.md`.
* **Gate 6:** 100% reproducible figures/tables from raw CSVs via a single command.

## 5. Telemetry CSV Schemas

* `frame_metrics.csv`: `run_id,config_name,frame_idx,timestamp_ms,cpu_frame_ms,gpu_frame_ms,present_ms,pcie_upload_mb,tiles_decoded_dgpu,tiles_decoded_npu,tiles_decoded_igpu,tiles_decoded_cpu`
* `memory_power.csv`: `run_id,config_name,sample_idx,timestamp_ms,local_vram_mb,nonlocal_vram_mb,sys_ram_latent_mb,dgpu_power_w,cpu_pkg_power_w,npu_power_w,total_est_power_w,dgpu_clock_mhz,dgpu_temp_c`
* `quality_metrics.csv`: `config_name,camera_checkpoint_id,psnr_db,flip_mean,flip_p99,reference_config`
* `scheduler_trace.csv`: `frame_idx,vram_pressure_ratio,power_headroom_w,pcie_queue_depth,gate_state,urgent_tiles,prefetch_tiles,active_backend`

## 6. Immediate Next Action

Do **NOT** scaffold all phases at once. Execute only:
1. Initialize the repo layout, base `CMakeLists.txt`, `.gitignore`, and `EXPERIMENT_LOG.md`.
2. Implement and run **Phase 0 (`tools/env_probe/`)** on this machine.
3. **Report back to the user in Simplified Chinese** with the real `configs/hardware_profile.json` findings and your concrete Phase 1 tech-stack plan, then wait for confirmation.