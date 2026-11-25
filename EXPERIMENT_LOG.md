# PA-AHNTS Experiment & Engineering Log

## Phase 0 (Gate 0) — Hardware & Environment Probe
- **Timestamp (UTC):** 2026-11-24T13:02:24Z
- **Status:** `PASSED` (`configs/hardware_profile.json` generated from live hardware/OS APIs; zero mocked metrics)

### 1. Host Platform & Compute Topology
- **System:** LENOVO `83F3`, Microsoft Windows 11 Pro (`10.0.26300`, 64-bit)
- **CPU:** `Intel(R) Core(TM) Ultra 7 255HX` (20 cores / 20 threads)
- **System RAM:** `31.43 GiB` (`33,752,997,888` bytes; `17.36 GiB` available at probe time)
- **dGPU (Adapter 0):** `NVIDIA GeForce RTX 5060 Laptop GPU` (`0x10DE:0x2D59`, driver `621.35`, CUDA CC `12.0` Blackwell)
  - Total VRAM (NVML): `8151.0 MiB` (`8,546,942,976` bytes) -> **confirms 8GB target laptop envelope**
  - DXGI Local Memory Budget: `7131.0 MiB`; Non-Local Shared Budget: `17579.94 MiB`
  - PCIe Topology: Max `Gen 5 x16`, active laptop link width `x8` (`Gen 5 x8` under active query; scales down to `Gen 2` at idle)
  - D3D12 Capabilities: Feature Level `12_2`, Shader Model `6.8`, Tiled Resources `Tier 4`, Sampler Feedback `Tier 0.9 (90)`, Native 16-bit Shader Ops `true`, Direct & Copy Queue Timestamps `1,000,000,000 Hz`
  - DirectML: `DMLCreateDevice` succeeded (`0x00000000`, `DirectML.dll 1.15.6`)
- **iGPU (Adapter 1):** `Intel(R) Graphics` (`0x8086:0x7D67`, driver `32.0.101.8860`)
  - D3D12 Feature Level `12_2`, SM `6.8`, UMA `true`, Native 16-bit Shader Ops `true`, DirectML `DMLCreateDevice` succeeded (`0x00000000`), Timestamp Freq `19.2 MHz`
- **NPU:** `Intel(R) AI Boost` (`PCI\VEN_8086&DEV_AD1D`, driver `32.0.100.4841`, PnP Status `OK`)

### 2. Telemetry Subsystem Verification
- **NVML (`nvml.dll 13.621.35`):** Live readings verified without elevation (`25.223 W`, default limit `50.0 W`, enforced limit `115.0 W`, `52 C`, GFX `1395 MHz` / Mem `12001 MHz`).
- **Windows PDH (`pdh.dll`) RAPL Energy Meter:** Verified **without Administrator privileges** (`is_admin: false`):
  - `\Energy Meter(RAPL_Package0_PKG)\Power` = `11.738 W`
  - `\Energy Meter(RAPL_Package0_PP0)\Power` = `5.255 W`
  - GPU PDH counter sets (`GPU Engine`, `GPU Process Memory`, `GPU Local Adapter Memory`, `GPU Non Local Adapter Memory`) all present.

### 3. Toolchain & Dependency Audit
- **Present:** `CMake 4.4.4`, `Ninja 1.13.2`, `MinGW-w64 g++ 16.2.0`, `Python 3.12.10`, `PyTorch 2.9.1+cu130` (CUDA `13.0` active).
- **Missing / Needed for Phases 1–3:**
  - `MSVC (VS 2022 Build Tools cl.exe)` & `Windows SDK` (`dxc.exe` / `dxcompiler.dll` for HLSL SM 6.6+ FP16 & Sampler Feedback compilation)
  - `third_party/agility_sdk` (for consistent DX12 Sampler Feedback Tier 1.0 & Enhanced Barriers)
  - `flip-evaluator` (Python package for NVIDIA FLIP visual quality evaluation in Phase 1)
  - `onnxruntime-directml` and/or `openvino` (for Phase 3 NPU/iGPU/CPU decode micro-benchmarks)

## Phase 1 (Gate 1) — Baseline Reproduction & Telemetry Verification
- **Timestamp (UTC):** 2026-11-24T13:58:48Z
- **Status:** `PASSED` (`results/phase1/{frame_metrics,memory_power,quality_metrics}.csv`, `results/phase1/phase1_summary.json`)
- **Testbed:** Native C++17 DX12 Ultimate (`1920x1080`, `ALLOW_TEARING`, Agility SDK `1.620.2`, SM 6.6 HLSL via `dxc 1.9.2609.5`, Sampler Feedback `Tier 0.9` `{256,256,1}`, `16`-bay architectural gallery, `32` `8192x8192` PBR materials = `8,192 MiB` raw BC7/BC5 footprint).
- **Protocol:** `N = 5` repeated runs per baseline, `120` warm-up + `1000` measured frames per run (`20,000` measured frame rows; `624` 20Hz NVML + PDH-RAPL + DXGI samples; `5` deterministic camera checkpoints evaluated with PSNR & NVIDIA `flip-evaluator 1.7`).

### Phase 1 Summary (`N = 5`, mean ± std)
| Config | Avg FPS | CPU Frame (ms) | GPU Frame (ms) | P99 CPU (ms) | Peak Local VRAM (MB) | Non-Local VRAM (MB) | PCIe Upload (MB) | dGPU Decoded Tiles | Avg dGPU / Total Power (W) | PSNR (dB) | FLIP (mean / max P99) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_BCn_Full` | `2769.9 ± 113.1` | `0.36 ± 0.01` | `0.33 ± 0.01` | `1.06 ± 0.06` | `8247.6 ± 0.4` | `50.3 ± 0.1` | `0.0` | `0` | `22.17 / 40.48` | `99.00` (ref) | `0.0000 / 0.0000` |
| `B1_SFS_BCn` | `781.9 ± 71.6` | `1.29 ± 0.11` | `0.44 ± 0.00` | `9.69 ± 1.43` | `1605.6 ± 0.0` | `80.3 ± 0.4` | `481.7` | `0` | `36.18 / 60.54` | `68.61` | `0.0002 / 0.0194` |
| `B2_NTC_OnSample` | `342.7 ± 5.5` | `2.92 ± 0.05` | `2.88 ± 0.04` | `4.62 ± 0.07` | `3126.7 ± 0.0` | `50.1 ± 0.0` | `0.0` | `0` | `81.80 / 94.50` | `32.45` | `0.0510 / 0.4446` |
| `B3_NTC_OnFeedback` | `537.8 ± 39.5` | `1.87 ± 0.13` | `0.53 ± 0.00` | `14.23 ± 1.63` | `4762.3 ± 0.2` | `54.7 ± 0.3` | `0.0` | `2569` | `39.60 / 58.36` | `31.62` | `0.0541 / 0.5064` |

### Key Empirical & Physical Findings
1. **Multi-Scale NTC Fix:** Initial run yielded ~23–24 dB PSNR because the NTC MLP was trained only at `mip=0` (`lod_atten=1.0`) and `G1` (`1024x1024`) had an unaligned mip offset in `ntc_decode_bcn.cs.hlsl`. Switching to multi-scale LOD-aware training (`mips 0..4`) and aligning `G1` mip bias (`max(0, lod - 2)`) improved rendered quality to `32.45 dB` (`B2`, FLIP `0.0510`) and `31.62 dB` (`B3`, FLIP `0.0541`).
2. **NTC-on-Sample Power/Shader Penalty (`B2`):** Per-pixel `16->32->32->8` FP16 MLP inference inflates GPU frame time `8.7x` (`0.33 -> 2.88 ms`) and drives dGPU power from `22.2 W` to `81.8 W` (`94.5 W` total).
3. **NTC-on-Feedback Bottleneck (`B3`):** Moving NTC decode to Async Compute on Sampler Feedback slashes GPU frame time back to `0.53 ms` and total power to `58.36 W`, **but still consumes `4,762.3 MB` Local VRAM** (`3,072 MB` VRAM-resident NTC latents + `1,536 MB` BCn tile cache) and incurs `14.23 ms` P99 CPU stalls from synchronous `UpdateTileMappings` + `fb_readback` fences. Offloading cold/warm NTC latents to System RAM (`Phase 2`) directly targets this `~3.0 GB` latent footprint and P99 jitter.

## Phase 2 (Gate 2) — RAM-Tiered Latent Pool & CSV Visualization Suite
- **Timestamp (UTC):** 2026-11-24T14:12:57Z
- **Status:** `PASSED` (`results/phase2/{frame_metrics,memory_power,pcie_transfer_trace,quality_metrics}.csv`, `results/phase2/phase2_summary.json`, `results/phase{1,2}/plots/*.{png,pdf}`)
- **Architecture:** `VRAM L1` (`G0 4096x4096 RGBA8` + `G1 1024x1024 RGBA16F` Reserved Resources with Feedback-Weighted LRU) $\leftrightarrow$ `RAM L2` (`VirtualAlloc`/`VirtualLock` pinned 64KB latent arena) $\leftrightarrow$ `SSD L3` (`CreateFileW` on-demand reads from `pbr_ntc_pack_v1.bin`); non-blocking 3-stage `Copy -> Compute -> Direct` pipelining with batched per-resource `UpdateTileMappings` and `1 GHz` Copy Queue hardware timestamps.

### Phase 2 Summary (`N = 5`, mean ± std)
| Config | L1 / L2 (MiB) | Avg FPS | CPU / GPU (ms) | P99 CPU (ms) | Peak Local VRAM (MB) | Savings vs B0 / B3 (MB) | Peak RAM L2 (MB) | PCIe Upload (MB) | Copy P50/P99 (µs) | Mean BW (GB/s) | Tot Power (W) | PSNR / FLIP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R1_Cap2.2GB_L1_256MB` | `256 / 2560` | `913.9 ± 27.4` | `1.09 / 0.58` | `5.79 ± 0.28` | `1957.3 ± 0.0` | `6290.4 / 2805.1` | `320.6` | `249.9` | `35.8 / 229.0` | `24.74` | `61.65` | `29.19 / 0.0779` |
| `R2_Cap3.0GB_L1_512MB` | `512 / 2560` | `902.7 ± 12.9` | `1.11 / 0.58` | `5.53 ± 0.24` | `2213.3 ± 0.0` | `6034.4 / 2549.1` | `317.2` | `246.1` | `35.0 / 221.5` | `24.55` | `60.46` | `29.19 / 0.0779` |
| `R3_Cap4.0GB_L1_1024MB` | `1024 / 2560` | `907.0 ± 10.3` | `1.10 / 0.59` | `5.54 ± 0.11` | `2725.3 ± 0.0` | `5522.4 / 2037.1` | `322.1` | `249.2` | `36.3 / 228.1` | `24.64` | `61.08` | `29.19 / 0.0779` |
| `R4_Cap5.0GB_L1_1792MB` | `1792 / 2560` | `897.4 ± 21.7` | `1.11 / 0.59` | `5.79 ± 0.08` | `3493.3 ± 0.0` | `4754.4 / 1269.1` | `320.8` | `249.9` | `36.3 / 228.9` | `24.46` | `64.30` | `29.19 / 0.0779` |
| `R5_Cap6.0GB_L1_2560MB` | `2560 / 2560` | `893.6 ± 9.9` | `1.12 / 0.59` | `5.87 ± 0.22` | `4261.0 ± 0.3` | `3986.6 / 501.3` | `319.0` | `248.7` | `35.3 / 225.9` | `24.61` | `62.09` | `29.19 / 0.0779` |

### Key Empirical Findings
1. **VRAM Savings Target Exceeded ($\ge 2\text{ GB}$):** `R1` (`256 MiB` L1), `R2` (`512 MiB` L1), and `R3` (`1024 MiB` L1) cut Peak Local VRAM to `1.91 GiB`, `2.16 GiB`, and `2.66 GiB`, saving **`2.74 GiB` (`2,805 MB`)**, **`2.49 GiB` (`2,549 MB`)**, and **`1.99 GiB` (`2,037 MB`)** vs `B3_NTC_OnFeedback` (`4,762.3 MB`), and up to **`6.14 GiB` (`6,290 MB`)** vs `B0_BCn_Full` (`8,247.6 MB`).
2. **P99 Tail Latency & FPS Improvement:** Batched `UpdateTileMappings` + non-blocking 3-stage pipelining boosted throughput to **`894–914 FPS`** (`+66–70%` over `B3`'s `537.8 FPS`) and slashed P99 CPU frame time from **`14.23 ms` (`B3`)** and **`9.69 ms` (`B1`)** down to **`5.53–5.87 ms`** (`2.5x` reduction vs `B3`).
3. **PCIe Gen5 x8 Efficiency:** Latent tile streaming required only **`246–250 MB`** total PCIe upload over `1000` frames (`48%` lower than `B1_SFS_BCn`'s `481.7 MB`), with Copy Queue DMA median latency `35–36 us` (`P99 ~221–229 us`) at **`24.5–24.7 GB/s`** sustained payload bandwidth.
4. **Quality Note:** Rendered quality is uniform at `29.19 dB` PSNR (`FLIP 0.0779`) across `R1..R5` (within `2.4 dB` of `B3`'s `31.62 dB`, due to 1-texel bilinear boundary taps at unmapped `G0` tile edges and 1-frame extra pipeline latency).

## Phase 3 (Gate 3) — Heterogeneous Decode Micro-Benchmark & Break-Even Analysis
- **Timestamp (UTC):** 2026-11-24T14:46:17Z
- **Status:** `PASSED` (`results/phase3/{hetero_benchmark_matrix,transfer_cost_matrix,power_contention_trace,quantization_quality}.csv`, `results/phase3/break_even_analysis.json`, `results/phase3/plots/*.{png,pdf}`)
- **Devices Verified:**
  - **dGPU:** `NVIDIA GeForce RTX 5060 Laptop GPU` (DX12 SM 6.6 Compute + Copy Queue over `PCIe Gen5 x8`, plus ORT `DmlExecutionProvider` id=0)
  - **iGPU:** `Intel(R) Graphics` (ORT `DmlExecutionProvider` id=1 & OpenVINO `GPU.0`)
  - **NPU:** `Intel(R) AI Boost` (OpenVINO `NPU` INT8 QDQ & FP16 verified via `EXECUTION_DEVICES=NPU`)
  - **CPU:** `Intel(R) Core(TM) Ultra 7 255HX` (OpenVINO `CPU` & ORT `CPUExecutionProvider`)
- **Protocol:** `N = 5` runs per configuration; real trained `16->32->32->8` NTC MLPs (`H=16/32/64`) exported as static `1x1 Conv2D` ONNX models (`FP32`, `FP16`, symmetric MinMax QDQ `INT8`) across batch sizes `B in {1, 4, 16, 64, 256}` `256x256` tiles.

### 1. Standalone Decode Throughput & Latency (`N = 5`, `H = 32`, `B = 1 / 16 / 256` Tiles)
| Backend | Precision | B=1 (ms / Mtex/s) | B=16 (ms / Mtex/s) | B=256 (ms / Mtex/s) | B=16 Eff GOP/s |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dGPU_DX12_Compute_FP16_BCn` | `FP16` | `0.363 / 180.8` | `3.430 / 305.8` | `47.779 / 351.3` | `1155.6` |
| `dGPU_DX12_Compute_FP16_Raw` | `FP16` | `0.353 / 185.8` | `3.267 / 320.9` | `46.950 / 357.4` | `1219.0` |
| `dGPU_ORT_DML_FP16` | `FP16` | `1.328 / 49.7` | `16.649 / 63.1` | `213.129 / 78.8` | `226.1` |
| `CPU_ORT_INT8` | `INT8` | `1.092 / 60.3` | `21.315 / 49.2` | `303.174 / 55.6` | `176.5` |
| `iGPU_ORT_DML_FP16` | `FP16` | `1.796 / 36.7` | `22.758 / 46.1` | `330.117 / 50.8` | `165.2` |
| `iGPU_OpenVINO_INT8` | `INT8` | `2.577 / 25.5` | `35.381 / 29.7` | `862.917 / 19.4` | `106.5` |
| `CPU_OpenVINO_INT8` | `INT8` | `2.562 / 25.6` | `32.390 / 32.4` | `524.533 / 32.0` | `116.2` |
| `NPU_OpenVINO_INT8` | `INT8` | `3.201 / 20.5` | `47.417 / 22.2` | `659.823 / 25.4` | `79.7` |
| `NPU_OpenVINO_FP16` | `FP16` | `7.543 / 8.7` | `114.860 / 9.1` | `1530.844 / 11.0` | `32.7` |

- **Hidden-Dim Scaling (`B = 16`, `H = 16 / 32 / 64`):** `dGPU_DX12_Compute` scales steeply with ALU load (`1985.4 / 305.8 / 148.0 Mtex/s`), whereas `NPU_OpenVINO_INT8` stays nearly flat (`23.30 / 22.23 / 21.44 Mtex/s` -> `29.8 / 79.7 / 241.5 GOP/s`) and `iGPU_ORT_DML_FP16` shifts from `54.4 -> 46.1 -> 38.6 Mtex/s` (`69.6 -> 165.2 -> 434.9 GOP/s`), proving the NPU is strictly **1x1-Conv activation memory-bandwidth bound** (~`0.61%` of 13 TOPS MAC utilization at `H=32`) rather than compute-bound.

### 2. End-to-End Transfer vs Compute & Physical Break-Even Boundary (`B = 16`, Gen5 x8)
- **Measured Copy Queue PCIe Gen5 x8 Bandwidth (`B = 16 / 256`):** `26.21 / 27.39 GB/s` (`72 KB` Latents = `0.045 ms` at `B=16`), `28.31 / 28.62 GB/s` (`192 KB` BCn = `0.111 ms`), `28.59 / 28.65 GB/s` (`512 KB` Raw = `0.293 ms` + `0.022 ms` dGPU BCn Pack).
- **Falsification Finding (Uncontended Offload Negative Net Gain):** Under uncontended dGPU conditions, compact **`72 KB` latent upload + RTX 5060 DX12 decode (`3.475 ms` for `16` tiles)** outperforms **`iGPU_ORT_DML_FP16 + Raw PCIe + Pack` (`42.87 ms`, or `23.07 ms` excl. NumPy gather)** by `6.6x–12.3x` and **`NPU_INT8 + Raw PCIe + Pack` (`67.53 ms`, or `47.73 ms` excl. gather)** by `13.7x–19.4x`. Overcoming `PCIe Gen5 x8` + RTX 5060 uncontended latency at `B = 16` requires **$\ge 1.19\text{ effective TOPS}$ (`Mtex/s >= 332`)** on `H = 32` `1x1 Conv2D` — OR a **dGPU compute-queue / frame-time congestion window $\ge 44.3\text{–}64.1\text{ ms}$** (i.e., strictly non-urgent speculative prefetching multiple frames ahead of camera arrival).

### 3. Power Contention & Dynamic Boost Telemetry (`Idle -> Solo -> +NPU -> +iGPU -> +CPU -> Cooldown`)
- **`dGPU_Solo` Steady-State:** `dGPU = 89.40 W`, `CPU PKG = 10.69 W`, `Total = 100.09 W`, `SM Clock = 2797.2 MHz`, `2277.0 tiles/s`.
- **`+NPU INT8`:** `dGPU = 90.14 W` (`+0.74 W`), `CPU PKG = 15.52 W` (`+4.82 W`), `Total = 105.65 W` (`+5.56 W`), `SM Clock = 2797.0 MHz` (`-0.2 MHz`), **`dGPU throughput = -0.39%`** — virtually zero Dynamic Boost stealing or dGPU interference!
- **`+iGPU FP16`:** `dGPU = 90.90 W`, `CPU PKG = 21.20 W` (`+10.51 W`, `PP1 = 3.80 W`), `Total = 112.10 W` (`+12.01 W`), `SM Clock = 2797.0 MHz`, `dGPU throughput = -1.38%`.
- **`+CPU INT8`:** `CPU PKG` surges to `25.98 W` (`+15.29 W`, `Total = 115.71 W`), degrading sustained dGPU compute throughput by **`-9.51%` (`2277 -> 2060 tiles/s`)** due to host dispatch & thermal/power envelope contention.

### 4. Quantization Fidelity (`Intel AI Boost NPU INT8` vs `FP16`)
- Across **all 32 materials (`512x512`)**: Mean 8-ch PSNR = **`50.15 dB`**, Albedo PSNR = **`51.79 dB`**, Mean Albedo NVIDIA FLIP = **`0.0205`** (`max P99 = 0.0657`).
- Across **all 5 DX12 1080p Gallery Checkpoints (`ckpt_00..04`)**: Mean Rendered PSNR (`NPU_INT8` vs `FP16`) = **`48.47 dB`**, Mean FLIP = **`0.0187`**, confirming `INT8` introduces negligible perceptual loss and is safe whenever Phase 4's hysteresis scheduler gates NPU offload on.


## Phase 2 Scientific Reinforcement Gate — Memory Accounting & 2x2 Factorial Ablation
- **Timestamp (UTC):** 2026-11-25T13:32:08Z
- **Status:** `PASSED` (`results/phase2_reinforced/{frame_extended_metrics,resource_memory_breakdown,historical_vs_reinforced_memory_table,pcie_transfer_trace,quality_metrics}.csv`, `results/phase2_reinforced/phase2_reinforced_summary.json`, `PHASE_2_REINFORCEMENT_REPORT.md`)
- **Bugfixes & Parity Gate:** Resolved `mip = 4` modular wraparound halo omission (`floor_div_i`), `CopyQueue -> DirectQueue` `vram_l1_staging` fence race, and Stage B `DirectQueue` vs `ComputeQueue` race. Live-streamed checkpoints achieved exact parity between VRAM-resident and RAM-tiered NTC (`C0/C1 = 31.63 dB`, `C2_VRAM_Async = 31.49 dB`, `C3_RAM_R2_Async = 31.40 dB`, `FLIP 0.0540–0.0559`, bit-exact `29.4059 dB` at `ckpt_00`).
- **2x2 Factorial Isolation (`N = 5`, `R2 = 512 MiB` L1):**
  - `Async Batching` (`C0 -> C2` / `C1_R2 -> C3_R2`): `+607.7 / +572.7` Present FPS (`+189.5% / +276.7%`), P99 display `-11.26 / -23.28 ms`, Package Energy `-0.123 / -0.186 J/frm`.
  - `RAM Tiering` (`C0 -> C1_R2` / `C2 -> C3_R2`): **`-2,525.4 MiB` (`-2.466 GiB`) DXGI Local VRAM**, at a throughput cost of `-113.6 FPS` (`-35.4%`) Sync / `-148.6 FPS` (`-16.0%`) Async and `+0.65 ms` Async P99 (`6.56 -> 7.21 ms`).
  - **Fair Async SFS-BCn Comparison (`B1_SFS_BCn_Async_RamCache`):** Reaches `1411.6 FPS` (`242.7` 1% Low), `3.42 ms` P99, `1.568 GiB` VRAM, `557.2 MiB` RAM cache, `0.0473 J/frm`, `67.51 dB` PSNR — demonstrating pre-encoded BCn wins when active BCn tiles fit in host RAM, whereas NTC + RAM tiering uniquely delivers `185x` asset-pack storage compression (`44.24 MiB` vs `8,192 MiB`).
- **Memory Accounting Proof:** Zero double-counting (`explicit sum + ~98–99.7 MiB driver/page-table overhead = DXGI Local VRAM`); confirmed $\text{VRAM} = 1.66\text{ GiB} + L1$ (`1.685 GiB + L1` with `32 MiB` atomic halo staging) and decomposed `2.74 GiB` VRAM savings into sparse demand-paging elimination (`~2.45–2.69 GiB`) + active L1→L2 eviction (`7,578` evictions at `128 MiB`, `4,963` at `256 MiB`, `531` at `512 MiB`). `GetTimestampFrequency()` verified `1,000,000,000 Hz`.

## Phase 3 Scientific Reinforcement Gate — Audit, Host Gather AVX2, Congestion & Scheduler Cost Model
- **Timestamp (UTC):** 2026-11-25T14:05:12Z
- **Status:** `PASSED` (`results/phase3_reinforced/*`, `scheduler_cost_model.json`, `configs/scheduler_cost_model.json`, `PHASE_3_REINFORCEMENT_REPORT.md`)
- **Backend Audit & Exact Op Count:** Verified `0` unexpected CPU fallback compute nodes via OpenVINO `get_runtime_model()` (`NPU`, `GPU.0`, `CPU`) and ORT profiler (`DmlExecutionProvider` fused nodes); standardized `1,792 MACs/texel` / `3,584 Ops/texel` (`3.758096384 GOP` at `H=32, B=16`) separating `kernel_gops` (`866.6 BCn / 1167.1 Raw`) from `e2e_infer_gops` (`739.5 BCn / 1026.2 Raw`).
- **Host Gather Optimization (`pahnts_host_gather.dll`):** Native C++ AVX2+FMA+F16C + Win32 thread pool cut `B=16` prepare time **`10.5x–12.3x`** (`21.68 ms` NumPy -> **`1.77 ms` FP16** [`93.24 dB`] / **`2.06 ms` FP32** [`126.52 dB`]), dropping `NPU_INT8` E2E to `55.94 ms` (`53.56 ms` ovlp) and `iGPU_ORT_DML` E2E to `27.70 ms` (`25.61 ms` ovlp).
- **Measured Congestion Break-Even & Energy Contention (`N = 5`):** Hardware `gpu_congestion_bench` confirmed empirical offload crossover at **`~22.5 ms` (`iGPU`)** and **`~50.8 ms` (`NPU`)** queued dGPU load. Thermally matched (`>= 66 °C`) `H = 32` energy trials **refuted standalone NPU energy superiority** (`dGPU 20.72` vs `iGPU 65.01` vs `CPU 107.06` vs `NPU 132.80 mJ/tile` total package due to `34.4 W` idle baseline), while confirming concurrent `dGPU + iGPU` (`19.12 mJ/tile`, `5853.5 tps`, `-0.27%` dGPU) and `dGPU + NPU` (`19.61 mJ/tile`, `5450.9 tps`, `-0.26%` dGPU) improve aggregate throughput and energy efficiency without stealing Dynamic Boost (`Core-0` pinning turns `-0.72%` CPU contention into `+2.20%`).
- **Scheduler Cost Model Validation (`17` held-out points: `B in {8,32,96}` + congestion `{15,30 ms}`):** Global OLS achieved `R^2 = 0.93745`, `MAE = 21.01 ms`, `MAPE = 23.98%` (`5.6%–10.6%` on congestion); piecewise-linear interpolation over the stored `observed_batches` anchors cuts held-out batch `MAPE` to `15.16%`.
